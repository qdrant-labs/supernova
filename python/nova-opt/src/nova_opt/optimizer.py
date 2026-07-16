"""The tuner's main loop — the experiment planner itself.

Each iteration: sample candidates (biased toward search-level variations of
artifacts that already exist, with a uniform remainder so fresh artifacts
keep being proposed), join them with the workload's dataset/query
statistics, predict recall feasibility (offline classifier, recalibrated
online against this run's measured recalls both globally — `online.py` — and
spatially, once there's enough evidence — `recall_online.py`), price each
candidate against the artifact cache, and score:

- cheap candidates by their own EHVI * W_R / cost^gamma_t
- expensive candidates by the *batch* they unlock — the candidate plus the
  cheap search-level children the scheduler would evaluate after the build —
  because that is what the build actually buys

gamma_t is cost-cooled: full penalty early, annealed toward zero as the
budget depletes. Before the surrogate has enough points, scoring falls back
to W_R / cost^gamma times an encoded-space diversity factor (a cheap
space-filling init).

The winner is evaluated by building only its missing artifact levels; an
expensive build is then amortized by measuring its cheap children. Every
measurement flows back into the GP surrogates (feasibility-marked for the
constrained Pareto front), the cost model, the artifact cache, the online
recall recalibrator, and the recall GP.

The `strategy` knob degrades this into the comparison baselines (random
sweep, vanilla BO, recall-pruned BO, cost-only BO) — see acquisition.py.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nova_opt import acquisition
from nova_opt.cost import ArtifactCache, CostModel
from nova_opt.evaluate import EvalError, Evaluator, SearchOutcome
from nova_opt.online import OnlineRecalibrator
from nova_opt.recall import THRESHOLDS, RecallClassifier, RecallPrediction
from nova_opt.recall_online import RecallSurrogate
from nova_opt.space import Candidate, ConfigSpace, config_features
from nova_opt.surrogate import MoboSurrogate, Observation

log = logging.getLogger("nova_opt")


@dataclass
class OptSettings:
    target_recall: float = 0.90
    strategy: acquisition.Strategy = "full"
    budget_seconds: float = 3600.0
    max_evaluations: int = 200
    gamma: float = 0.5
    beta: float = 0.1
    prune_min_probability: float = 0.3
    n_candidates: int = 256
    seed: int = 0
    # cost-cooling: gamma_t = gamma * remaining/total (see acquisition.py)
    cost_cooling: bool = True
    # proposal biasing toward existing artifacts' cheap region
    bias_fraction: float = 0.5
    # shrinkage prior of the online recall recalibration
    online_prior_strength: float = 4.0
    # measured recalls (with feature variation) needed before the run-local
    # recall GP is consulted at all (see recall_online.py)
    recall_gp_min_observations: int = 5
    # scheduler: cheap search-level children after an expensive build
    children_ef_search: tuple[int, ...] = (16, 32, 64, 128, 256)
    children_batch_size: tuple[int, ...] = (1, 8, 32, 128)
    max_children: int = 8
    # which levels count as "expensive enough to amortize"
    amortize_levels: tuple[str, ...] = ("layout", "index", "quant")


@dataclass
class Trial:
    """Everything recorded about one evaluated search point — enough to
    retrain the recall model and re-analyze the planner afterwards."""

    iteration: int
    candidate: Candidate
    selected: bool  # False for amortization children
    ok: bool
    error: str | None
    # planner state at selection time
    probs: dict[float, float]
    confidence: dict[float, float]
    recall_weight: float
    ehvi: float | None
    predicted_cost: float
    score: float | None
    built_levels: tuple[str, ...]
    # measurements
    build_seconds: dict[str, float] = field(default_factory=dict)
    outcome: SearchOutcome | None = None
    feasible: bool | None = None

    def row(self, workload: dict[str, Any], stats_meta: dict[str, Any]) -> dict:
        r: dict[str, Any] = {
            "iteration": self.iteration,
            "selected": self.selected,
            "ok": self.ok,
            "error": self.error,
            "recall_weight": self.recall_weight,
            "ehvi": self.ehvi,
            "predicted_cost": self.predicted_cost,
            "score": self.score,
            "feasible": self.feasible,
            "built_levels": ",".join(self.built_levels),
            "layout_key": repr(self.candidate.layout_key),
            "index_key": repr(self.candidate.index_key),
            "quant_key": repr(self.candidate.quant_key),
            "search_key": repr(self.candidate.search_key),
            "stats_provenance": repr(stats_meta),
        }
        for t in THRESHOLDS:
            r[f"p_recall_ge_{t:.2f}"] = self.probs.get(t)
            r[f"conf_recall_ge_{t:.2f}"] = self.confidence.get(t)
        for level, secs in self.build_seconds.items():
            r[f"{level}_seconds"] = secs
        if self.outcome is not None:
            r.update(
                qps=self.outcome.qps,
                p50_ms=self.outcome.p50_ms,
                p95_ms=self.outcome.p95_ms,
                p99_ms=self.outcome.p99_ms,
                mean_recall=self.outcome.mean_recall,
                search_seconds=self.outcome.seconds,
            )
        # the full recall-training feature row (dataset stats + config),
        # so completed runs can feed straight back into data.csv-style files
        r.update(config_features(self.candidate, workload))
        r.update({k: v for k, v in workload.items() if k not in r})
        return r


@dataclass
class _Scored:
    """Per-candidate scoring state, aligned with one sampled batch."""

    scores: np.ndarray
    ehvi: np.ndarray  # NaN pre-warmup
    w_r: np.ndarray
    cost: np.ndarray
    pred: RecallPrediction  # offline -> Platt -> recall-GP blend (decisions)
    raw_pred: RecallPrediction  # offline classifier only (feeds the recalibrator)


class Optimizer:
    def __init__(
        self,
        *,
        space: ConfigSpace,
        evaluator: Evaluator,
        recall_model: RecallClassifier,
        workload: dict[str, Any],
        stats_meta: dict[str, Any],
        cost_model: CostModel,
        settings: OptSettings,
        cache: ArtifactCache | None = None,
    ):
        """`workload` is the joined dict of dataset/query statistics plus
        workload metadata (corpus_size, query_count, vector_dim,
        distance_metric, ...) that every candidate's config features are
        merged with before hitting the recall classifier."""
        self.space = space
        self.evaluator = evaluator
        self.recall_model = recall_model
        self.workload = workload
        self.stats_meta = stats_meta
        self.cost_model = cost_model
        self.settings = settings
        self.cache = cache or ArtifactCache()
        self.surrogate = MoboSurrogate(space.axes, seed=settings.seed)
        self.online = OnlineRecalibrator(
            prior_strength=settings.online_prior_strength
        )
        self.recall_gp = RecallSurrogate(
            space.axes,
            min_observations=settings.recall_gp_min_observations,
            seed=settings.seed,
        )
        self.rng = np.random.default_rng(settings.seed)
        self.trials: list[Trial] = []
        self.spent_seconds = 0.0
        self._evaluated: set[tuple] = set()

    # -- scoring --------------------------------------------------------------

    def _gamma_now(self) -> float:
        s = self.settings
        if not s.cost_cooling:
            return s.gamma
        remaining = max(0.0, 1.0 - self.spent_seconds / s.budget_seconds)
        return s.gamma * remaining

    def _predict_recall(
        self, cands: list[Candidate]
    ) -> tuple[RecallPrediction, RecallPrediction]:
        """(blended, raw_offline). `raw_offline` is the untouched classifier
        output — what `OnlineRecalibrator` must be trained against. Feeding
        it the already-recalibrated (or GP-blended) prediction instead would
        make it fit a correction to its own output, a feedback loop rather
        than a calibration against the fixed offline prior."""
        rows = pd.DataFrame(
            [config_features(c, self.workload) | self.workload for c in cands]
        )
        raw = self.recall_model.predict(rows)
        blended = self.recall_gp.blend(cands, self.online.apply(raw))
        return blended, raw

    def _children_for(self, cand: Candidate) -> list[Candidate]:
        s = self.settings
        return self.space.children(
            cand,
            ef_search=s.children_ef_search,
            batch_size=s.children_batch_size,
            max_children=s.max_children,
            exclude=self._evaluated,
        )

    def _diversity(self, cands: list[Candidate]) -> np.ndarray:
        """Cold-start space-filling factor: normalized min encoded-space
        distance to everything already evaluated (1 when nothing is)."""
        evaluated = [t.candidate for t in self.trials]
        if not evaluated:
            return np.ones(len(cands))
        x = self.surrogate.encoder.encode(cands)
        e = self.surrogate.encoder.encode(evaluated)
        d = np.sqrt(((x[:, None, :] - e[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        peak = d.max()
        return d / peak if peak > 0 else np.ones(len(cands))

    def _score_candidates(self, cands: list[Candidate]) -> _Scored:
        """Score a batch. For cost-aware strategies, a candidate that needs
        an expensive build is scored by the whole batch it unlocks (itself +
        the scheduler's cheap children), per unit of the batch's cost."""
        s = self.settings
        gamma_now = self._gamma_now()
        t0 = acquisition.nearest_threshold(s.target_recall)
        batch_aware = s.strategy in ("bo_cost", "full")

        children_of: dict[int, list[Candidate]] = {}
        if batch_aware:
            for i, c in enumerate(cands):
                needs_build = any(
                    lvl in self.cache.missing_levels(c) for lvl in s.amortize_levels
                )
                children_of[i] = self._children_for(c) if needs_build else []

        flat = list(cands) + [ch for kids in children_of.values() for ch in kids]
        pred_flat, raw_flat = self._predict_recall(flat)
        w_r_flat = acquisition.recall_weight(
            pred_flat, target_recall=s.target_recall, beta=s.beta
        )
        if self.surrogate.ready:
            # the vanilla-BO baseline is recall-oblivious by definition, so
            # it improves the all-points front; every other strategy targets
            # the feasible front
            ehvi_flat = self.surrogate.ehvi(
                flat, feasible_only=(s.strategy != "bo")
            )
        else:
            ehvi_flat = np.ones(len(flat))

        n = len(cands)
        pred = RecallPrediction(
            probs={t: pred_flat.probs[t][:n] for t in THRESHOLDS},
            confidence={t: pred_flat.confidence[t][:n] for t in THRESHOLDS},
        )
        raw_pred = RecallPrediction(
            probs={t: raw_flat.probs[t][:n] for t in THRESHOLDS},
            confidence={t: raw_flat.confidence[t][:n] for t in THRESHOLDS},
        )
        w_r = w_r_flat[:n]
        ehvi = ehvi_flat[:n]
        cost = np.array(
            [self.cost_model.marginal_cost(c, self.cache) for c in cands]
        )

        # batch value: candidate + its children (children are search-only,
        # so their marginal cost is one search each on the new artifact).
        # Two accumulators because the bo_cost baseline is recall-free by
        # definition — its batch value must not smuggle W_R back in.
        value_weighted = ehvi * w_r
        value_plain = ehvi.copy()
        batch_cost = cost.copy()
        if batch_aware:
            pos = n
            for i in range(n):
                kids = children_of.get(i, [])
                if not kids:
                    continue
                k = len(kids)
                value_weighted[i] += float(
                    (ehvi_flat[pos:pos + k] * w_r_flat[pos:pos + k]).sum()
                )
                value_plain[i] += float(ehvi_flat[pos:pos + k].sum())
                batch_cost[i] += sum(
                    self.cost_model.estimate("search", ch) for ch in kids
                )
                pos += k

        if not self.surrogate.ready:
            # cold start: no Pareto information yet, so a space-filling
            # diversity factor stands in for EHVI. Routed through the same
            # per-strategy formula as post-warmup scoring so each baseline's
            # contract (what it's allowed to zero out) holds from the very
            # first iteration, not just once the GPs are ready — `bo` must
            # stay cost/recall-oblivious, `bo_cost` recall-free, `bo_recall`
            # a hard prune with no cost term, even during cold start.
            scores = acquisition.score(
                ehvi=self._diversity(cands),
                w_r=w_r,
                prob_target=pred.probs[t0],
                cost=cost,
                gamma=gamma_now,
                strategy=s.strategy,
                prune_min_probability=s.prune_min_probability,
            )
            return _Scored(
                scores=scores, ehvi=np.full(n, np.nan), w_r=w_r, cost=cost,
                pred=pred, raw_pred=raw_pred,
            )

        if batch_aware:
            scores = acquisition.score(
                ehvi=value_weighted if s.strategy == "full" else value_plain,
                w_r=np.ones(n),  # feasibility already inside value_weighted
                prob_target=pred.probs[t0],
                cost=batch_cost,
                gamma=gamma_now,
                strategy=s.strategy,
                prune_min_probability=s.prune_min_probability,
            )
        else:
            scores = acquisition.score(
                ehvi=ehvi,
                w_r=w_r,
                prob_target=pred.probs[t0],
                cost=cost,
                gamma=gamma_now,
                strategy=s.strategy,
                prune_min_probability=s.prune_min_probability,
            )
        return _Scored(
            scores=scores, ehvi=ehvi, w_r=w_r, cost=cost, pred=pred, raw_pred=raw_pred
        )

    # -- evaluation -----------------------------------------------------------

    def _is_feasible(self, outcome: SearchOutcome, probs: dict[float, float]) -> bool:
        if outcome.mean_recall is not None:
            return outcome.mean_recall >= self.settings.target_recall
        t0 = acquisition.nearest_threshold(self.settings.target_recall)
        return probs.get(t0, 0.0) >= 0.5

    def _evaluate_one(
        self,
        iteration: int,
        cand: Candidate,
        *,
        selected: bool,
        scored: _Scored,
        idx: int,
        score_val: float | None,
    ) -> Trial:
        levels = self.cache.missing_levels(cand)
        probs = {t: float(scored.pred.probs[t][idx]) for t in THRESHOLDS}
        raw_probs = {t: float(scored.raw_pred.probs[t][idx]) for t in THRESHOLDS}
        trial = Trial(
            iteration=iteration,
            candidate=cand,
            selected=selected,
            ok=False,
            error=None,
            probs=probs,
            confidence={
                t: float(scored.pred.confidence[t][idx]) for t in THRESHOLDS
            },
            recall_weight=float(scored.w_r[idx]),
            ehvi=(None if np.isnan(scored.ehvi[idx]) else float(scored.ehvi[idx])),
            predicted_cost=float(scored.cost[idx]),
            score=score_val,
            built_levels=tuple(lvl for lvl in levels if lvl != "search"),
        )
        self._evaluated.add(cand.search_key)
        try:
            build_secs = self.evaluator.build(cand, levels)
            trial.build_seconds = build_secs
            for level, secs in build_secs.items():
                self.cost_model.observe(level, secs, cand)
                self.spent_seconds += secs
            # build succeeded: those artifacts now exist whether or not the
            # search measurement below also succeeds
            self.cache.add(cand, trial.built_levels)

            outcome = self.evaluator.search(cand)
            trial.outcome = outcome
            trial.ok = True
            trial.feasible = self._is_feasible(outcome, probs)
            self.cache.add(cand, ("search",))
            self.cost_model.observe("search", outcome.seconds, cand)
            self.spent_seconds += outcome.seconds
            self.surrogate.add(
                Observation(
                    candidate=cand,
                    qps=outcome.qps,
                    latency_ms=outcome.p95_ms,
                    feasible=trial.feasible,
                )
            )
            if outcome.mean_recall is not None:
                self.online.add(raw_probs, outcome.mean_recall)
                self.recall_gp.add(cand, outcome.mean_recall)
        except EvalError as e:
            trial.error = str(e)
            log.warning("evaluation failed: %s", e)
        self.trials.append(trial)
        return trial

    # -- main loop -------------------------------------------------------------

    def run(self) -> list[Trial]:
        s = self.settings
        iteration = 0
        while (
            self.spent_seconds < s.budget_seconds
            and len(self.trials) < s.max_evaluations
        ):
            iteration += 1
            cands = self.space.sample(
                s.n_candidates,
                self.rng,
                exclude=self._evaluated,
                bias_quant_keys=tuple(self.cache.quants),
                bias_fraction=s.bias_fraction,
            )
            if not cands:
                log.info("search space exhausted after %d trials", len(self.trials))
                break

            scored = self._score_candidates(cands)
            if s.strategy == "random" or not np.any(scored.scores > 0):
                # random baseline — or every candidate scored to zero
                # (e.g. all recall-pruned): fall back to uniform choice
                # rather than argmax of ties
                pick = int(self.rng.integers(len(cands)))
            else:
                pick = int(np.argmax(scored.scores))
            chosen = cands[pick]

            trial = self._evaluate_one(
                iteration, chosen,
                selected=True, scored=scored, idx=pick,
                score_val=float(scored.scores[pick]),
            )
            log.info(
                "iter %d: strategy=%s score=%.4g gamma=%.2f levels=%s ok=%s",
                iteration, s.strategy, scored.scores[pick], self._gamma_now(),
                ",".join(trial.built_levels) or "none", trial.ok,
            )

            # amortize an expensive build with cheap search-level children
            built_expensive = any(
                lvl in trial.built_levels for lvl in s.amortize_levels
            )
            if trial.ok and built_expensive and s.max_children > 0:
                children = self._children_for(chosen)
                if children:
                    child_scored = self._score_candidates(children)
                    for j, child in enumerate(children):
                        if (
                            self.spent_seconds >= s.budget_seconds
                            or len(self.trials) >= s.max_evaluations
                        ):
                            break
                        self._evaluate_one(
                            iteration, child,
                            selected=False, scored=child_scored, idx=j,
                            score_val=None,
                        )

            self.surrogate.fit()
            self.recall_gp.fit()

        return self.trials

    # -- reporting --------------------------------------------------------------

    def rows(self) -> list[dict]:
        return [t.row(self.workload, self.stats_meta) for t in self.trials]

    def best_feasible(self) -> Trial | None:
        """Highest-QPS successful trial whose *measured* recall met the
        target (falling back to predicted feasibility when the evaluator
        produced no recall, e.g. storm without ground truth)."""
        feasible = [t for t in self.trials if t.ok and t.feasible]
        if not feasible:
            return None
        return max(feasible, key=lambda t: t.outcome.qps)
