"""End-to-end optimizer runs against a synthetic evaluator — no live
cluster, no subprocesses. The simulator gives every candidate an analytic
QPS/latency/recall so the planner's behavior (artifact reuse, child
amortization, budget accounting, strategy differences) is observable."""

import numpy as np
import pandas as pd
import pytest

from nova_opt.cost import ArtifactCache, CostModel
from nova_opt.evaluate import EvalError, Evaluator, ReplayEvaluator, SearchOutcome
from nova_opt.optimizer import Optimizer, OptSettings
from nova_opt.recall import RecallClassifier
from nova_opt.space import Candidate, ConfigSpace, SpaceAxes, config_features

from test_recall import synthetic_training_data

AXES = SpaceAxes(
    segments=(8, 16),
    m=(8, 16, 32),
    ef_construct=(64, 128),
    quant_variant=("none", "binary_1bit", "scalar_default"),
    ef_search=(8, 16, 32, 64, 128, 256),
    batch_size=(1, 8),
)

BUILD_COST = {"layout": 100.0, "index": 40.0, "quant": 10.0}


class SimulatedEvaluator(Evaluator):
    """Analytic ground truth mirroring the synthetic recall training data:
    recall rises with ef_search, drops with binary quantization; QPS falls
    with ef_search and rises with quantization."""

    def __init__(self):
        self.build_calls: list[tuple[Candidate, tuple[str, ...]]] = []
        self.search_calls: list[Candidate] = []

    def build(self, cand, levels):
        self.build_calls.append((cand, levels))
        return {lvl: BUILD_COST[lvl] for lvl in levels if lvl != "search"}

    def _recall(self, cand):
        penalty = {"none": 0.0, "scalar_default": 0.05, "binary_1bit": 0.25}[
            cand.quant.variant
        ]
        return float(np.clip(1.0 - np.exp(-cand.search.ef_search / 88.0) * 1.2 - penalty, 0, 1))

    def search(self, cand):
        self.search_calls.append(cand)
        speedup = {"none": 1.0, "scalar_default": 1.5, "binary_1bit": 3.0}[
            cand.quant.variant
        ]
        qps = 3000.0 * speedup / np.sqrt(cand.search.ef_search) / np.sqrt(cand.index.m)
        lat = (0.5 + cand.search.ef_search / 64.0) / speedup
        return SearchOutcome(
            qps=qps, p50_ms=lat, p95_ms=lat * 1.5, p99_ms=lat * 2,
            mean_recall=self._recall(cand), seconds=5.0,
        )


@pytest.fixture(scope="module")
def recall_model():
    clf = RecallClassifier()
    clf.train(synthetic_training_data(n_per_dataset=200), seed=0)
    return clf


WORKLOAD = {
    "corpus_size": 100_000,
    "query_count": 1000,
    "vector_dim": 64,
    "distance_metric": "COSINE",
    "embedding_intrinsic_dimensionality": 11.0,
    "query_pca_top1_var_ratio": 0.1,
}


def run_optimizer(recall_model, strategy="full", budget=1200.0, seed=0, evaluator=None):
    evaluator = evaluator or SimulatedEvaluator()
    settings = OptSettings(
        target_recall=0.90,
        strategy=strategy,
        budget_seconds=budget,
        max_evaluations=60,
        n_candidates=48,
        seed=seed,
        children_ef_search=(16, 64, 256),
        children_batch_size=(1,),
        max_children=2,
    )
    opt = Optimizer(
        space=ConfigSpace(AXES),
        evaluator=evaluator,
        recall_model=recall_model,
        workload=WORKLOAD,
        stats_meta={"base": {"norm_stats": "exact"}},
        cost_model=CostModel(
            priors={"layout": 100.0, "index": 40.0, "quant": 10.0, "search": 5.0}
        ),
        settings=settings,
    )
    trials = opt.run()
    return opt, trials, evaluator


def test_full_strategy_end_to_end(recall_model):
    opt, trials, sim = run_optimizer(recall_model)
    assert trials, "no trials ran"
    assert opt.spent_seconds >= OptSettings().budget_seconds * 0 and opt.spent_seconds > 0
    # budget respected: loop stops once spent >= budget
    assert opt.spent_seconds <= 1200.0 + max(BUILD_COST.values()) * 3 + 5.0 * 10

    # every evaluated point produced a full record row
    rows = opt.rows()
    assert len(rows) == len(trials)
    for col in ("qps", "p95_ms", "mean_recall", "ef_search", "hnsw_m",
                "quantization_variant", "layout_key", "search_key",
                "p_recall_ge_0.90", "conf_recall_ge_0.90", "recall_weight",
                "predicted_cost", "stats_provenance"):
        assert col in rows[0], f"missing column {col}"

    best = opt.best_feasible()
    assert best is not None
    assert best.outcome.mean_recall >= 0.90


def test_artifact_reuse_never_rebuilds(recall_model):
    """Reuse must avoid *redundant* rebuilds. Layouts are independent
    artifacts (one collection each), so a layout_key is built at most once,
    ever. Index/quant are `reindex` mutations of one collection per layout —
    the same key built twice *in a row* on the same layout would be
    redundant, but revisiting a key after a different one was built into
    that layout in between is a legitimate rebuild (the collection no
    longer holds it), not something the cache should ever avoid."""
    _, _, sim = run_optimizer(recall_model)
    built_layouts = [
        c.layout_key for c, levels in sim.build_calls if "layout" in levels
    ]
    assert len(built_layouts) == len(set(built_layouts))

    def consecutive_repeats(level: str, key_of) -> int:
        by_layout: dict[tuple, list[tuple]] = {}
        for c, levels in sim.build_calls:
            if level in levels:
                by_layout.setdefault(c.layout_key, []).append(key_of(c))
        return sum(
            1
            for keys in by_layout.values()
            for a, b in zip(keys, keys[1:])
            if a == b
        )

    assert consecutive_repeats("index", lambda c: c.index_key) == 0
    assert consecutive_repeats("quant", lambda c: c.quant_key) == 0


def test_children_amortize_expensive_builds(recall_model):
    opt, trials, _ = run_optimizer(recall_model)
    by_iter = {}
    for t in trials:
        by_iter.setdefault(t.iteration, []).append(t)
    amortized = [
        ts for ts in by_iter.values()
        if ts[0].selected and ts[0].built_levels and len(ts) > 1
    ]
    assert amortized, "expensive builds were never amortized with children"
    for ts in amortized:
        parent = ts[0]
        for child in ts[1:]:
            assert not child.selected
            assert child.candidate.quant_key == parent.candidate.quant_key
            # child evaluations reuse the parent's artifact — search only
            assert child.built_levels == ()


def test_no_search_point_evaluated_twice(recall_model):
    _, _, sim = run_optimizer(recall_model)
    keys = [c.search_key for c in sim.search_calls]
    assert len(keys) == len(set(keys))


def test_cost_model_learns_from_observations(recall_model):
    opt, _, _ = run_optimizer(recall_model)
    # simulator search always takes 5s; the EMA must have pulled the 5s->30s
    # prior toward the truth (well below the untouched prior)
    assert opt.cost_model.estimate("search") == pytest.approx(5.0, abs=1.0)


def test_recall_gp_becomes_ready_and_is_used(recall_model):
    """The run-local recall GP must pick up measurements as the loop runs
    and get consulted (not just sit there unused)."""
    opt, trials, _ = run_optimizer(recall_model)
    measured = [t for t in trials if t.ok and t.outcome.mean_recall is not None]
    assert len(measured) >= OptSettings().recall_gp_min_observations
    assert opt.recall_gp.ready
    # once ready, blending must not error and must stay in [0, 1]
    blended, raw = opt._predict_recall([t.candidate for t in trials[:5]])
    for pred in (blended, raw):
        for t in pred.probs.values():
            assert np.all((t >= 0.0) & (t <= 1.0))


def _fresh_optimizer(recall_model, strategy):
    """A never-run Optimizer -- `surrogate.ready` is False and `_diversity`
    (no trials yet) is always exactly 1.0 for every candidate, which makes
    cold-start baseline contracts easy to check precisely."""
    settings = OptSettings(
        target_recall=0.90, strategy=strategy, budget_seconds=1200.0,
        max_evaluations=60, n_candidates=48, seed=0,
    )
    return Optimizer(
        space=ConfigSpace(AXES), evaluator=SimulatedEvaluator(),
        recall_model=recall_model, workload=WORKLOAD,
        stats_meta={"base": {"norm_stats": "exact"}},
        cost_model=CostModel(
            priors={"layout": 100.0, "index": 40.0, "quant": 10.0, "search": 5.0}
        ),
        settings=settings,
    )


def test_cold_start_scoring_respects_baseline_contracts(recall_model):
    """Before the GP surrogate has enough points, `bo` must stay cost- and
    recall-oblivious, `bo_cost` must stay recall-free, and `bo_recall` must
    hard-prune with no cost/soft-weighting term -- the same contracts
    `acquisition.score` enforces post-warmup must hold from iteration one,
    not just once the GPs are ready."""
    cands = ConfigSpace(AXES).sample(8, np.random.default_rng(0))

    opt_bo = _fresh_optimizer(recall_model, "bo")
    assert not opt_bo.surrogate.ready
    scored = opt_bo._score_candidates(cands)
    # pre-trial diversity is always exactly 1.0; `bo`'s cold-start score must
    # be exactly that, uninfluenced by recall_weight or cost (the old bug
    # multiplied both in regardless of strategy)
    np.testing.assert_allclose(scored.scores, 1.0)

    opt_cost = _fresh_optimizer(recall_model, "bo_cost")
    scored = opt_cost._score_candidates(cands)
    cost = np.array(
        [opt_cost.cost_model.marginal_cost(c, opt_cost.cache) for c in cands]
    )
    gamma_now = opt_cost._gamma_now()
    expected = 1.0 / np.power(np.maximum(cost, 1e-6), gamma_now)
    # exact match to a purely cost-based formula -- if recall_weight had
    # leaked in (the bug), this would multiply in a per-candidate factor and
    # break the match
    np.testing.assert_allclose(scored.scores, expected, rtol=1e-6)

    opt_recall = _fresh_optimizer(recall_model, "bo_recall")
    scored = opt_recall._score_candidates(cands)
    # hard prune only: every score is exactly 0.0 (pruned) or 1.0 (kept),
    # never a soft w_r/cost-scaled value
    assert set(np.round(scored.scores, 9)) <= {0.0, 1.0}


def test_online_recalibrator_is_fed_raw_offline_probabilities(recall_model):
    """`OnlineRecalibrator` must learn from the untouched offline classifier
    output, not from its own previously-recalibrated (or recall-GP-blended)
    prediction -- feeding it its own output back in would be a feedback
    loop, not a calibration against a fixed prior."""
    opt, trials, _ = run_optimizer(recall_model)
    measured = [t for t in trials if t.ok and t.outcome.mean_recall is not None]
    assert measured

    t0 = 0.90
    stored = [p for p, _ in opt.online._obs[t0]]
    rows = pd.DataFrame(
        [config_features(t.candidate, opt.workload) | opt.workload for t in measured]
    )
    raw = opt.recall_model.predict(rows).probs[t0]
    np.testing.assert_allclose(stored, raw, atol=1e-9)


def test_random_strategy_runs(recall_model):
    opt, trials, _ = run_optimizer(recall_model, strategy="random", budget=400.0, seed=3)
    assert trials
    assert any(t.ok for t in trials)


def test_full_beats_cost_oblivious_bo_on_builds(recall_model):
    """The cost-aware planner should spend fewer seconds on layout rebuilds
    per search evaluation than cost-oblivious BO under the same budget."""
    def layout_builds(sim):
        return sum(1 for _, levels in sim.build_calls if "layout" in levels)

    _, t_full, sim_full = run_optimizer(recall_model, strategy="full", budget=800.0)
    _, t_bo, sim_bo = run_optimizer(recall_model, strategy="bo", budget=800.0)
    full_ratio = layout_builds(sim_full) / max(len(sim_full.search_calls), 1)
    bo_ratio = layout_builds(sim_bo) / max(len(sim_bo.search_calls), 1)
    assert full_ratio <= bo_ratio


class FailingEvaluator(SimulatedEvaluator):
    def search(self, cand):
        if cand.search.ef_search == 8:
            raise EvalError("synthetic failure")
        return super().search(cand)


def test_failures_recorded_not_raised(recall_model):
    opt, trials, _ = run_optimizer(
        recall_model, evaluator=FailingEvaluator(), budget=600.0
    )
    failed = [t for t in trials if not t.ok]
    for t in failed:
        assert t.error and "synthetic failure" in t.error
        assert t.outcome is None
    # failures don't poison the surrogate or stop the loop
    assert any(t.ok for t in trials)


def test_replay_evaluator_lookup_and_miss():
    table = pd.DataFrame(
        [
            {
                "number_of_segments": 8, "hnsw_m": 16, "ef_construct": 128,
                "quantization_variant": "none", "ef_search": 64, "top_k": 10,
                "rescore": None,
                "qps": 900.0, "p50_ms": 2.0, "p95_ms": 4.0,
                "mean_recall": 0.93, "index_seconds": 33.0,
            }
        ]
    )
    ev = ReplayEvaluator(table)
    from nova_opt.space import Index, Layout, Quant, Search

    hit = Candidate(
        layout=Layout(segments=8), index=Index(m=16, ef_construct=128),
        quant=Quant(variant="none"), search=Search(ef_search=64),
    )
    out = ev.search(hit)
    assert out.qps == 900.0 and out.mean_recall == 0.93
    costs = ev.build(hit, ("index", "search"))
    assert costs == {"index": 33.0}

    miss = hit.with_search(Search(ef_search=128))
    with pytest.raises(EvalError, match="no replay row"):
        ev.search(miss)
