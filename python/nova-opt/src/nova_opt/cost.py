"""Reuse-aware marginal cost model + the artifact cache it consults.

The cache remembers which layout artifacts exist, and — separately, because
they behave differently — which index/quant artifact is *currently*
materialized on top of each one (as the `space.py` prefix keys). Layouts are
independent, permanent artifacts (one Qdrant collection each); index/quant
are in-place `reindex` mutations of a layout's one collection, so only the
most recently built index/quant key is actually live for that layout, no
matter how many others were built there earlier in the run (see
`ArtifactCache`'s docstring). The cost model prices a candidate as the sum of
the levels that would actually have to be built:

    layout missing  -> insert + index + quant + search
    index missing   ->          index + quant + search
    quant missing   ->                  quant + search
    otherwise       ->                          search

Per-level estimates come from three sources, best-available first:

1. a small ridge regression of log(seconds) on level-relevant config
   features (index-build time genuinely scales with m and ef_construct;
   pretending it's a constant makes the acquisition's denominator lie) —
   used once a level has a few observations spanning different configs;
2. an EMA of observed durations (config-blind, but tracks the actual
   instance/dataset);
3. the configured prior, before anything has been measured.

Regression predictions are clamped to a band around the EMA so a wild
extrapolation from two points can't distort the acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nova_opt.space import QUANT_VARIANTS, Candidate

# Cheapest-last, matching the artifact hierarchy. "layout" is the
# insert/reinsert step; "search" is one nova-storm evaluation.
LEVELS = ("layout", "index", "quant", "search")

_QUANT_FAMILIES = ("NONE", "SCALAR", "BINARY", "PRODUCT", "TURBO")


def _level_features(cand: Candidate, level: str) -> np.ndarray:
    """The config features a level's build time plausibly scales with."""
    if level == "layout":
        return np.array([1.0, np.log2(cand.layout.segments),
                         np.log2(cand.layout.shard_count)])
    if level == "index":
        return np.array([1.0, np.log2(cand.index.m),
                         np.log2(cand.index.ef_construct)])
    if level == "quant":
        family = QUANT_VARIANTS[cand.quant.variant].quantization
        return np.array([1.0, *(1.0 if family == f else 0.0
                                for f in _QUANT_FAMILIES)])
    if level == "search":
        return np.array([1.0, np.log2(cand.search.batch_size),
                         np.log2(cand.search.concurrency)])
    raise ValueError(f"unknown level '{level}'")


@dataclass
class ArtifactCache:
    """What is physically materialized right now, per layout.

    A layout is its own persistent artifact — `LiveQdrantEvaluator` gives
    each `layout_key` its own collection (recreated on a layout rebuild),
    and different layouts coexist independently forever. Index and
    quantization are **not** independent artifacts in the same sense: the
    live evaluator applies them as in-place `reindex` patches on that one
    collection, so at any moment a layout has exactly one live index and one
    live quant, whichever was built into it most recently — not one per
    index/quant key ever seen. `current_index` / `current_quant` model that:
    a layout mapping to a *different* key than the candidate's means the
    physical collection no longer matches, however long ago that key was
    itself built (see DESIGN.md invariant #8b)."""

    layouts: set[tuple] = field(default_factory=set)
    current_index: dict[tuple, tuple] = field(default_factory=dict)  # layout_key -> index_key
    current_quant: dict[tuple, tuple] = field(default_factory=dict)  # layout_key -> quant_key
    searches: set[tuple] = field(default_factory=set)

    def missing_levels(self, cand: Candidate) -> tuple[str, ...]:
        """The levels that must be built to evaluate `cand`, in build order.
        A missing prefix invalidates everything under it (a fresh layout has
        no index yet, whatever the cache says about other layouts), so the
        first miss decides the whole suffix. Search is always included —
        it's the evaluation itself, never a reusable artifact."""
        if cand.layout_key not in self.layouts:
            return ("layout", "index", "quant", "search")
        if self.current_index.get(cand.layout_key) != cand.index_key:
            return ("index", "quant", "search")
        if self.current_quant.get(cand.layout_key) != cand.quant_key:
            return ("quant", "search")
        return ("search",)

    def add(self, cand: Candidate, levels: tuple[str, ...]) -> None:
        """Record the artifacts a successful build of `levels` produced."""
        if "layout" in levels:
            self.layouts.add(cand.layout_key)
        if "index" in levels:
            self.current_index[cand.layout_key] = cand.index_key
        if "quant" in levels:
            self.current_quant[cand.layout_key] = cand.quant_key
        if "search" in levels:
            self.searches.add(cand.search_key)

    @property
    def quants(self) -> tuple[tuple, ...]:
        """Every currently-live quant artifact, one per layout — used only
        to bias candidate sampling toward artifacts that are cheap *right
        now* (see `space.ConfigSpace.sample`'s `bias_quant_keys`)."""
        return tuple(self.current_quant.values())


@dataclass
class CostModel:
    """Per-level cost estimates in seconds — see module docstring for the
    prior -> EMA -> regression escalation."""

    priors: dict[str, float]
    alpha: float = 0.3
    # observations needed (with >= 2 distinct feature rows) before the
    # per-level regression takes over from the EMA
    min_regression_obs: int = 3
    ridge_lambda: float = 1.0
    _ema: dict[str, float] = field(default_factory=dict)
    _obs: dict[str, list[tuple[np.ndarray, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [lvl for lvl in LEVELS if lvl not in self.priors]
        if missing:
            raise ValueError(f"cost priors missing levels: {missing}")
        non_positive = {lvl: v for lvl, v in self.priors.items() if v <= 0}
        if non_positive:
            # the EMA is seeded directly from these, and the ridge-regression
            # clamp band ([EMA/5, EMA*5], see `estimate`) collapses to a
            # single point (or inverts) at zero/negative
            raise ValueError(f"cost priors must be positive, got {non_positive}")
        self._ema = dict(self.priors)
        self._obs = {lvl: [] for lvl in LEVELS}

    def observe(self, level: str, seconds: float, cand: Candidate | None = None) -> None:
        if seconds <= 0:
            return
        self._ema[level] = (1 - self.alpha) * self._ema[level] + self.alpha * seconds
        if cand is not None:
            self._obs[level].append((_level_features(cand, level), np.log(seconds)))

    def _regress(self, level: str, x: np.ndarray) -> float | None:
        obs = self._obs[level]
        if len(obs) < self.min_regression_obs:
            return None
        feats = np.stack([f for f, _ in obs])
        if len(np.unique(feats, axis=0)) < 2:
            return None  # no feature variation observed — nothing to fit
        y = np.array([s for _, s in obs])
        # ridge, intercept unpenalized (feature 0 is the constant term)
        pen = np.eye(feats.shape[1]) * self.ridge_lambda
        pen[0, 0] = 0.0
        w = np.linalg.solve(feats.T @ feats + pen, feats.T @ y)
        return float(np.exp(x @ w))

    def estimate(self, level: str, cand: Candidate | None = None) -> float:
        ema = self._ema[level]
        if cand is None:
            return ema
        pred = self._regress(level, _level_features(cand, level))
        if pred is None:
            return ema
        # clamp: a 3-point fit must inform the estimate, not replace it
        return float(np.clip(pred, ema / 5.0, ema * 5.0))

    def marginal_cost(self, cand: Candidate, cache: ArtifactCache) -> float:
        """C_reuse(x | artifacts): seconds to evaluate `cand` given what
        already exists. Always >= the search-level cost (floored at a small
        positive value so the acquisition's cost division is safe)."""
        cost = sum(
            self.estimate(lvl, cand) for lvl in cache.missing_levels(cand)
        )
        return max(cost, 1e-6)
