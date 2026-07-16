"""Multi-objective BO surrogate over (QPS, latency).

Two independent Gaussian-process regressors model log(QPS) and log(latency)
as functions of an encoded candidate vector. The acquisition works in the
"maximize both" frame f = (log qps, -log latency); expected hypervolume
improvement (EHVI) is computed in **closed form** (exact for two objectives
with independent normal marginals — no Monte-Carlo noise in candidate
ranking, and cheap enough to score thousands of candidates per iteration).

The Pareto front the EHVI improves upon contains **feasible observations
only** (measured recall met the target). GPs still train on every
observation — an infeasible point is perfectly good evidence about QPS and
latency — but it must not sit on the front: a sky-high-QPS config at recall
0.4 would otherwise make every feasible candidate look like a negligible
improvement, blinding the acquisition exactly where the answer lives. The
vanilla-BO baseline (which by definition knows nothing about recall) asks
for the all-points front explicitly.

Log-space objectives keep the hypervolume scale-sane (a 2x QPS gain counts
the same at 100 or 10k QPS) and make the GP's stationarity assumption far
less wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.stats import norm

from nova_opt.space import Candidate, SpaceAxes


class CandidateEncoder:
    """Candidate -> fixed-width float vector: log2 for the scale-like numeric
    knobs, one-hot over the vocabularies the space was declared with (so the
    encoding is stable across the whole run)."""

    def __init__(self, axes: SpaceAxes):
        self.dtypes = sorted(set(axes.dtype))
        self.variants = sorted(set(axes.quant_variant))

    def encode(self, cands: list[Candidate]) -> np.ndarray:
        rows = []
        for c in cands:
            row = [
                np.log2(c.layout.segments),
                np.log2(c.layout.shard_count),
                float(c.layout.on_disk_payload),
                np.log2(c.index.m),
                np.log2(c.index.ef_construct),
                float(c.index.on_disk),
                np.log2(1 + (c.index.indexing_threshold or 0)),
                # `or 0` maps None and 0 to the same log2 value -- this
                # indicator disambiguates "not configured" from "explicitly 0"
                1.0 if c.index.indexing_threshold is None else 0.0,
                float(c.quant.always_ram),
                np.log2(c.search.ef_search),
                np.log2(c.search.batch_size),
                np.log2(c.search.top_k),
                np.log2(c.search.concurrency),
                # tri-state rescore: unset / false / true
                1.0 if c.search.rescore is True else 0.0,
                1.0 if c.search.rescore is None else 0.0,
            ]
            row.extend(1.0 if c.layout.dtype == d else 0.0 for d in self.dtypes)
            row.extend(1.0 if c.quant.variant == v else 0.0 for v in self.variants)
            rows.append(row)
        return np.asarray(rows, dtype=float)


def make_gp(seed: int, n_features: int):
    """ARD Matern: one length scale per encoded feature, not a single scalar
    shared by every axis. `CandidateEncoder` mixes log2-scale numeric knobs
    with one-hot dummies — a shared length scale forces one distance notion
    onto both, and an irrelevant one-hot column has no way to be discounted.
    Per-feature length scales let the optimizer push irrelevant dimensions
    toward their upper bound (effectively "ignore this axis") independently
    of how relevant dimensions get scaled — the standard remedy for
    mixed-scale features, and cheap here since the encoded width is small."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e3), nu=2.5
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed
    )


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Non-dominated subset of `points` (shape (n, 2), maximize both),
    sorted by f1 descending."""
    if len(points) == 0:
        return points.reshape(0, 2)
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    front = []
    best_f2 = -np.inf
    for i in order:
        if points[i, 1] > best_f2:
            front.append(points[i])
            best_f2 = points[i, 1]
    return np.asarray(front)


def hypervolume_2d(front: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D hypervolume of `front` (non-dominated, f1-descending, both
    maximized) with respect to reference point `ref` (dominated by all)."""
    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in front:
        if f1 <= ref[0] or f2 <= prev_f2:
            continue
        hv += (f1 - ref[0]) * (f2 - prev_f2)
        prev_f2 = f2
    return hv


def _g(l: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """E[max(Z - l, 0)] for Z ~ N(mu, sd) — the partial expectation both
    EHVI strip integrals reduce to. Handles l = -inf (-> mu - l would blow
    up; callers never pass it) and sd ~ 0 (degenerate normal)."""
    sd = np.maximum(sd, 1e-12)
    alpha = (l - mu) / sd
    return sd * norm.pdf(alpha) + (mu - l) * norm.sf(alpha)


def ehvi_2d_exact(
    front: np.ndarray,
    ref: np.ndarray,
    mu1: np.ndarray,
    sd1: np.ndarray,
    mu2: np.ndarray,
    sd2: np.ndarray,
) -> np.ndarray:
    """Exact EHVI for two maximized objectives with independent normal
    predictive marginals, vectorized over candidates.

    Uses HVI(z) = ∫ 1[y ≥ ref, y ≤ z, y undominated by front] dy and Fubini:
    EHVI = ∫_A P(Z1 ≥ y1) P(Z2 ≥ y2) dy over the undominated region A, which
    for a 2-D front F = {(a_i, b_i)} (a descending, b ascending) decomposes
    into vertical strips with closed-form partial expectations:

        strip 0:  y1 ∈ (a_1, ∞),        y2 ∈ (ref2, ∞)
        strip i:  y1 ∈ (a_{i+1}, a_i],  y2 ∈ (b_i, ∞)     (a_{n+1} = ref1)

    An empty front is one strip over the whole [ref, ∞) quadrant.
    """
    if front.shape[0] == 0:
        return _g(np.full_like(mu1, ref[0]), mu1, sd1) * _g(
            np.full_like(mu2, ref[1]), mu2, sd2
        )
    a = front[:, 0]
    b = front[:, 1]
    # strip lower/upper y1 bounds and y2 lower bounds, strips 0..n
    lo1 = np.concatenate([[a[0]], a[1:], [ref[0]]])  # len n+1
    hi1 = np.concatenate([[np.inf], a])  # strip 0 unbounded above
    lo2 = np.concatenate([[ref[1]], b])

    out = np.zeros_like(mu1, dtype=float)
    for i in range(len(lo1)):
        upper = (
            np.zeros_like(mu1) if np.isinf(hi1[i]) else _g(np.full_like(mu1, hi1[i]), mu1, sd1)
        )
        width = _g(np.full_like(mu1, lo1[i]), mu1, sd1) - upper
        out += width * _g(np.full_like(mu2, lo2[i]), mu2, sd2)
    return np.maximum(out, 0.0)


@dataclass
class Observation:
    candidate: Candidate
    qps: float
    latency_ms: float
    # measured recall met the run's target (or best available proxy). Decides
    # front membership, never GP membership.
    feasible: bool = True


class MoboSurrogate:
    """The pair of GPs plus the observed front, refit after each batch of
    evaluations. Until `min_observations` points exist the surrogate reports
    itself not-ready and the optimizer falls back to cost/recall-only
    scoring (its initial exploration phase)."""

    def __init__(self, axes: SpaceAxes, *, seed: int = 0, min_observations: int = 3):
        self.encoder = CandidateEncoder(axes)
        self.seed = seed
        self.min_observations = min_observations
        self.observations: list[Observation] = []
        self._gp_qps = None
        self._gp_lat = None

    @property
    def ready(self) -> bool:
        return self._gp_qps is not None

    def add(self, obs: Observation) -> None:
        if obs.qps > 0 and obs.latency_ms > 0:
            self.observations.append(obs)

    def fit(self) -> None:
        if len(self.observations) < self.min_observations:
            return
        x = self.encoder.encode([o.candidate for o in self.observations])
        y_qps = np.log([o.qps for o in self.observations])
        y_lat = np.log([o.latency_ms for o in self.observations])
        self._gp_qps = make_gp(self.seed, x.shape[1]).fit(x, y_qps)
        self._gp_lat = make_gp(self.seed, x.shape[1]).fit(x, y_lat)

    def _objective_points(self, feasible_only: bool) -> np.ndarray:
        obs = [o for o in self.observations if o.feasible or not feasible_only]
        if not obs:
            return np.zeros((0, 2))
        return np.array([[np.log(o.qps), -np.log(o.latency_ms)] for o in obs])

    def ehvi(self, cands: list[Candidate], *, feasible_only: bool = True) -> np.ndarray:
        """Exact EHVI per candidate over the current front. `feasible_only`
        picks which front (the vanilla-BO baseline passes False — it is
        recall-oblivious by definition). Requires `ready`."""
        if not self.ready:
            raise RuntimeError("surrogate not fitted yet")
        # reference point / objective span come from ALL observations (a
        # stable scale), the front only from the requested subset
        all_pts = self._objective_points(feasible_only=False)
        span = all_pts.max(axis=0) - all_pts.min(axis=0)
        # margin proportional to the observed objective range — a fixed
        # epsilon would zero out candidates that *extend* the front past the
        # observed nadir, which are exactly the Pareto extensions worth
        # finding
        ref = all_pts.min(axis=0) - np.maximum(0.1, 0.5 * span)
        front = pareto_front(self._objective_points(feasible_only=feasible_only))

        x = self.encoder.encode(cands)
        mu_q, sd_q = self._gp_qps.predict(x, return_std=True)
        mu_l, sd_l = self._gp_lat.predict(x, return_std=True)
        # objective 2 is -log latency: negate the mean, keep the std
        return ehvi_2d_exact(front, ref, mu_q, sd_q, -mu_l, sd_l)
