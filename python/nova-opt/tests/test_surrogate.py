import numpy as np
import pytest

from scipy.stats import norm

from nova_opt.space import Candidate, Index, Layout, Quant, Search, SpaceAxes
from nova_opt.surrogate import (
    CandidateEncoder,
    MoboSurrogate,
    Observation,
    ehvi_2d_exact,
    hypervolume_2d,
    pareto_front,
)

AXES = SpaceAxes(
    m=(8, 16, 32),
    ef_construct=(64, 128),
    ef_search=(8, 16, 32, 64, 128, 256),
    quant_variant=("none", "scalar_default"),
)


def cand(ef_search, m=16, variant="none") -> Candidate:
    return Candidate(
        layout=Layout(segments=8),
        index=Index(m=m, ef_construct=128),
        quant=Quant(variant=variant),
        search=Search(ef_search=ef_search),
    )


def test_pareto_front_filters_dominated():
    pts = np.array([[1, 1], [2, 0.5], [0.5, 2], [1.5, 0.4], [0.9, 0.9]])
    front = pareto_front(pts)
    assert front.shape == (3, 2)
    assert [2, 0.5] in front.tolist()
    assert [0.5, 2] in front.tolist()
    assert [1, 1] in front.tolist()
    assert [0.9, 0.9] not in front.tolist()


def test_hypervolume_known_value():
    front = pareto_front(np.array([[2.0, 1.0], [1.0, 2.0]]))
    ref = np.array([0.0, 0.0])
    # rectangles: (2-0)*(1-0) + (1-0)*(2-1) = 3
    assert hypervolume_2d(front, ref) == pytest.approx(3.0)


def test_hypervolume_ignores_points_under_reference():
    front = pareto_front(np.array([[2.0, 1.0], [-1.0, 5.0]]))
    ref = np.array([0.0, 0.0])
    assert hypervolume_2d(front, ref) == pytest.approx(2.0)


# -- exact EHVI ---------------------------------------------------------------


def test_ehvi_exact_degenerate_matches_hvi():
    """With ~zero predictive variance, EHVI must equal the deterministic
    hypervolume improvement."""
    front = pareto_front(np.array([[2.0, 1.0], [1.0, 2.0]]))
    ref = np.array([0.0, 0.0])
    sd = np.array([1e-9])
    # (3, 3) dominates the whole front: HV 9 - HV 3 = 6
    out = ehvi_2d_exact(front, ref, np.array([3.0]), sd, np.array([3.0]), sd)
    assert out[0] == pytest.approx(6.0, rel=1e-6)
    # deep-dominated point: zero improvement
    out = ehvi_2d_exact(front, ref, np.array([0.5]), sd, np.array([0.5]), sd)
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    # empty front: EHVI = full box volume above ref
    empty = np.zeros((0, 2))
    out = ehvi_2d_exact(empty, ref, np.array([2.0]), sd, np.array([1.5]), sd)
    assert out[0] == pytest.approx(3.0, rel=1e-6)


def _mc_ehvi(front, ref, mu1, sd1, mu2, sd2, n=200_000, seed=0):
    from nova_opt.surrogate import hypervolume_2d, pareto_front

    rng = np.random.default_rng(seed)
    base = hypervolume_2d(front, ref)
    z1 = rng.normal(mu1, sd1, size=n)
    z2 = rng.normal(mu2, sd2, size=n)
    total = 0.0
    for a, b in zip(z1, z2):
        hv = hypervolume_2d(pareto_front(np.vstack([front, [[a, b]]])), ref)
        total += max(0.0, hv - base)
    return total / n


def test_ehvi_exact_matches_monte_carlo():
    front = pareto_front(np.array([[2.0, 0.5], [1.5, 1.5], [0.5, 2.5]]))
    ref = np.array([-0.5, -0.5])
    cases = [(2.2, 0.4, 1.0, 0.6), (1.0, 0.8, 2.0, 0.3), (0.0, 0.5, 0.0, 0.5)]
    for mu1, sd1, mu2, sd2 in cases:
        exact = ehvi_2d_exact(
            front, ref,
            np.array([mu1]), np.array([sd1]), np.array([mu2]), np.array([sd2]),
        )[0]
        mc = _mc_ehvi(front, ref, mu1, sd1, mu2, sd2, n=60_000)
        assert exact == pytest.approx(mc, rel=0.05, abs=5e-3)


def test_encoder_stable_width():
    enc = CandidateEncoder(AXES)
    x = enc.encode([cand(8), cand(256, m=32, variant="scalar_default")])
    assert x.shape[0] == 2
    assert x.shape[1] == enc.encode([cand(16)]).shape[1]
    assert not np.allclose(x[0], x[1])


def test_encoder_distinguishes_unset_from_zero_indexing_threshold():
    """`log2(1 + (threshold or 0))` alone maps None and 0 to the same
    value -- an explicit indicator column must disambiguate "not
    configured" from "explicitly set to 0"."""
    enc = CandidateEncoder(AXES)
    unset = cand(8)
    unset = Candidate(
        layout=unset.layout,
        index=Index(m=unset.index.m, ef_construct=unset.index.ef_construct,
                    indexing_threshold=None),
        quant=unset.quant, search=unset.search,
    )
    zero = Candidate(
        layout=unset.layout,
        index=Index(m=unset.index.m, ef_construct=unset.index.ef_construct,
                    indexing_threshold=0),
        quant=unset.quant, search=unset.search,
    )
    x = enc.encode([unset, zero])
    assert not np.allclose(x[0], x[1])


def fitted_surrogate(feasible=None) -> MoboSurrogate:
    """Synthetic *tradeoff*: along ef_search, QPS rises while latency also
    rises — so every observed point is Pareto-optimal and the front is
    non-degenerate (a single dominating point would make every EHVI ~0)."""
    sur = MoboSurrogate(AXES, seed=0)
    for i, efs in enumerate((8, 16, 32, 64, 128)):
        qps = 100.0 * np.sqrt(efs)
        lat = 1.0 + efs / 32.0
        feas = True if feasible is None else feasible[i]
        sur.add(Observation(candidate=cand(efs), qps=qps, latency_ms=lat,
                            feasible=feas))
    sur.fit()
    return sur


def test_surrogate_not_ready_until_enough_points():
    sur = MoboSurrogate(AXES, seed=0, min_observations=3)
    sur.add(Observation(candidate=cand(8), qps=100, latency_ms=5))
    sur.fit()
    assert not sur.ready
    with pytest.raises(RuntimeError):
        sur.ehvi([cand(16)])


def test_invalid_observations_dropped():
    sur = MoboSurrogate(AXES, seed=0)
    sur.add(Observation(candidate=cand(8), qps=0.0, latency_ms=5))
    sur.add(Observation(candidate=cand(8), qps=10.0, latency_ms=-1))
    assert sur.observations == []


def test_ehvi_prefers_unexplored_region():
    sur = fitted_surrogate()
    assert sur.ready
    # ef_search=256 sits outside the observed range (extrapolation, high GP
    # uncertainty) while ef_search=32 is dead-center of observed points
    scores = sur.ehvi([cand(256), cand(32)])
    assert scores.shape == (2,)
    assert np.all(scores >= 0)
    assert scores[0] > scores[1]


def test_infeasible_points_leave_the_front():
    """An infeasible high-QPS observation must not deflate EHVI for
    candidates improving the *feasible* front."""
    # last observation (ef=128: highest qps, worst latency) marked infeasible
    sur_feas = fitted_surrogate(feasible=[True, True, True, True, False])
    sur_all = fitted_surrogate()
    probe = [cand(128, m=32)]  # near the infeasible point's region
    ehvi_feas = sur_feas.ehvi(probe)[0]
    ehvi_all = sur_all.ehvi(probe)[0]
    # with the infeasible point off the front, that region is open again
    assert ehvi_feas > ehvi_all
    # the vanilla-BO view (all-points front) must be available on request
    ehvi_bo = sur_feas.ehvi(probe, feasible_only=False)[0]
    assert ehvi_bo == pytest.approx(ehvi_all, rel=1e-6)


def test_all_infeasible_still_scores():
    sur = fitted_surrogate(feasible=[False] * 5)
    scores = sur.ehvi([cand(64, m=32), cand(256)])
    assert np.all(scores > 0)  # empty feasible front: everything is upside
