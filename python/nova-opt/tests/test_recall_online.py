import numpy as np

from nova_opt.recall import THRESHOLDS, RecallPrediction
from nova_opt.recall_online import RecallSurrogate
from nova_opt.space import Candidate, Index, Layout, Quant, Search, SpaceAxes

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


def true_recall(ef_search: int) -> float:
    """Monotone synthetic ground truth: recall rises with ef_search."""
    return float(np.clip(np.log2(ef_search) / 8.0, 0.02, 0.98))


def fitted_surrogate(min_observations: int = 5) -> RecallSurrogate:
    sur = RecallSurrogate(AXES, min_observations=min_observations, seed=0)
    for ef in (8, 16, 32, 64, 128):
        sur.add(cand(ef), true_recall(ef))
    sur.fit()
    return sur


def flat_pred(n: int, value: float = 0.9) -> RecallPrediction:
    v = np.full(n, value)
    return RecallPrediction(
        probs={t: v.copy() for t in THRESHOLDS},
        confidence={t: np.full(n, 0.5) for t in THRESHOLDS},
    )


def test_not_ready_below_min_observations():
    sur = RecallSurrogate(AXES, min_observations=5)
    for ef in (8, 16, 32):
        sur.add(cand(ef), true_recall(ef))
    sur.fit()
    assert not sur.ready


def test_not_ready_without_feature_variation():
    sur = RecallSurrogate(AXES, min_observations=3)
    for recall in (0.9, 0.1, 0.5):
        sur.add(cand(64), recall)  # identical candidate every time
    sur.fit()
    assert not sur.ready


def test_ready_with_enough_varied_observations():
    assert fitted_surrogate().ready


def test_confidence_high_near_data_low_far_away():
    sur = fitted_surrogate()
    conf = sur.confidence([cand(32), cand(2**24)])
    assert conf[0] > 0.9
    assert conf[1] < 0.3
    assert conf[0] > conf[1]


def test_prob_feasible_monotone_in_threshold():
    sur = fitted_surrogate()
    probs = sur.prob_feasible([cand(48)], THRESHOLDS)
    ordered = sorted(THRESHOLDS)
    values = [float(probs[t][0]) for t in ordered]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))


def test_blend_is_noop_when_not_ready():
    sur = RecallSurrogate(AXES, min_observations=5)
    pred = flat_pred(2, 0.9)
    out = sur.blend([cand(8), cand(256)], pred)
    for t in THRESHOLDS:
        np.testing.assert_allclose(out.probs[t], pred.probs[t])
    # confidence is untouched, always — GP-readiness doesn't affect it
    for t in THRESHOLDS:
        np.testing.assert_allclose(out.confidence[t], pred.confidence[t])


def test_blend_pulls_overconfident_offline_prediction_down_locally():
    """Offline claims a flat 0.9 everywhere; reality is that ef_search=8 is
    nowhere close to 0.80 recall while ef_search=128 comfortably clears it.
    The spatial GP — unlike the global Platt scalar — must correct these
    two points in opposite directions, not by the same amount."""
    sur = fitted_surrogate()
    assert sur.ready
    pred = flat_pred(2, 0.9)
    out = sur.blend([cand(8), cand(128)], pred)
    assert out.probs[0.80][0] < 0.2  # low ef_search: corrected down hard
    assert out.probs[0.80][1] > 0.8  # high ef_search: offline was right anyway
