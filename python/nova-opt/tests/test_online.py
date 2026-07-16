import numpy as np

from nova_opt.online import OnlineRecalibrator
from nova_opt.recall import THRESHOLDS, RecallPrediction


def flat_pred(values):
    v = np.asarray(values, dtype=float)
    return RecallPrediction(
        probs={t: v.copy() for t in THRESHOLDS},
        confidence={t: np.full(len(v), 0.5) for t in THRESHOLDS},
    )


def test_identity_with_no_observations():
    rec = OnlineRecalibrator()
    pred = flat_pred([0.2, 0.7])
    out = rec.apply(pred)
    for t in THRESHOLDS:
        np.testing.assert_allclose(out.probs[t], pred.probs[t])


def test_overconfident_offline_model_gets_pulled_down():
    """Offline says 0.9 feasible, reality keeps failing: after several
    measurements the recalibrated probability must drop substantially."""
    rec = OnlineRecalibrator(prior_strength=4.0)
    for _ in range(10):
        rec.add({t: 0.9 for t in THRESHOLDS}, measured_recall=0.1)
    out = rec.apply(flat_pred([0.9]))
    assert out.probs[0.90][0] < 0.5


def test_prior_dampens_first_observation():
    rec = OnlineRecalibrator(prior_strength=4.0)
    rec.add({t: 0.9 for t in THRESHOLDS}, measured_recall=0.1)
    out = rec.apply(flat_pred([0.9]))
    # one contradicting point moves the estimate, but nowhere near zero
    assert 0.3 < out.probs[0.90][0] < 0.9


def test_confirming_observations_change_little():
    rec = OnlineRecalibrator()
    for p_off, recall in [(0.9, 0.95), (0.8, 0.93), (0.2, 0.5), (0.1, 0.3)]:
        rec.add({t: p_off for t in THRESHOLDS}, measured_recall=recall)
    out = rec.apply(flat_pred([0.85]))
    assert abs(out.probs[0.90][0] - 0.85) < 0.25


def test_ranking_preserved_and_monotone():
    rec = OnlineRecalibrator()
    for _ in range(6):
        rec.add({t: 0.8 for t in THRESHOLDS}, measured_recall=0.6)
    out = rec.apply(flat_pred([0.2, 0.5, 0.9]))
    for t in THRESHOLDS:
        assert np.all(np.diff(out.probs[t]) > 0)  # ranking survives
    ordered = sorted(THRESHOLDS)
    for loose, strict in zip(ordered, ordered[1:]):
        assert np.all(out.probs[loose] >= out.probs[strict] - 1e-12)
