import numpy as np
import pytest

from nova_opt.venn_abers import VennAbers


@pytest.fixture(scope="module")
def va():
    rng = np.random.default_rng(0)
    # scores correlate with labels through a miscalibrated (overconfident) link
    scores = rng.uniform(0, 1, size=4000)
    p_true = 0.5 + 0.5 * np.tanh(3 * (scores - 0.5))  # true prob, sharper
    labels = (rng.uniform(size=4000) < p_true).astype(float)
    return VennAbers.fit(scores, labels, seed=0)


def test_interval_orders_and_bounds(va):
    s = np.linspace(0, 1, 50)
    p0, p1 = va.interval(s)
    assert np.all(p0 <= p1 + 1e-12)
    assert np.all((0 <= p0) & (p1 <= 1))


def test_merged_probability_inside_interval(va):
    s = np.linspace(0.05, 0.95, 30)
    p, width = va.predict(s)
    p0, p1 = va.interval(s)
    assert np.all(p >= p0 - 1e-9) and np.all(p <= p1 + 1e-9)
    assert np.all(width >= -1e-12)


def test_calibration_is_monotone(va):
    s = np.linspace(0, 1, 200)
    p, _ = va.predict(s)
    assert np.all(np.diff(p) >= -1e-9)


def test_roughly_calibrated_on_held_out(va):
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, size=4000)
    p_true = 0.5 + 0.5 * np.tanh(3 * (scores - 0.5))
    labels = (rng.uniform(size=4000) < p_true).astype(float)
    p, _ = va.predict(scores)
    # binned calibration error well under the raw scores' miscalibration
    bins = np.clip((p * 10).astype(int), 0, 9)
    err = 0.0
    raw_err = 0.0
    for b in range(10):
        m = bins == b
        if m.sum() < 50:
            continue
        err = max(err, abs(p[m].mean() - labels[m].mean()))
        raw_err = max(raw_err, abs(scores[m].mean() - labels[m].mean()))
    assert err < 0.1
    assert err < raw_err


def test_width_grows_with_sparsity():
    rng = np.random.default_rng(2)
    scores = rng.uniform(0, 1, size=60)
    labels = (rng.uniform(size=60) < scores).astype(float)
    small = VennAbers.fit(scores, labels)
    big = VennAbers.fit(
        np.repeat(scores, 50), np.repeat(labels, 50)
    )
    s = np.linspace(0.1, 0.9, 20)
    assert small.predict(s)[1].mean() > big.predict(s)[1].mean()


def test_dict_roundtrip(va):
    clone = VennAbers.from_dict(va.to_dict())
    s = np.linspace(0, 1, 40)
    np.testing.assert_allclose(va.predict(s)[0], clone.predict(s)[0])
