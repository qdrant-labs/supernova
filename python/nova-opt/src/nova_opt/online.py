"""Online recall feedback: recalibrate the offline classifier against the
recalls actually measured on the *current* workload.

The offline classifier's dominant failure mode on a new dataset (visible in
its own LODO evaluation) is miscalibration under dataset shift while the
*ranking* of configs stays useful. So the correction is a per-threshold
Platt recalibration of the offline probability,

    p_online = sigmoid(a + b * logit(p_offline)),

fitted by penalized MLE with a shrinkage prior centered at the identity
(a=0, b=1). With zero measurements it returns the offline probabilities
unchanged; each measured (config, recall) pair pulls (a, b) toward what this
workload actually does, with the prior keeping the first few observations
from whipsawing the calibration. After ~10 measurements the tuner
effectively trusts its own eyes over data.csv — which is the point.
"""

from __future__ import annotations

import logging

import numpy as np

from nova_opt.recall import THRESHOLDS, RecallPrediction

log = logging.getLogger("nova_opt")

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_platt(
    p_off: np.ndarray, y: np.ndarray, prior_strength: float
) -> tuple[float, float]:
    """Penalized-MLE (a, b) for y ~ Bernoulli(sigmoid(a + b·logit(p_off)))
    with prior (a - 0)^2 + (b - 1)^2 scaled by `prior_strength`."""
    from scipy.optimize import minimize

    z = _logit(p_off)

    def nll(params: np.ndarray) -> float:
        a, b = params
        q = _sigmoid(a + b * z)
        q = np.clip(q, _EPS, 1 - _EPS)
        ll = np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
        return -ll + 0.5 * prior_strength * (a**2 + (b - 1.0) ** 2)

    res = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    if not res.success:
        log.warning("Platt recalibration MLE did not converge: %s", res.message)
    a, b = res.x
    # a negative slope would invert the offline ranking — the one thing we
    # trust; the prior makes this near-impossible, the clip makes it certain
    return float(a), float(max(b, 0.05))


class OnlineRecalibrator:
    def __init__(self, *, prior_strength: float = 4.0):
        self.prior_strength = prior_strength
        # per threshold: list of (offline probability, binary outcome)
        self._obs: dict[float, list[tuple[float, int]]] = {t: [] for t in THRESHOLDS}

    @property
    def n_observations(self) -> int:
        return len(self._obs[THRESHOLDS[0]])

    def add(self, offline_probs: dict[float, float], measured_recall: float) -> None:
        """Record one measurement: the offline probabilities predicted for
        the evaluated config, and the recall it actually achieved."""
        for t in THRESHOLDS:
            self._obs[t].append((float(offline_probs[t]), int(measured_recall >= t)))

    def apply(self, pred: RecallPrediction) -> RecallPrediction:
        """Recalibrated copy of `pred`. Confidence passes through — the VA/
        ensemble/OOD uncertainty is about the offline model, and the prior
        already grades how much the online correction can move things."""
        if self.n_observations == 0:
            return pred
        probs: dict[float, np.ndarray] = {}
        for t in THRESHOLDS:
            obs = np.array(self._obs[t], dtype=float)
            a, b = _fit_platt(obs[:, 0], obs[:, 1], self.prior_strength)
            probs[t] = _sigmoid(a + b * _logit(pred.probs[t]))
        # per-threshold refits can jitter cross-threshold monotonicity
        ordered = sorted(THRESHOLDS, reverse=True)
        for prev, t in zip(ordered, ordered[1:]):
            probs[t] = np.maximum(probs[t], probs[prev])
        return RecallPrediction(probs=probs, confidence=pred.confidence)
