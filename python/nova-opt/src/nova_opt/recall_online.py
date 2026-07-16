"""Online recall *surrogate*: a run-local GP that learns measured recall as a
function of config, on top of (never instead of) the global Platt
recalibration in `online.py`.

`OnlineRecalibrator` fits one 2-parameter curve shared by every candidate —
it can correct "the classifier is systematically over/under-confident on
this workload" but not "ef_search=16 is worse than the offline model
thinks, ef_search=256 is fine". That's a spatial pattern, and a spatial
pattern needs a spatial model — the same reason `surrogate.py` gives QPS and
latency real per-config GPs instead of a single scalar correction.

`RecallSurrogate` regresses `logit(measured_recall)` on the same encoded
candidate vector the QPS/latency GPs use (`surrogate.CandidateEncoder`), via
the same GP kernel (`surrogate.make_gp`). Because it is a genuine
regression rather than five independent per-threshold classifiers,
`P(recall(x) >= t) = Phi((mu(x) - logit(t)) / sigma(x))` is computable for
*any* t from one fit, and — since it is the same Gaussian CDF evaluated at
different points — automatically monotone non-increasing in t. No
monotone-constraint machinery needed here, unlike the offline classifier.

The GP's own posterior uncertainty decides how much to trust it: near
observed points sigma is small (trust the GP); far from data sigma reverts
to the kernel's prior variance (defer entirely to the offline-classifier +
Platt-recalibrated prediction, unchanged). This is the same mechanism that
already makes the QPS/latency GPs behave sensibly away from data — no new
hand-tuned trust schedule, only a readiness gate (too few observations, or
no feature variation among them, and the GP is not consulted at all).

This surrogate is local to one optimizer run and is never trained on
data.csv — it feeds only the W_R feasibility weight, exactly like
`OnlineRecalibrator`. It must never enter EHVI or the Pareto front (see
DESIGN.md invariant #4 and its recall-GP counterpart).
"""

from __future__ import annotations

import numpy as np

from scipy.stats import norm

from nova_opt.recall import THRESHOLDS, RecallPrediction
from nova_opt.space import Candidate, SpaceAxes
from nova_opt.surrogate import CandidateEncoder, make_gp

_EPS = 1e-6


def _logit(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class RecallSurrogate:
    """Run-local GP over measured recall, consulted only once it has enough
    evidence to say something a global scalar correction can't."""

    def __init__(self, axes: SpaceAxes, *, min_observations: int = 5, seed: int = 0):
        self.encoder = CandidateEncoder(axes)
        self.min_observations = min_observations
        self.seed = seed
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._gp = None

    @property
    def ready(self) -> bool:
        return self._gp is not None

    def add(self, cand: Candidate, recall: float) -> None:
        self._X.append(self.encoder.encode([cand])[0])
        self._y.append(float(_logit(recall)))

    def fit(self) -> None:
        if len(self._y) < self.min_observations:
            return
        x = np.stack(self._X)
        if len(np.unique(x, axis=0)) < 2:
            return  # no feature variation observed — nothing to fit
        self._gp = make_gp(self.seed, x.shape[1]).fit(x, np.array(self._y))

    def _predict_logit(self, cands: list[Candidate]) -> tuple[np.ndarray, np.ndarray]:
        if not self.ready:
            raise RuntimeError("recall surrogate not fitted yet")
        x = self.encoder.encode(cands)
        return self._gp.predict(x, return_std=True)

    def _prior_std(self) -> float:
        """The posterior std a candidate infinitely far from every observed
        point would have — i.e. the GP's belief with zero local evidence.
        Stationary kernels (ours is) have a constant diagonal, so this needs
        no probing: k(x, x) is the same at every x."""
        dummy = np.zeros((1, self._gp.X_train_.shape[1]))
        y_train_std = getattr(self._gp, "_y_train_std", 1.0)
        prior_var = self._gp.kernel_.diag(dummy)[0] * y_train_std**2
        return float(np.sqrt(max(prior_var, 1e-12)))

    def prob_feasible(
        self, cands: list[Candidate], thresholds: tuple[float, ...] = THRESHOLDS
    ) -> dict[float, np.ndarray]:
        """P(recall(x) >= t) per threshold, from the same fit — automatically
        monotone non-increasing in t (same Gaussian CDF, different points)."""
        mu, sd = self._predict_logit(cands)
        sd = np.maximum(sd, _EPS)
        return {t: norm.cdf((mu - _logit(t)) / sd) for t in thresholds}

    def confidence(self, cands: list[Candidate]) -> np.ndarray:
        """How much to trust the GP here: 1 near observed points, ->0 far
        away where its posterior reverts to the (uninformative) prior."""
        _, sd = self._predict_logit(cands)
        return np.clip(1.0 - sd / self._prior_std(), 0.0, 1.0)

    def blend(self, cands: list[Candidate], pred: RecallPrediction) -> RecallPrediction:
        """Blend the offline+Platt prediction with the GP's own view,
        weighted by local confidence in the GP. A no-op until `ready`
        (too few measurements, or no feature variation among them yet).
        `confidence` on the result passes through unchanged — it is the
        offline model's VA/OOD signal, not something this surrogate owns."""
        if not self.ready:
            return pred
        alpha = self.confidence(cands)
        gp_probs = self.prob_feasible(cands, THRESHOLDS)
        probs = {
            t: alpha * gp_probs[t] + (1.0 - alpha) * pred.probs[t] for t in THRESHOLDS
        }
        return RecallPrediction(probs=probs, confidence=pred.confidence)
