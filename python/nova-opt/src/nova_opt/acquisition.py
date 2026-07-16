"""The cost-aware, recall-weighted acquisition:

    score(x) = EHVI_qps_latency(x) * W_R(x) / C_reuse(x | artifacts)^gamma_t

W_R blends the calibrated feasibility probability at the target threshold
with a "safety" threshold one notch down (a candidate confidently above
0.80 is worth something even when the 0.95 estimate is shaky), plus an
exploration bonus where the recall model itself is uncertain:

    W_R(x) = blend(p_target, p_safety) + beta * (1 - confidence_target)

The probabilities enter *unweighted by confidence*: they are calibrated, so
expected-utility reasoning says use them as-is — multiplying by confidence
would systematically bias selection toward regions where the model is
opinionated rather than where recall is likely. Confidence appears only in
the exploration bonus (uncertain candidates get a nudge, not a veto).

gamma_t is the cost-cooled exponent the optimizer passes in: high early
(learn from cheap evaluations), annealed toward zero as the budget depletes
(a final expensive rebuild is fine if the predicted payoff justifies it).

`Strategy` switches implement the baselines the tuner is meant to be
compared against — each one zeroes out a term of the full acquisition.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from nova_opt.recall import THRESHOLDS, RecallPrediction

Strategy = Literal["random", "bo", "bo_recall", "bo_cost", "full"]

STRATEGIES: tuple[Strategy, ...] = ("random", "bo", "bo_recall", "bo_cost", "full")

# target threshold -> (safety threshold, primary weight). Anything below the
# loosest configured pair falls back to primary-only.
_SAFETY_BLEND: dict[float, tuple[float, float]] = {
    0.95: (0.80, 0.80),
    0.90: (0.80, 0.85),
    0.80: (0.50, 0.85),
    0.50: (0.25, 0.85),
}


def nearest_threshold(target_recall: float) -> float:
    """The strictest modeled threshold that still guarantees the user's
    constraint: the smallest t with t >= target (or the strictest available
    when the target exceeds them all)."""
    at_least = [t for t in THRESHOLDS if t >= target_recall]
    return min(at_least) if at_least else max(THRESHOLDS)


def recall_weight(
    pred: RecallPrediction,
    *,
    target_recall: float,
    beta: float = 0.1,
) -> np.ndarray:
    """W_R per candidate from the full multi-threshold profile."""
    t0 = nearest_threshold(target_recall)
    primary = pred.probs[t0]
    safety_t, w = _SAFETY_BLEND.get(t0, (None, 1.0))
    if safety_t is not None:
        blended = w * primary + (1.0 - w) * pred.probs[safety_t]
    else:
        blended = primary
    return blended + beta * (1.0 - pred.confidence[t0])


def score(
    *,
    ehvi: np.ndarray,
    w_r: np.ndarray,
    prob_target: np.ndarray,
    cost: np.ndarray,
    gamma: float,
    strategy: Strategy,
    prune_min_probability: float = 0.3,
) -> np.ndarray:
    """Acquisition value per candidate under the chosen strategy. `gamma` is
    the already-cooled exponent for this iteration.

    random    caller picks uniformly; scoring is a uniform constant here
              so argmax-based selection still behaves if called
    bo        EHVI only (cost- and recall-oblivious)
    bo_recall EHVI hard-pruned by the calibrated target-threshold
              probability (no soft weighting, no cost)
    bo_cost   EHVI per unit marginal cost (no recall model)
    full      EHVI * W_R / cost^gamma
    """
    if strategy == "random":
        return np.ones_like(ehvi)
    if strategy == "bo":
        return ehvi
    if strategy == "bo_recall":
        return np.where(prob_target >= prune_min_probability, ehvi, 0.0)
    cost_term = np.power(np.maximum(cost, 1e-6), gamma)
    if strategy == "bo_cost":
        return ehvi / cost_term
    if strategy == "full":
        return ehvi * w_r / cost_term
    raise ValueError(f"unknown strategy '{strategy}'; available: {STRATEGIES}")
