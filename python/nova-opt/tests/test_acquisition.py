import numpy as np
import pytest

from nova_opt.acquisition import (
    STRATEGIES,
    nearest_threshold,
    recall_weight,
    score,
)
from nova_opt.recall import THRESHOLDS, RecallPrediction


def make_pred(p_by_threshold, conf=0.8):
    n = len(next(iter(p_by_threshold.values())))
    return RecallPrediction(
        probs={t: np.asarray(p_by_threshold[t], dtype=float) for t in THRESHOLDS},
        confidence={t: np.full(n, conf) for t in THRESHOLDS},
    )


def flat_pred(values, conf=0.8):
    return make_pred({t: values for t in THRESHOLDS}, conf)


def test_nearest_threshold():
    assert nearest_threshold(0.90) == 0.90
    assert nearest_threshold(0.85) == 0.90  # conservative: strictest that covers
    assert nearest_threshold(0.99) == 0.95  # beyond the strictest modeled
    assert nearest_threshold(0.10) == 0.25


def test_recall_weight_orders_by_feasibility():
    pred = flat_pred([0.95, 0.2])
    w = recall_weight(pred, target_recall=0.90, beta=0.0)
    assert w[0] > w[1]


def test_recall_weight_safety_blend():
    # candidate B: hopeless at 0.95 but confidently fine at 0.80 — the safety
    # term must keep it above a candidate hopeless at both
    probs = {t: [0.05, 0.05] for t in THRESHOLDS}
    probs[0.80] = [0.05, 0.95]
    probs[0.50] = [0.05, 0.99]
    probs[0.25] = [0.05, 1.0]
    pred = make_pred(probs)
    w = recall_weight(pred, target_recall=0.95, beta=0.0)
    assert w[1] > w[0]


def test_probabilities_enter_unweighted_by_confidence():
    # calibrated probabilities are used as-is: two candidates with the same
    # probability profile but different confidence get the same base weight
    high_conf = flat_pred([0.7], conf=1.0)
    low_conf = flat_pred([0.7], conf=0.2)
    assert recall_weight(high_conf, target_recall=0.90, beta=0.0)[
        0
    ] == pytest.approx(recall_weight(low_conf, target_recall=0.90, beta=0.0)[0])


def test_exploration_bonus_lifts_uncertain_candidates():
    certain = flat_pred([0.5], conf=1.0)
    uncertain = flat_pred([0.5], conf=0.0)
    w_certain = recall_weight(certain, target_recall=0.90, beta=0.2)
    w_uncertain = recall_weight(uncertain, target_recall=0.90, beta=0.2)
    # same probability, but the model knows less about the uncertain one:
    # it gets a nudge (never a veto — the base weight stays)
    assert w_certain[0] == pytest.approx(0.5)
    assert w_uncertain[0] == pytest.approx(0.7)
    assert w_uncertain[0] > w_certain[0]


def test_score_strategies():
    ehvi = np.array([1.0, 1.0])
    w_r = np.array([0.9, 0.1])
    prob = np.array([0.9, 0.1])
    cost = np.array([100.0, 1.0])

    s_bo = score(ehvi=ehvi, w_r=w_r, prob_target=prob, cost=cost,
                 gamma=1.0, strategy="bo")
    np.testing.assert_allclose(s_bo, ehvi)

    s_prune = score(ehvi=ehvi, w_r=w_r, prob_target=prob, cost=cost,
                    gamma=1.0, strategy="bo_recall", prune_min_probability=0.3)
    assert s_prune[0] == 1.0 and s_prune[1] == 0.0

    s_cost = score(ehvi=ehvi, w_r=w_r, prob_target=prob, cost=cost,
                   gamma=1.0, strategy="bo_cost")
    assert s_cost[1] > s_cost[0]  # cheap candidate wins when recall is ignored

    s_full = score(ehvi=ehvi, w_r=w_r, prob_target=prob, cost=cost,
                   gamma=1.0, strategy="full")
    np.testing.assert_allclose(s_full, ehvi * w_r / cost)

    s_rand = score(ehvi=ehvi, w_r=w_r, prob_target=prob, cost=cost,
                   gamma=1.0, strategy="random")
    np.testing.assert_allclose(s_rand, [1.0, 1.0])


def test_gamma_controls_cost_penalty():
    ehvi = np.array([1.0, 1.0])
    w_r = np.array([1.0, 1.0])
    prob = np.array([1.0, 1.0])
    cost = np.array([1000.0, 10.0])
    weak = score(ehvi=ehvi, w_r=w_r, prob_target=prob, cost=cost,
                 gamma=0.1, strategy="full")
    strong = score(ehvi=ehvi, w_r=w_r, prob_target=prob, cost=cost,
                   gamma=1.0, strategy="full")
    assert (weak[1] / weak[0]) < (strong[1] / strong[0])


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        score(ehvi=np.ones(1), w_r=np.ones(1), prob_target=np.ones(1),
              cost=np.ones(1), gamma=1.0, strategy="bogus")
    assert "full" in STRATEGIES
