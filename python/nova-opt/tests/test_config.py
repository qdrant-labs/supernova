import pytest

from pydantic import ValidationError

from nova_opt.config import CostPriorsConfig, OptimizerConfig


def test_budget_seconds_must_be_positive():
    """budget_seconds=0 makes the optimizer's `while spent < budget` guard
    false from the start, silently producing zero trials -- reject it at
    config load instead."""
    with pytest.raises(ValidationError):
        OptimizerConfig(budget_seconds=0)
    with pytest.raises(ValidationError):
        OptimizerConfig(budget_seconds=-1)
    OptimizerConfig(budget_seconds=1.0)  # positive is fine


def test_max_evaluations_and_n_candidates_must_be_positive():
    with pytest.raises(ValidationError):
        OptimizerConfig(max_evaluations=0)
    with pytest.raises(ValidationError):
        OptimizerConfig(n_candidates=0)


def test_cost_priors_must_be_positive():
    """CostModel seeds its EMA directly from these and clamps ridge-
    regression predictions to a band around the EMA -- zero or negative
    collapses or inverts that band."""
    with pytest.raises(ValidationError):
        CostPriorsConfig(quant_s=0)
    with pytest.raises(ValidationError):
        CostPriorsConfig(layout_s=-100)


def test_ema_alpha_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        CostPriorsConfig(ema_alpha=0.0)
    with pytest.raises(ValidationError):
        CostPriorsConfig(ema_alpha=1.5)
    CostPriorsConfig(ema_alpha=1.0)  # inclusive upper bound
