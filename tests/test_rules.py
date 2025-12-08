import pytest

from types import SimpleNamespace

from haven.domain.rules import apply_rules
from .fixtures.deals import (
    cheap_cashflow_beast,
    bad_negative_cashflow,
    borderline_maybe,
)


class DummyRulesConfig:
    """
    Thresholds consistent with the earlier design:
      - strong buy: base DSCR >= 1.25, CoC >= 10%, downside DSCR >= 1.10, CoC >= 6%
      - maybe: base DSCR >= 1.10, CoC >= 5%
    """
    min_dscr_buy = 1.25
    min_coc_buy = 0.10
    min_dscr_downside = 1.10
    min_coc_downside = 0.06

    min_dscr_maybe = 1.10
    min_coc_maybe = 0.05

    uncertainty_weight = 1.0
    min_confidence_for_buy = 0.6


@pytest.fixture
def rules_cfg():
    return DummyRulesConfig()


def test_rules_cheap_cashflow_beast_is_buy_low_risk(rules_cfg):
    eval_obj = cheap_cashflow_beast()

    eval_after = apply_rules(eval_obj, rules_cfg)

    assert eval_after.label == "buy"
    assert eval_after.risk_tier == "low"
    assert eval_after.hard_flags == []
    assert eval_after.confidence > 0.5  # narrow quantile spreads


def test_rules_bad_negative_cashflow_is_pass_with_hard_flag(rules_cfg):
    eval_obj = bad_negative_cashflow()

    eval_after = apply_rules(eval_obj, rules_cfg)

    assert eval_after.label == "pass"
    assert "Downside DSCR" in " ".join(eval_after.hard_flags) or eval_after.base.dscr < 1.0
    assert eval_after.risk_tier in ("medium", "high")


def test_rules_borderline_maybe_is_maybe_medium_risk(rules_cfg):
    eval_obj = borderline_maybe()

    eval_after = apply_rules(eval_obj, rules_cfg)

    assert eval_after.label == "maybe"
    assert eval_after.risk_tier == "medium"
    assert eval_after.confidence > 0.0
