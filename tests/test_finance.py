import math

import pytest

from types import SimpleNamespace

from haven.domain.finance import FinanceConfig, build_scenario_metrics


def _make_baseline_norm(list_price=200_000.0, rehab_budget=0.0):
    # Only fields used by build_scenario_metrics need to exist
    return SimpleNamespace(
        list_price=list_price,
        rehab_budget=rehab_budget,
        zipcode="48009",
        property_type="single_family",
    )


def test_finance_simple_zero_expenses_zero_interest():
    """
    No taxes, insurance, maintenance, vacancy, mgmt, 0% interest.
    Check:
      - DSCR = NOI / debt
      - cap rate = NOI / price
      - CoC = NOI / equity
    """

    norm = _make_baseline_norm(list_price=200_000.0)
    cfg = FinanceConfig(
        interest_rate=0.0,
        amort_years=30,
        ltv=0.80,
        closing_cost_rate=0.0,
        rehab_contingency=0.0,
        taxes_rate=0.0,
        insurance_rate=0.0,
        vacancy_rate=0.0,
        maintenance_rate=0.0,
        property_management_rate=0.0,
    )

    rent = 1000.0  # monthly
    arv = 200_000.0

    m = build_scenario_metrics(norm, arv=arv, rent=rent, config=cfg)

    # Compute expected values
    price = 200_000.0
    loan_amount = price * 0.80
    down_payment = price - loan_amount
    n_months = 30 * 12
    # 0% interest => principal / n_months per month
    monthly_pi = loan_amount / n_months
    annual_debt = monthly_pi * 12

    gross_rent_annual = rent * 12
    noi = gross_rent_annual  # no expenses at all

    expected_dscr = noi / annual_debt
    expected_cap_rate = noi / price
    expected_coc = noi / down_payment

    assert m.noi == pytest.approx(noi)
    assert m.dscr == pytest.approx(expected_dscr, rel=1e-6)
    assert m.cap_rate == pytest.approx(expected_cap_rate, rel=1e-6)
    assert m.coc == pytest.approx(expected_coc, rel=1e-6)


def test_finance_zero_loan_amount_dscr_inf():
    norm = _make_baseline_norm(list_price=200_000.0)
    cfg = FinanceConfig(
        interest_rate=0.05,
        amort_years=30,
        ltv=0.0,  # no debt
        closing_cost_rate=0.0,
        rehab_contingency=0.0,
        taxes_rate=0.0,
        insurance_rate=0.0,
        vacancy_rate=0.0,
        maintenance_rate=0.0,
        property_management_rate=0.0,
    )

    m = build_scenario_metrics(norm, arv=200_000.0, rent=1500.0, config=cfg)

    assert math.isinf(m.dscr)
    assert m.dscr > 0


def test_finance_zero_rent_breakeven_occ_is_one():
    norm = _make_baseline_norm(list_price=200_000.0)
    cfg = FinanceConfig(
        interest_rate=0.05,
        amort_years=30,
        ltv=0.80,
        closing_cost_rate=0.0,
        rehab_contingency=0.0,
        taxes_rate=0.01,
        insurance_rate=0.003,
        vacancy_rate=0.0,
        maintenance_rate=0.0,
        property_management_rate=0.0,
    )

    m = build_scenario_metrics(norm, arv=200_000.0, rent=0.0, config=cfg)

    # With zero rent, we define breakeven occupancy as 1.0 (100%)
    assert m.breakeven_occ == pytest.approx(1.0)
    assert m.noi < 0
