# tests/test_underwriting_scenarios.py

from types import SimpleNamespace

from hypothesis import given, strategies as st

from haven.domain.finance import FinanceConfig, build_scenario_metrics


def _baseline_norm():
    return SimpleNamespace(
        list_price=250_000.0,
        rehab_budget=0.0,
        zipcode="48009",
        property_type="single_family",
    )


def _baseline_config():
    return FinanceConfig(
        interest_rate=0.06,
        amort_years=30,
        ltv=0.80,
        closing_cost_rate=0.03,
        rehab_contingency=0.10,
        taxes_rate=0.015,
        insurance_rate=0.003,
        vacancy_rate=0.05,
        maintenance_rate=0.08,
        property_management_rate=0.08,
    )


@given(
    rent=st.floats(min_value=500.0, max_value=4000.0),
    delta=st.floats(min_value=50.0, max_value=500.0),
)
def test_higher_rent_improves_metrics(rent, delta):
    norm = _baseline_norm()
    cfg = _baseline_config()

    m1 = build_scenario_metrics(norm, arv=300_000.0, rent=rent, config=cfg)
    m2 = build_scenario_metrics(norm, arv=300_000.0, rent=rent + delta, config=cfg)

    # With everything else fixed, more rent should not hurt DSCR, CoC, or cap rate
    assert m2.dscr >= m1.dscr
    assert m2.coc >= m1.coc
    assert m2.cap_rate >= m1.cap_rate


@given(
    price=st.floats(min_value=100_000.0, max_value=600_000.0),
    delta=st.floats(min_value=10_000.0, max_value=150_000.0),
)
def test_higher_price_reduces_cap_and_coc(price, delta):
    """
    If NOI is held constant but you pay more for the asset, cap & CoC should go down.
    We approximate this here by adjusting list_price while keeping rent fixed.
    """
    cfg = _baseline_config()

    norm1 = SimpleNamespace(
        list_price=price,
        rehab_budget=0.0,
        zipcode="48009",
        property_type="single_family",
    )
    norm2 = SimpleNamespace(
        list_price=price + delta,
        rehab_budget=0.0,
        zipcode="48009",
        property_type="single_family",
    )

    # Same rent in both cases
    rent = 2000.0

    m1 = build_scenario_metrics(norm1, arv=300_000.0, rent=rent, config=cfg)
    m2 = build_scenario_metrics(norm2, arv=300_000.0, rent=rent, config=cfg)

    # With fixed income and higher price, cap and CoC should be lower or equal
    assert m2.cap_rate <= m1.cap_rate
    assert m2.coc <= m1.coc
