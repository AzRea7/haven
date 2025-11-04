from haven.domain.property import Property
from haven.domain.assumptions import UnderwritingAssumptions
from haven.analysis.finance import analyze_property_financials

def test_single_family_finance_smoke():
    prop = Property(
        property_type="single_family",
        address="123 Main St",
        city="Detroit",
        state="MI",
        zipcode="48201",
        list_price=200000,
        down_payment_pct=0.20,
        interest_rate_annual=0.065,
        loan_term_years=30,
        taxes_annual=3000,
        insurance_annual=1200,
        hoa_monthly=0,
        est_market_rent=1800.0,
        units=None,
    )

    assumptions = UnderwritingAssumptions(
        vacancy_rate=0.05,
        maintenance_rate=0.08,
        property_mgmt_rate=0.10,
        capex_rate=0.05,
        closing_cost_pct=0.03,
        min_dscr_good=1.20,
    )

    finance = analyze_property_financials(prop, assumptions)

    assert finance["noi_annual"] > 0
    assert finance["dscr"] > 0
    assert "cash_on_cash_return" in finance
