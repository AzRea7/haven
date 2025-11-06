
from haven.analysis.finance import analyze_property_financials
from haven.domain.assumptions import UnderwritingAssumptions
from haven.domain.property import Property


def _estimate_value_income_approach(noi_annual: float, market_cap_rate: float) -> float:
    if market_cap_rate <= 0:
        return 0.0
    return noi_annual / market_cap_rate

def _estimate_value_residential_price_per_sqft(sqft: float, price_per_sqft: float) -> float:
    if sqft is None or sqft <= 0:
        return 0.0
    return sqft * price_per_sqft

def summarize_deal_pricing(
    property: Property,
    sqft: float,
    assumptions: UnderwritingAssumptions,
    assumed_price_per_sqft: float = 200.0,
    assumed_market_cap_rate: float = 0.07,
) -> dict[str, float]:
    """
    Decide if this deal looks under/overpriced.
    Income approach for apartment_complex.
    Price/sqft heuristic otherwise.
    """
    fin = analyze_property_financials(property, assumptions)
    ask_price = property.list_price

    if property.property_type in ["apartment_complex"]:
        fair_value = _estimate_value_income_approach(
            noi_annual=fin["noi_annual"],
            market_cap_rate=assumed_market_cap_rate,
        )
    else:
        fair_value = _estimate_value_residential_price_per_sqft(
            sqft=sqft,
            price_per_sqft=assumed_price_per_sqft,
        )

    delta_vs_fair = ask_price - fair_value
    pct_diff = (delta_vs_fair / fair_value) if fair_value > 0 else 0.0

    return {
        "ask_price": ask_price,
        "fair_value_estimate": fair_value,
        "price_delta": delta_vs_fair,
        "price_delta_pct": pct_diff,
    }
