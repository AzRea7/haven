from typing import Dict
from haven.domain.property import Property
from haven.analysis.finance import analyze_property_financials
from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)

def _estimate_value_income_approach(noi_annual: float, market_cap_rate: float) -> float:
    """
    Commercial valuation:
    value = NOI / cap_rate
    If local cap rate is ~0.07 (7%), then value = NOI / 0.07
    """
    if market_cap_rate <= 0:
        return 0.0
    return noi_annual / market_cap_rate

def _estimate_value_residential_price_per_sqft(sqft: float, price_per_sqft: float) -> float:
    """
    Simple comparable-based heuristic.
    Later this is replaced by a (LightGBM/XGBoost) trained price model.
    """
    if sqft is None or sqft <= 0:
        return 0.0
    return sqft * price_per_sqft

def summarize_deal_pricing(
    property: Property,
    sqft: float,
    assumed_price_per_sqft: float = 200.0,
    assumed_market_cap_rate: float = 0.07
) -> Dict[str, float]:
    """
    Returns how 'good' the asking price looks compared to fair value, using
    the correct valuation style for the asset class.
    """

    fin = analyze_property_financials(property)
    ask_price = property.list_price

    if property.property_type in ["apartment_complex"]:
        # income-based valuation
        fair_value = _estimate_value_income_approach(
            noi_annual=fin["noi_annual"],
            market_cap_rate=assumed_market_cap_rate
        )
    else:
        # comp-style valuation
        fair_value = _estimate_value_residential_price_per_sqft(
            sqft=sqft,
            price_per_sqft=assumed_price_per_sqft
        )

    delta_vs_fair = ask_price - fair_value
    pct_diff = 0.0
    if fair_value > 0:
        pct_diff = delta_vs_fair / fair_value

    result = {
        "ask_price": ask_price,
        "fair_value_estimate": fair_value,
        "price_delta": delta_vs_fair,
        "price_delta_pct": pct_diff,
    }

    logger.info("summarize_deal_pricing complete",
                extra={"context": result})
    return result
