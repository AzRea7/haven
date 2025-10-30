from typing import Dict, Any
from haven.domain.property import Property, Unit
from haven.analysis.finance import analyze_property_financials
from haven.analysis.valuation import summarize_deal_pricing
from haven.domain.assumptions import UnderwritingAssumptions
from haven.domain.ports import RentEstimator
from haven.adapters.config import config
from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)

def _fill_missing_rents(prop: Property, rent_estimator: RentEstimator) -> Property:
    if prop.units:
        for u in prop.units:
            if u.market_rent is None:
                u.market_rent = rent_estimator.predict_unit_rent(
                    bedrooms=u.bedrooms or 0.0,
                    bathrooms=u.bathrooms or 0.0,
                    sqft=u.sqft or 0.0,
                    zipcode=prop.zipcode,
                    property_type=prop.property_type,
                )
    if not prop.units and prop.est_market_rent is None:
        prop.est_market_rent = rent_estimator.predict_unit_rent(
            bedrooms=0.0,
            bathrooms=0.0,
            sqft=0.0,
            zipcode=prop.zipcode,
            property_type=prop.property_type,
        )
    return prop

def analyze_deal(payload: Dict[str, Any], rent_estimator: RentEstimator) -> Dict[str, Any]:
    # normalize units from raw dicts
    units_data = payload.get("units")
    if units_data:
        payload["units"] = [Unit(**u) for u in units_data]

    prop = Property(**payload)
    prop = _fill_missing_rents(prop, rent_estimator)

    # build underwriting assumptions from config
    assumptions = UnderwritingAssumptions(
        vacancy_rate=config.VACANCY_RATE,
        maintenance_rate=config.MAINTENANCE_RATE,
        property_mgmt_rate=config.PROPERTY_MGMT_RATE,
        capex_rate=config.CAPEX_RATE,
        closing_cost_pct=config.DEFAULT_CLOSING_COST_PCT,
        min_dscr_good=config.MIN_DSCR_GOOD,
    )

    finance = analyze_property_financials(prop, assumptions)
    sqft = float(payload.get("sqft") or 0.0)
    pricing = summarize_deal_pricing(prop, sqft=sqft, assumptions=assumptions)

    logger.info("analyze_property_financials complete", extra={"context": finance})
    logger.info("summarize_deal_pricing complete", extra={"context": pricing})

    return {
        "finance": finance,
        "pricing": pricing,
    }
