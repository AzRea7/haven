from typing import Any

from haven.adapters.config import config
from haven.adapters.logging_utils import get_logger
from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator
from haven.adapters.sql_repo import SqlDealRepository
from haven.analysis.finance import analyze_property_financials
from haven.analysis.scoring import score_deal
from haven.analysis.valuation import summarize_deal_pricing
from haven.domain.assumptions import UnderwritingAssumptions
from haven.domain.ports import DealRepository, RentEstimator
from haven.domain.property import Property, Unit
from haven.services.validation import validate_and_prepare_payload

logger = get_logger(__name__)

def _fill_missing_rents(prop: Property, rent_estimator: RentEstimator) -> Property:
    # Fill per-unit rents if not provided
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

    # Single-door deal fallback
    if not prop.units and prop.est_market_rent is None:
        prop.est_market_rent = rent_estimator.predict_unit_rent(
            bedrooms=0.0,
            bathrooms=0.0,
            sqft=0.0,
            zipcode=prop.zipcode,
            property_type=prop.property_type,
        )

    return prop

def analyze_deal(
    raw_payload: dict[str, Any],
    rent_estimator: RentEstimator,
    repo: DealRepository | None = None,
) -> dict[str, Any]:

    # 1. Clean/normalize data
    payload = validate_and_prepare_payload(raw_payload)

    # 2. Cast dict units -> Unit models
    units_data = payload.get("units")
    if units_data:
        payload["units"] = [Unit(**u) for u in units_data]

    # 3. Build Property
    prop = Property(**payload)

    # 4. Fill missing rents
    prop = _fill_missing_rents(prop, rent_estimator)

    # 5. Assemble underwriting assumptions
    assumptions = UnderwritingAssumptions(
        vacancy_rate=config.VACANCY_RATE,
        maintenance_rate=config.MAINTENANCE_RATE,
        property_mgmt_rate=config.PROPERTY_MGMT_RATE,
        capex_rate=config.CAPEX_RATE,
        closing_cost_pct=config.DEFAULT_CLOSING_COST_PCT,
        min_dscr_good=config.MIN_DSCR_GOOD,
    )

    # 6. Compute finance & pricing
    finance = analyze_property_financials(prop, assumptions)
    sqft = float(payload.get("sqft") or 0.0)
    pricing = summarize_deal_pricing(prop, sqft=sqft, assumptions=assumptions)

    # 7. Score deal
    score = score_deal(finance)

    result = {
        "address": {
            "address": prop.address,
            "city": prop.city,
            "state": prop.state,
            "zipcode": prop.zipcode,
        },
        "property_type": prop.property_type,
        "finance": finance,
        "pricing": pricing,
        "score": score,
    }

    logger.info("deal_analyzed", extra={"context": result})

    # 8. Persist
    deal_id = None
    if repo is not None:
        deal_id = repo.save_analysis(result, raw_payload)

    if deal_id is not None:
        result["deal_id"] = deal_id
    return result

# convenient defaults
_default_repo = SqlDealRepository(uri="sqlite:///haven.db")
_default_estimator = LightGBMRentEstimator()

def analyze_deal_with_defaults(raw_payload: dict[str, Any]) -> dict[str, Any]:
    return analyze_deal(
        raw_payload=raw_payload,
        rent_estimator=_default_estimator,
        repo=_default_repo,
    )
