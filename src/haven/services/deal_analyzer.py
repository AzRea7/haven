from typing import Dict, Any
from haven.domain.property import Property, Unit
from haven.analysis.finance import analyze_property_financials
from haven.analysis.valuation import summarize_deal_pricing
from haven.models.rent_model import RentModel
from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)

rent_model = RentModel()

def _fill_missing_rents(prop: Property) -> Property:
    """
    If user didn't provide rent for units or whole property,
    infer using our rent model stub.
    """
    # per-unit inference
    if prop.units:
        missing_units = [u for u in prop.units if u.market_rent is None]
        if missing_units:
            # naive split for now: predict each unit individually
            for u in prop.units:
                if u.market_rent is None:
                    u.market_rent = rent_model.predict_unit_rent(
                        bedrooms=u.bedrooms or 0.0,
                        bathrooms=u.bathrooms or 0.0,
                        sqft=u.sqft or 0.0,
                        zipcode=prop.zipcode,
                        property_type=prop.property_type
                    )

    # single-door inference
    if not prop.units and prop.est_market_rent is None:
        prop.est_market_rent = rent_model.predict_unit_rent(
            bedrooms=0.0,
            bathrooms=0.0,
            sqft=0.0,
            zipcode=prop.zipcode,
            property_type=prop.property_type
        )

    return prop

def analyze_deal(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-level "analyze this listing" call.
    1. Build Property object
    2. Infer rents if missing
    3. Run finance
    4. Run valuation
    5. Return a full summary
    """

    # Turn payload["units"] (plain dicts) into Unit models if present
    units_data = payload.get("units")
    if units_data:
        payload["units"] = [Unit(**u) for u in units_data]

    prop = Property(**payload)
    prop = _fill_missing_rents(prop)

    finance = analyze_property_financials(prop)
    # Note: caller must supply sqft for valuation; easiest path is payload["sqft"]
    sqft = float(payload.get("sqft") or 0.0)
    pricing = summarize_deal_pricing(prop, sqft=sqft)

    result = {
        "finance": finance,
        "pricing": pricing,
    }

    logger.info("analyze_deal complete", extra={"context": result})
    return result
