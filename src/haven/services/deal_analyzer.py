from __future__ import annotations

from typing import Any

import pandas as pd

from haven.adapters.arv_quantile_bundle import predict_arv_quantiles
from haven.adapters.config import config
from haven.adapters.flip_classifier import FlipClassifier
from haven.adapters.logging_utils import get_logger
from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator
from haven.adapters.sql_repo import SqlDealRepository
from haven.analysis.finance import analyze_property_financials
from haven.analysis.scoring import score_deal, score_property
from haven.analysis.valuation import summarize_deal_pricing
from haven.domain.assumptions import UnderwritingAssumptions
from haven.domain.ports import DealRepository, RentEstimator
from haven.domain.property import Property, Unit
from haven.services.guardrails import apply_guardrails
from haven.services.validation import validate_and_prepare_payload

logger = get_logger(__name__)

_flip_clf = FlipClassifier()

_default_repo: DealRepository = SqlDealRepository(uri="sqlite:///haven.db")
_default_estimator: RentEstimator = LightGBMRentEstimator()

# ---------------------------------------------------------------------
# Property type rules:
# - Your upstream providers can send: Single Family, Condo, Townhouse,
#   Manufactured, Multi-Family(2-4), Apartment(5+), Land
# - Your internal model expects:
#   single_family | condo_townhome | duplex_4plex | apartment_unit | apartment_complex
#
# You said: "I dont want manufactured or townhouse or condo"
# => we HARD-REJECT those types (and also Land).
# ---------------------------------------------------------------------

# what we will not analyze (skip / reject hard)
_EXCLUDED_UPSTREAM_TYPES = {
    "condo",
    "condominium",
    "townhouse",
    "town home",
    "manufactured",
    "mobile",
    "mobile home",
    "land",
    "lot",
    "vacant land",
}

# internal accepted types (for safety checks)
_ALLOWED_INTERNAL_TYPES = {
    "single_family",
    "condo_townhome",
    "duplex_4plex",
    "apartment_unit",
    "apartment_complex",
}


def _is_missing_rent(value: float | None) -> bool:
    if value is None:
        return True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return v <= 50.0


def _coerce_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _normalize_property_type(payload: dict[str, Any]) -> str:
    """
    Normalize upstream 'property_type' variants to internal literal values.

    Also enforces your rule: reject condo/townhouse/manufactured (and land).

    Behavior:
    - If upstream sends an excluded type => raise ValueError
    - Otherwise map to one of the allowed internal types
    """
    raw = payload.get("raw") or {}
    pt = payload.get("property_type")

    # pull from payload first, otherwise try common upstream keys
    pt_raw = str(pt or "").strip()
    if not pt_raw:
        pt_raw = str(
            raw.get("propertyType")
            or raw.get("homeType")
            or raw.get("home_type")
            or raw.get("type")
            or ""
        ).strip()

    t = pt_raw.lower().strip()

    # if already internal, accept (but still enforce exclusion if it’s condo_townhome)
    if t in _ALLOWED_INTERNAL_TYPES:
        if t == "condo_townhome":
            raise ValueError("excluded property_type: condo/townhome not allowed")
        return t

    # hard exclusions
    for bad in _EXCLUDED_UPSTREAM_TYPES:
        if bad in t and t != "single_family":
            # example: "Condo", "Townhouse", "Manufactured", "Vacant Land"
            raise ValueError(f"excluded property_type: {pt_raw}")

    # now map remaining upstream descriptions
    # Single Family
    if "single" in t and "family" in t:
        return "single_family"
    if t in {"sfh", "sfr"}:
        return "single_family"

    # Multi-family (2-4 units)
    if "multi" in t and "family" in t:
        return "duplex_4plex"
    if "duplex" in t or "triplex" in t or "fourplex" in t or "4plex" in t:
        return "duplex_4plex"

    # Apartment / 5+ units (commercial multi-family / complex)
    if "apartment" in t or "complex" in t:
        return "apartment_complex"

    # If we can infer by units count (if present)
    units = payload.get("units")
    if isinstance(units, list) and units:
        # 1 unit -> single door; 2-4 -> duplex_4plex; 5+ -> apartment_complex
        n = len(units)
        if n <= 1:
            return "single_family"
        if 2 <= n <= 4:
            return "duplex_4plex"
        return "apartment_complex"

    # default
    return "single_family"


def _fill_missing_rents(prop: Property, rent_estimator: RentEstimator, payload: dict[str, Any]) -> Property:
    """
    Standardized rent fill:
    - always calls estimator with address/city/state/zipcode + beds/baths/sqft
    """
    addr = str(payload.get("address") or getattr(prop, "address", "") or "")
    city = str(payload.get("city") or getattr(prop, "city", "") or "")
    state = str(payload.get("state") or getattr(prop, "state", "") or "")
    zipcode = str(payload.get("zipcode") or getattr(prop, "zipcode", "") or "")
    ptype = str(payload.get("property_type") or getattr(prop, "property_type", "single_family") or "single_family")

    # Multi-unit: fill per unit
    if getattr(prop, "units", None):
        for u in prop.units:
            if _is_missing_rent(u.market_rent):
                u.market_rent = rent_estimator.predict_unit_rent(
                    address=addr,
                    city=city,
                    state=state,
                    zipcode=zipcode,
                    bedrooms=_coerce_float(u.bedrooms, 0.0),
                    bathrooms=_coerce_float(u.bathrooms, 0.0),
                    sqft=_coerce_float(u.sqft, 0.0),
                    property_type=ptype,
                )
        return prop

    # Single-door: fill est_market_rent
    if _is_missing_rent(getattr(prop, "est_market_rent", None)):
        b_raw = payload.get("bedrooms") or payload.get("num_bedrooms") or getattr(prop, "bedrooms", None)
        ba_raw = payload.get("bathrooms") or payload.get("num_bathrooms") or getattr(prop, "bathrooms", None)
        sqft_raw = payload.get("sqft") or payload.get("building_sqft") or payload.get("living_area") or getattr(prop, "sqft", None)

        est = rent_estimator.predict_unit_rent(
            address=addr,
            city=city,
            state=state,
            zipcode=zipcode,
            bedrooms=_coerce_float(b_raw, 0.0),
            bathrooms=_coerce_float(ba_raw, 0.0),
            sqft=_coerce_float(sqft_raw, 0.0),
            property_type=ptype,
        )
        prop.est_market_rent = est

    return prop


def _compute_flip_probability(finance: dict[str, Any], payload: dict[str, Any]) -> float | None:
    if not getattr(_flip_clf, "is_ready", False):
        return None
    try:
        features = {
            "dscr": float(finance.get("dscr", 0.0)),
            "cash_on_cash_return": float(finance.get("cash_on_cash_return", 0.0)),
            "breakeven_occupancy_pct": float(finance.get("breakeven_occupancy_pct", 0.0)),
            "price": float(payload.get("list_price") or 0.0),
            "sqft": float(payload.get("sqft") or 0.0),
            "days_on_market": float(payload.get("days_on_market") or 0.0),
        }
        return _flip_clf.predict_proba_one(features)
    except Exception as e:
        logger.warning("flip_probability_failed", extra={"error": str(e)})
        return None


def _sanitize_quantiles(q: dict[str, float] | None, fallback: float) -> dict[str, float]:
    if q is None:
        base = float(fallback)
        return {"p10": base * 0.95, "p50": base, "p90": base * 1.05}

    out: dict[str, float] = {}
    for k in ("p10", "p50", "p90"):
        v = q.get(k)
        try:
            vf = float(v)
        except (TypeError, ValueError):
            vf = float(fallback)
        if not pd.notna(vf):
            vf = float(fallback)
        out[k] = vf

    p10 = out["p10"]
    p50 = max(out["p50"], p10)
    p90 = max(out["p90"], p50)
    out["p10"], out["p50"], out["p90"] = p10, p50, p90
    return out


def analyze_deal(
    raw_payload: dict[str, Any],
    rent_estimator: RentEstimator,
    repo: DealRepository | None = None,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """
    Main analysis entrypoint.

    NEW:
    - save=False => preview mode (does not write to deals DB)
      This is required for fast /leads/from-properties bulk scoring.
    - property_type normalization + hard exclusions (no condo/townhouse/manufactured/land)
    """
    payload = validate_and_prepare_payload(raw_payload)

    # normalize strategy
    strategy_raw = str(payload.get("strategy", "hold")).lower()
    if strategy_raw == "rental":
        strategy = "hold"
    elif strategy_raw in ("hold", "flip"):
        strategy = strategy_raw
    else:
        strategy = "hold"
    payload["strategy"] = strategy

    # normalize + enforce property_type rules
    # (this will raise ValueError for excluded types)
    payload["property_type"] = _normalize_property_type(payload)

    # units
    units_data = payload.get("units")
    if units_data:
        payload["units"] = [Unit(**u) for u in units_data]

    # parse into domain model (pydantic validation happens here too)
    prop = Property(**payload)

    # rent fill
    prop = _fill_missing_rents(prop, rent_estimator, payload)

    # derive gross rent
    if getattr(prop, "units", None):
        gross_rent = sum(float(u.market_rent or 0.0) for u in prop.units)
    else:
        gross_rent = float(getattr(prop, "est_market_rent", 0.0) or 0.0)

    # underwriting assumptions
    assumptions = UnderwritingAssumptions(
        vacancy_rate=config.VACANCY_RATE,
        maintenance_rate=config.MAINTENANCE_RATE,
        property_mgmt_rate=config.PROPERTY_MGMT_RATE,
        capex_rate=config.CAPEX_RATE,
        closing_cost_pct=config.DEFAULT_CLOSING_COST_PCT,
        min_dscr_good=config.MIN_DSCR_GOOD,
    )

    finance = analyze_property_financials(prop, assumptions)
    finance["gross_monthly_rent"] = gross_rent

    list_price = float(payload.get("list_price") or 0.0)

    try:
        arv_q_raw = predict_arv_quantiles({"base": list_price})
    except Exception as exc:
        logger.warning("arv_quantile_inference_failed", extra={"error": str(exc)})
        arv_q_raw = None

    arv_q = _sanitize_quantiles(arv_q_raw, fallback=list_price)

    pricing = summarize_deal_pricing(
        prop=prop,
        sqft=float(payload.get("sqft") or 0.0),
        assumptions=assumptions,
        arv_q=arv_q,
    )

    flip_p = _compute_flip_probability(finance, payload)

    score_new = score_property(finance=finance, arv_q=arv_q, rent_q=None)
    score_legacy = score_deal(finance)

    result: dict[str, Any] = {
        "address": {"address": prop.address, "city": prop.city, "state": prop.state, "zipcode": prop.zipcode},
        "property_type": prop.property_type,
        "strategy": strategy,
        "finance": finance,
        "pricing": pricing,
        "score": score_new,
        "score_legacy": score_legacy,
        "flip_p_good": flip_p,
        "arv_q": arv_q,
    }

    result = apply_guardrails(payload=payload, result=result)

    # Persistence: only if save=True AND repo provided
    deal_id: int | None = None
    if save and repo is not None:
        try:
            deal_id = repo.save_analysis(result, raw_payload)
        except Exception as e:
            # Do NOT break analysis if DB persistence fails (important for stability)
            logger.warning("save_analysis_failed", extra={"error": str(e)})
            deal_id = None

    if deal_id is not None:
        result["deal_id"] = deal_id

    return result


def analyze_deal_with_defaults(raw_payload: dict[str, Any]) -> dict[str, Any]:
    # FIXED: no trailing comma; returns dict, not tuple
    return analyze_deal(raw_payload=raw_payload, rent_estimator=_default_estimator, repo=_default_repo, save=True)
