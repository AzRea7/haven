# src/haven/services/deal_analyzer.py
from __future__ import annotations

from typing import Any

from haven.adapters.config import config
from haven.adapters.logging_utils import get_logger
from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator
from haven.adapters.sql_repo import SqlDealRepository
from haven.adapters.flip_classifier import FlipClassifier
from haven.analysis.finance import analyze_property_financials
from haven.analysis.scoring import score_deal, score_property
from haven.analysis.valuation import summarize_deal_pricing
from haven.domain.assumptions import UnderwritingAssumptions
from haven.domain.ports import DealRepository, RentEstimator
from haven.domain.property import Property, Unit
from haven.services.validation import validate_and_prepare_payload

logger = get_logger(__name__)

# Optional flip classifier (safe no-op if no model artifact)
_flip_clf = FlipClassifier()

# Defaults used by tests and simple consumers
_default_repo: DealRepository = SqlDealRepository(uri="sqlite:///haven.db")
_default_estimator: RentEstimator = LightGBMRentEstimator()


def _fill_missing_rents(prop: Property, rent_estimator: RentEstimator) -> Property:
    """
    Ensure rents are populated:

    - For multi-unit deals: fill missing unit.market_rent using the rent_estimator.
    - For single-door deals: if est_market_rent is missing, estimate based on whatever
      attributes are available on the Property (sqft, beds, baths if present).
    """
    # Multi-unit: per-unit inference
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

    # Single-door: property-level inference
    if not prop.units and prop.est_market_rent is None:
        bedrooms = getattr(prop, "bedrooms", 0.0) or 0.0
        bathrooms = getattr(prop, "bathrooms", 0.0) or 0.0
        sqft = getattr(prop, "sqft", 0.0) or 0.0

        prop.est_market_rent = rent_estimator.predict_unit_rent(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            sqft=sqft,
            zipcode=prop.zipcode,
            property_type=prop.property_type,
        )

    return prop


def _compute_flip_probability(finance: dict[str, Any], payload: dict[str, Any]) -> float | None:
    """
    Minimal, robust integration with FlipClassifier.
    Returns probability in [0,1] or None if model not available.
    """
    if not getattr(_flip_clf, "is_ready", False):
        return None

    try:
        features = {
            # Must align with features used in scripts/train_flip.py
            "dscr": float(finance.get("dscr", 0.0)),
            "cash_on_cash_return": float(finance.get("cash_on_cash_return", 0.0)),
            "breakeven_occupancy_pct": float(finance.get("breakeven_occupancy_pct", 0.0)),
            "price": float(payload.get("list_price") or 0.0),
            "sqft": float(payload.get("sqft") or 0.0),
            "days_on_market": float(payload.get("days_on_market") or 0.0),
        }
        return _flip_clf.predict_proba_one(features)
    except Exception as e:
        logger.warning(
            "flip_probability_failed",
            extra={"context": {"error": str(e)}},
        )
        return None


def _attach_suggestion(score: dict, finance: dict) -> dict:
    """
    Backward-compatible suggestion field for tests and UI.

    Tests expect suggestion in:
      {"buy", "maybe negotiate", "maybe (low DSCR)", "pass"}.
    """
    label = str(score.get("label", "pass")).lower()
    dscr = float(finance.get("dscr", 0.0))

    if label == "buy":
        suggestion = "buy"
    elif label == "maybe":
        if dscr < 1.15:
            suggestion = "maybe (low DSCR)"
        else:
            suggestion = "maybe negotiate"
    else:
        suggestion = "pass"

    s = dict(score)
    s["suggestion"] = suggestion
    return s


def analyze_deal(
    raw_payload: dict[str, Any],
    rent_estimator: RentEstimator,
    repo: DealRepository | None = None,
) -> dict[str, Any]:
    """
    Core deal analysis entrypoint.

    - Normalizes payload.
    - Builds Property and fills missing rents.
    - Applies underwriting assumptions.
    - Computes financials & pricing.
    - Computes:
        * score: new risk-adjusted score (score_property)
        * score_legacy: old-style score_deal output
        * flip_p_good: optional flip classifier probability
    """
    # 1. Validate + normalize + apply defaults
    payload = validate_and_prepare_payload(raw_payload)

    # 2. Cast dict units -> Unit models
    units_data = payload.get("units")
    if units_data:
        payload["units"] = [Unit(**u) for u in units_data]

    # 3. Build Property
    prop = Property(**payload)

    # 4. Ensure rents present
    prop = _fill_missing_rents(prop, rent_estimator)

    # 5. Underwriting assumptions
    assumptions = UnderwritingAssumptions(
        vacancy_rate=config.VACANCY_RATE,
        maintenance_rate=config.MAINTENANCE_RATE,
        property_mgmt_rate=config.PROPERTY_MGMT_RATE,
        capex_rate=config.CAPEX_RATE,
        closing_cost_pct=config.DEFAULT_CLOSING_COST_PCT,
        min_dscr_good=config.MIN_DSCR_GOOD,
    )

    # 6. Financials & pricing
    finance = analyze_property_financials(prop, assumptions)
    sqft = float(payload.get("sqft") or 0.0)
    pricing = summarize_deal_pricing(prop, sqft=sqft, assumptions=assumptions)

    # 7. Flip probability (optional)
    flip_p = _compute_flip_probability(finance, payload)

    # 8. New risk-adjusted score
    dom = float(payload.get("days_on_market") or 0.0)
    strategy = payload.get("strategy") or "hold"

    score_new = score_property(
        finance=finance,
        arv_q=None,          # hook ARV quantiles here later
        rent_q=None,         # hook rent quantiles here later
        dom=dom,
        strategy=strategy,
        flip_p_good=flip_p,
    )
    # Ensure suggestion is present for compatibility
    score_with_suggestion = _attach_suggestion(score_new, finance)

    # 9. Legacy score for debugging/backwards compatibility
    score_legacy = score_deal(finance)

    result: dict[str, Any] = {
        "address": {
            "address": prop.address,
            "city": prop.city,
            "state": prop.state,
            "zipcode": prop.zipcode,
        },
        "property_type": prop.property_type,
        "finance": finance,
        "pricing": pricing,
        "score": score_with_suggestion,
        "score_legacy": score_legacy,
        "flip_p_good": flip_p,
    }

    logger.info("deal_analyzed", extra={"context": result})

    # 10. Persist if repo provided
    deal_id: int | None = None
    if repo is not None:
        deal_id = repo.save_analysis(result, raw_payload)

    if deal_id is not None:
        result["deal_id"] = deal_id

    return result


def analyze_deal_with_defaults(raw_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Public helper using default SQL repo and rent estimator.
    """
    return analyze_deal(
        raw_payload=raw_payload,
        rent_estimator=_default_estimator,
        repo=_default_repo,
    )
