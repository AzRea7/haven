from __future__ import annotations

from types import SimpleNamespace
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
from haven.domain.finance import build_scenario_metrics
from haven.domain.ports import DealRepository, RentEstimator
from haven.domain.property import Property, Unit
from haven.domain.rules import apply_rules
from haven.domain.underwriting import DealEvaluation
from haven.features.common_features import build_property_features
from haven.services.guardrails import apply_guardrails
from haven.services.validation import validate_and_prepare_payload

logger = get_logger(__name__)

# Optional flip classifier (safe no-op if no model artifact)
_flip_clf = FlipClassifier()

# Defaults used by tests and simple consumers
_default_repo: DealRepository = SqlDealRepository(uri="sqlite:///haven.db")
_default_estimator: RentEstimator = LightGBMRentEstimator()


def _fill_missing_rents(
    prop: Property,
    rent_estimator: RentEstimator,
    payload: dict[str, Any],
) -> Property:
    """
    Ensure rents are populated:

    - For multi-unit deals: fill missing or zero-ish unit.market_rent using the rent_estimator.
    - For single-door deals: if est_market_rent is missing or ~0, estimate based on property attrs
      OR raw payload fields (bedrooms, bathrooms, sqft).
    """

    def _is_missing_rent(value: float | None) -> bool:
        # Treat None, 0, or tiny values as "unknown"
        if value is None:
            return True
        try:
            v = float(value)
        except (TypeError, ValueError):
            return True
        return v <= 50.0  # basically "no rent" / placeholder

    def _coerce_float(val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    logger.info(
        "fill_missing_rents_start",
        extra={
            "has_units": bool(getattr(prop, "units", None)),
            "est_market_rent": getattr(prop, "est_market_rent", None),
            "bedrooms": getattr(prop, "bedrooms", None),
            "bathrooms": getattr(prop, "bathrooms", None),
            "sqft": getattr(prop, "sqft", None),
            "zipcode": prop.zipcode,
            "property_type": prop.property_type,
        },
    )

    # --- Multi-unit: per-unit inference ---
    if getattr(prop, "units", None):
        for u in prop.units:
            logger.info(
                "unit_before_rent_fill",
                extra={
                    "bedrooms": u.bedrooms,
                    "bathrooms": u.bathrooms,
                    "sqft": u.sqft,
                    "market_rent": u.market_rent,
                },
            )
            if _is_missing_rent(u.market_rent):
                u.market_rent = rent_estimator.predict_unit_rent(
                    bedrooms=_coerce_float(u.bedrooms, 0.0),
                    bathrooms=_coerce_float(u.bathrooms, 0.0),
                    sqft=_coerce_float(u.sqft, 0.0),
                    zipcode=prop.zipcode,
                    property_type=prop.property_type,
                )
                logger.info(
                    "unit_after_rent_fill",
                    extra={"market_rent": u.market_rent},
                )

    # --- Single-door: property-level inference ---
    if not getattr(prop, "units", None) and _is_missing_rent(
        getattr(prop, "est_market_rent", None)
    ):
        # IMPORTANT: pull features from payload first, fall back to Property
        b_raw = (
            payload.get("bedrooms")
            or payload.get("num_bedrooms")
            or getattr(prop, "bedrooms", None)
        )
        ba_raw = (
            payload.get("bathrooms")
            or payload.get("num_bathrooms")
            or getattr(prop, "bathrooms", None)
        )
        sqft_raw = (
            payload.get("sqft")
            or payload.get("building_sqft")
            or payload.get("living_area")
            or getattr(prop, "sqft", None)
        )

        bedrooms = _coerce_float(b_raw, 0.0)
        bathrooms = _coerce_float(ba_raw, 0.0)
        sqft = _coerce_float(sqft_raw, 0.0)

        est_rent = rent_estimator.predict_unit_rent(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            sqft=sqft,
            zipcode=prop.zipcode,
            property_type=prop.property_type,
        )
        prop.est_market_rent = est_rent

        logger.info(
            "single_door_rent_filled",
            extra={
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft": sqft,
                "est_market_rent": est_rent,
            },
        )

    return prop


def _compute_flip_probability(
    finance: dict[str, Any],
    payload: dict[str, Any],
) -> float | None:
    """
    Minimal, robust integration with FlipClassifier.
    Returns probability in [0,1] or None if model not available.
    """
    if not getattr(_flip_clf, "is_ready", False):
        return None

    try:
        features = {
            # Must align with features used in train_flip.py / audit_flip_classifier
            "dscr": float(finance.get("dscr", 0.0)),
            "cash_on_cash_return": float(finance.get("cash_on_cash_return", 0.0)),
            "breakeven_occupancy_pct": float(
                finance.get("breakeven_occupancy_pct", 0.0)
            ),
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


def _attach_suggestion_and_rank(
    score: dict[str, Any],
    finance: dict[str, Any],
    pricing: dict[str, Any],
    strategy: str,
    flip_p_good: float | None,
    arv_q: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Turn the neutral score dict + finance/pricing into a strategy-aware
    rank_score and suggestion/label.

    Key points:
      - For flips, spread is computed vs ARV median (p50/q50) when available.
      - Flip rank is driven by spread + flip probability, with DSCR/CoC
        acting as stabilizers.
    """
    dscr = float(finance.get("dscr", 0.0) or 0.0)
    coc = float(finance.get("cash_on_cash_return", 0.0) or 0.0)
    breakeven = float(finance.get("breakeven_occupancy_pct", 0.0) or 0.0)
    dom = float(finance.get("days_on_market", 0.0) or 0.0)

    ask_price = float(pricing.get("ask_price") or 0.0)

    # --- 1. Start from pricing's generic spread ----------------------------
    price_delta_pct = float(pricing.get("price_delta_pct") or 0.0)

    # If we have ARV, compute an ARV-based spread and store it explicitly.
    arv_price_delta_pct: float | None = None
    if arv_q is not None:
        # Support both p*/q* layouts
        def pick(*names: str) -> float | None:
            for n in names:
                if n in arv_q and arv_q[n] is not None:
                    try:
                        return float(arv_q[n])
                    except (TypeError, ValueError):
                        continue
            return None

        p50 = pick("p50", "q50", "median")
        if p50 is not None and p50 > 0 and ask_price > 0:
            arv_fair_value = float(p50)
            arv_price_delta = ask_price - arv_fair_value
            arv_price_delta_pct = arv_price_delta / arv_fair_value

            pricing["arv_fair_value_estimate"] = arv_fair_value
            pricing["arv_price_delta"] = float(arv_price_delta)
            pricing["arv_price_delta_pct"] = float(arv_price_delta_pct)

    # For flips, prefer ARV-based spread if available
    if strategy == "flip" and arv_price_delta_pct is not None:
        price_delta_pct = arv_price_delta_pct

    # Spread is defined so that *better deals* (cheaper vs fair value)
    # have larger positive spread_score.
    spread_score = -price_delta_pct  # ask < fair => negative delta => positive spread_score

    flip_p = float(flip_p_good or 0.0)

    # --- 2. Strategy-aware rank_score --------------------------------------
    if strategy == "hold":
        # Rental / buy-and-hold mode: prioritize DSCR & CoC,
        # penalize fragile breakeven, gently penalize long DOM.
        penalty = max(breakeven - 1.0, 0.0)
        dom_penalty = min(dom / 365.0, 1.0) * 0.5  # up to -0.5 for very stale listings

        rank_score = dscr + 6.0 * coc - 0.25 * penalty - dom_penalty

        suggestion = "buy" if rank_score >= 2.0 and dscr >= 1.2 and coc >= 0.08 else "watch"
        if dscr < 1.0:
            suggestion = "avoid"

    else:
        # Flip mode: lean hard on spread & flip probability.
        # Heuristic weights:
        #   - 6x spread_score: a 10% discount to ARV => +0.6
        #   - 4x flip_p: 0.7 flip prob => +2.8
        #   - 0.5x dscr, 1.5x CoC as secondary stabilizers
        spread_component = 6.0 * spread_score
        flip_component = 4.0 * flip_p
        dscr_component = 0.5 * dscr
        coc_component = 1.5 * coc

        rank_score = spread_component + flip_component + dscr_component + coc_component

        suggestion = "watch"
        if spread_score > 0.10 and flip_p >= 0.6:
            suggestion = "buy"
        elif spread_score < -0.05 or flip_p < 0.2:
            suggestion = "avoid"

    # --- 3. Final labeling --------------------------------------------------
    score["rank_score"] = float(rank_score)
    score["suggestion"] = suggestion
    score["label"] = suggestion  # for now label == suggestion

    return score



def _normalize_payload_for_underwriting(
    payload: dict[str, Any],
) -> SimpleNamespace:
    """
    Wrap validate_and_prepare_payload so underwriting logic can work with attributes
    instead of raw dicts.
    """
    cleaned = validate_and_prepare_payload(payload)
    return SimpleNamespace(**cleaned)


def _sanitize_quantiles(
    q: dict[str, float] | None,
    name: str,
    fallback: float,
) -> dict[str, float]:
    """
    Ensure a quantile dict always has p10/p50/p90 and no obvious NaNs.
    If the bundle is missing or partially broken, fall back to a tight band
    around the fallback value.
    """
    if q is None:
        base = float(fallback)
        return {"p10": base * 0.95, "p50": base, "p90": base * 1.05}

    out: dict[str, float] = {}
    for k in ("p10", "p50", "p90"):
        v = q.get(k)
        try:
            v_f = float(v)
        except (TypeError, ValueError):
            v_f = float(fallback)
        if not pd.notna(v_f):
            v_f = float(fallback)
        out[k] = v_f

    # Ensure ordering p10 <= p50 <= p90
    p10 = out["p10"]
    p50 = max(out["p50"], p10)
    p90 = max(out["p90"], p50)
    out["p10"], out["p50"], out["p90"] = p10, p50, p90

    logger.info(
        "sanitize_quantiles",
        extra={"name": name, "quantiles": out},
    )
    return out


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
        * score: new risk-adjusted score (score_property) + suggestion + rank_score
        * score_legacy: old-style score_deal output
        * flip_p_good: optional flip classifier probability
        * arv_q: ARV quantiles from LightGBM bundle
        * guardrails: simple sanity flags
    """
    # 1. Validate + normalize + apply defaults
    payload = validate_and_prepare_payload(raw_payload)

    # Normalize strategy:
    # - UI may send "rental"/"flip"
    # - internal engine uses "hold"/"flip"
    strategy_raw = str(payload.get("strategy", "hold")).lower()
    if strategy_raw == "rental":
        strategy = "hold"
    elif strategy_raw in ("hold", "flip"):
        strategy = strategy_raw
    else:
        strategy = "hold"
    payload["strategy"] = strategy

    # 2. Cast dict units -> Unit models
    units_data = payload.get("units")
    if units_data:
        payload["units"] = [Unit(**u) for u in units_data]

    # 3. Build Property
    prop = Property(**payload)

    # 4. Ensure rents present (now uses payload features)
    prop = _fill_missing_rents(prop, rent_estimator, payload)

    # 4b. Derive gross rent and push into Property before finance
    gross_rent = 0.0
    if getattr(prop, "units", None):
        gross_rent = sum(float(u.market_rent or 0.0) for u in prop.units)
    else:
        gross_rent = float(getattr(prop, "est_market_rent", 0.0) or 0.0)

    if hasattr(prop, "monthly_rent"):
        prop.monthly_rent = gross_rent
    elif hasattr(prop, "gross_rent"):
        prop.gross_rent = gross_rent
    else:
        # Debug-only attribute if finance doesn't look at a named field
        prop._debug_gross_rent = gross_rent  # type: ignore[attr-defined]

    logger.info(
        "debug_gross_rent_before_finance",
        extra={"gross_rent": gross_rent},
    )

    # 5. Underwriting assumptions
    assumptions = UnderwritingAssumptions(
        vacancy_rate=config.VACANCY_RATE,
        maintenance_rate=config.MAINTENANCE_RATE,
        property_mgmt_rate=config.PROPERTY_MGMT_RATE,
        capex_rate=config.CAPEX_RATE,
        closing_cost_pct=config.DEFAULT_CLOSING_COST_PCT,
        min_dscr_good=config.MIN_DSCR_GOOD,
    )

    # 6. Financials ---------------------------------------------------------
    finance = analyze_property_financials(prop, assumptions)

    # 7. ARV quantiles from the bundle (or safe fallback) -------------------
    #
    # We *only* pass a simple "base" feature here.
    # - If an ARV bundle is configured and compatible, it will use it.
    # - If not, predict_arv_quantiles will fall back to a +/-10% band
    #   around this base value.
    list_price = float(payload.get("list_price") or 0.0)

    arv_q: dict[str, float] | None = None
    try:
        arv_q = predict_arv_quantiles({"base": list_price})
    except Exception as exc:
        logger.warning(
            "arv_quantile_inference_failed",
            extra={
                "error": str(exc),
                "list_price": list_price,
                "zipcode": payload.get("zipcode"),
            },
        )
        arv_q = None

    # 8. Pricing summary (ARV-aware when available) -------------------------
    sqft = float(payload.get("sqft") or 0.0)
    pricing = summarize_deal_pricing(
        prop=prop,
        sqft=sqft,
        assumptions=assumptions,
        arv_q=arv_q,
    )

    # 9. Flip probability (optional; safe no-op if classifier not ready) ----
    flip_p = _compute_flip_probability(finance, payload)

    # 10. Rank / score using ARV quantiles (no rent quantiles yet) ----------
    score_new = score_property(
        finance=finance,
        arv_q=arv_q,
        rent_q=None,
    )

    # Attach suggestion + ensure rank_score present (strategy-aware)
    score_with_suggestion = _attach_suggestion_and_rank(
        score_new,
        finance,
        pricing,
        strategy=strategy,
        flip_p_good=flip_p,
        arv_q=arv_q,
    )

    # 11. Legacy score for debugging/backwards compatibility ----------------
    score_legacy = score_deal(finance)


    result: dict[str, Any] = {
        "address": {
            "address": prop.address,
            "city": prop.city,
            "state": prop.state,
            "zipcode": prop.zipcode,
        },
        "property_type": prop.property_type,
        "strategy": strategy,
        "finance": finance,
        "pricing": pricing,
        "score": score_with_suggestion,
        "score_legacy": score_legacy,
        "flip_p_good": flip_p,
        "arv_q": arv_q,
    }

    # 12. Guardrails (sanity flags)
    result = apply_guardrails(payload=payload, result=result)

    logger.info("deal_analyzed", extra=result)

    # 13. Persist if repo provided
    deal_id: int | None = None
    if repo is not None:
        deal_id = repo.save_analysis(result, raw_payload)

    if deal_id is not None:
        result["deal_id"] = deal_id

    return result


def evaluate_deal(
    raw_payload: dict[str, Any],
    config_obj: Any,
    models: Any,
) -> DealEvaluation:
    """
    Canonical underwriting evaluation:
      - normalize payload
      - predict ARV & rent quantiles
      - build downside/base/upside scenarios
      - apply rules to assign label/risk_tier/confidence

    This returns a structured DealEvaluation object, which you can then
    serialize into the API / UI however you like.
    """

    # 1. Normalize inputs (attributes form)
    norm = _normalize_payload_for_underwriting(raw_payload)

    # 2. Predict ARV & rent quantiles using the provided models bundle
    #    models.arv and models.rent must expose predict_quantiles(norm) or similar.
    arv_q_raw = models.arv.predict_quantiles(norm)
    rent_q_raw = models.rent.predict_quantiles(norm)

    # Use list_price and a simple rent heuristic as fallbacks
    base_price = float(getattr(norm, "list_price", 0.0))
    base_rent_fallback = float(getattr(norm, "est_market_rent", 0.0) or 0.0)

    arv_q = _sanitize_quantiles(arv_q_raw, "ARV", fallback=base_price)
    rent_q = _sanitize_quantiles(
        rent_q_raw,
        "rent",
        fallback=base_rent_fallback if base_rent_fallback > 0 else 1500.0,
    )

    # 3. Build downside/base/upside scenarios using finance math
    downside = build_scenario_metrics(
        norm,
        arv=arv_q["p10"],
        rent=rent_q["p10"],
        config=config_obj.finance,
    )
    base = build_scenario_metrics(
        norm,
        arv=arv_q["p50"],
        rent=rent_q["p50"],
        config=config_obj.finance,
    )
    upside = build_scenario_metrics(
        norm,
        arv=arv_q["p90"],
        rent=rent_q["p90"],
        config=config_obj.finance,
    )

    # 4. Build DealEvaluation and apply rules
    eval_obj = DealEvaluation(
        address=getattr(norm, "address", ""),
        city=getattr(norm, "city", ""),
        state=getattr(norm, "state", ""),
        zipcode=str(getattr(norm, "zipcode", "")),
        list_price=float(getattr(norm, "list_price", 0.0)),
        strategy=getattr(norm, "strategy", "rental"),
        downside=downside,
        base=base,
        upside=upside,
        arv_quantiles=arv_q,
        rent_quantiles=rent_q,
        model_versions=getattr(models, "versions", {}),
        label="maybe",          # rules will override
        risk_tier="medium",
        confidence=0.0,
        warnings=[],
        hard_flags=[],
    )

    eval_obj = apply_rules(eval_obj, config_obj.rules)
    return eval_obj


def analyze_deal_with_defaults(
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Thin wrapper used by the HTTP API.

    Matches the older signature that http.py imports:
        analyze_deal_with_defaults(raw_payload=payload)

    Uses the default rent estimator and default SQL repo, and then delegates
    to analyze_deal.
    """
    rent_estimator = _default_estimator
    repo_to_use: DealRepository | None = _default_repo

    return analyze_deal(
        raw_payload=raw_payload,
        rent_estimator=rent_estimator,
        repo=repo_to_use,
    )

