# src/haven/analysis/valuation.py
from __future__ import annotations

from typing import Any

from haven.domain.assumptions import UnderwritingAssumptions
from haven.domain.property import Property
from haven.adapters.config import config


def _estimate_value_income_approach(noi_annual: float, market_cap_rate: float) -> float:
    if market_cap_rate <= 0:
        return 0.0
    return noi_annual / market_cap_rate


def _estimate_value_residential_price_per_sqft(
    sqft: float,
    price_per_sqft: float,
) -> float:
    if sqft is None or sqft <= 0:
        return 0.0
    return sqft * price_per_sqft


def _get_arv_p(
    arv_q: dict[str, float] | None,
) -> tuple[float | None, float | None, float | None]:
    """
    Helper to read ARV quantiles in a tolerant way.

    We support both:
      - { "p10", "p50", "p90" }
      - { "q10", "q50", "q90" }
    and fall back to None if missing.
    """
    if arv_q is None:
        return None, None, None

    def pick(*names: str) -> float | None:
        for n in names:
            if n in arv_q and arv_q[n] is not None:
                try:
                    return float(arv_q[n])
                except (TypeError, ValueError):
                    continue
        return None

    p10 = pick("p10", "q10")
    p50 = pick("p50", "q50", "median")
    p90 = pick("p90", "q90")
    return p10, p50, p90


def summarize_deal_pricing(
    prop: Property,
    sqft: float,
    assumptions: UnderwritingAssumptions,
    arv_q: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Basic pricing summary for a single deal.

    For rentals, this can fall back to a simple price-per-sqft heuristic.
    For flips, we really care about ARV. If arv_q is provided and has a
    reasonable p50/q50, we treat that as the fair-value anchor and compute
    spread vs ARV.

    Returned fields:
        ask_price:             List price
        fair_value_estimate:   Our best current estimate of fair value
        price_delta:           ask - fair_value
        price_delta_pct:       (ask - fair_value) / fair_value
        arv_*:                 Optional ARV quantiles mirrored into pricing
    """
    list_price = float(prop.list_price or 0.0)

    # --- 1. Choose a base fair value ---------------------------------------
    # Priority:
    #   1) ARV median (p50 / q50), if available
    #   2) comparables_fair_value from the property
    #   3) sqft * assumed_price_per_sqft
    #   4) list_price as last resort
    fair_value_estimate: float | None = None

    # 1) ARV median (supports both p*/q*)
    if arv_q is not None:
        _, p50, _ = _get_arv_p(arv_q)
        if p50 is not None and p50 > 0:
            fair_value_estimate = float(p50)

    # 2) Explicit comparables fair value if no ARV
    if fair_value_estimate is None and getattr(prop, "comparables_fair_value", None):
        try:
            fair_value_estimate = float(prop.comparables_fair_value)
        except (TypeError, ValueError):
            fair_value_estimate = None

    # 3) Simple price-per-sqft heuristic
    if fair_value_estimate is None:
        assumed_ppsf = float(config.valuation.assumed_price_per_sqft)
        if sqft and sqft > 0:
            fair_value_estimate = float(
                _estimate_value_residential_price_per_sqft(sqft, assumed_ppsf)
            )

    # 4) Last resort: treat list price as "fair"
    if fair_value_estimate is None or fair_value_estimate <= 0:
        fair_value_estimate = list_price

    # --- 2. Spread vs fair value -------------------------------------------
    price_delta = list_price - fair_value_estimate
    price_delta_pct = price_delta / fair_value_estimate if fair_value_estimate else 0.0

    result: dict[str, Any] = {
        "ask_price": float(list_price),
        "fair_value_estimate": float(fair_value_estimate),
        "price_delta": float(price_delta),
        "price_delta_pct": float(price_delta_pct),
    }

    # --- 3. Mirror ARV quantiles into pricing for convenience --------------
    if arv_q is not None:
        p10, p50, p90 = _get_arv_p(arv_q)

        if p10 is not None:
            result["arv_p10"] = float(p10)
        if p50 is not None:
            result["arv_p50"] = float(p50)
        if p90 is not None:
            result["arv_p90"] = float(p90)

    return result
