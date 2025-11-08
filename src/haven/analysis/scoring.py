# src/haven/analysis/scoring.py
from __future__ import annotations

from typing import Dict, Optional


# ============================================================================
# score_deal: legacy/simple scoring used by deal_analyzer + /analyze
# ============================================================================

def score_deal(finance: dict) -> dict:
    """
    Classify a single deal based on basic safety & return metrics.

    Expected keys in `finance` (output of analyze_property_financials or equivalent):
      - cashflow_monthly_after_debt
      - cash_on_cash_return
      - dscr
      - breakeven_occupancy_pct

    Returns a dict:
      {
        "label": "buy" | "maybe" | "pass",
        "reason": str,
        "cash_on_cash_return": float,
        "dscr": float,
        "breakeven_occupancy_pct": float,
    }
    """

    cashflow = float(finance.get("cashflow_monthly_after_debt", 0.0))
    coc = float(finance.get("cash_on_cash_return", 0.0))
    dscr = float(finance.get("dscr", 0.0))
    breakeven = float(finance.get("breakeven_occupancy_pct", 1.0))

    # Hard safety gates
    if cashflow < 0:
        return {
            "label": "pass",
            "reason": "Negative monthly cashflow after debt.",
            "cash_on_cash_return": coc,
            "dscr": dscr,
            "breakeven_occupancy_pct": breakeven,
        }

    if dscr < 1.0:
        return {
            "label": "pass",
            "reason": "DSCR < 1.0 indicates debt coverage is too weak.",
            "cash_on_cash_return": coc,
            "dscr": dscr,
            "breakeven_occupancy_pct": breakeven,
        }

    # Positive but fragile deals
    if dscr < 1.15 or coc < 0.05:
        return {
            "label": "maybe",
            "reason": "Marginal returns; consider negotiation or better terms.",
            "cash_on_cash_return": coc,
            "dscr": dscr,
            "breakeven_occupancy_pct": breakeven,
        }

    # Solid long-term hold thresholds
    if dscr >= 1.25 and coc >= 0.08 and breakeven <= 0.85:
        return {
            "label": "buy",
            "reason": "Strong cashflow, healthy DSCR, and resilient breakeven.",
            "cash_on_cash_return": coc,
            "dscr": dscr,
            "breakeven_occupancy_pct": breakeven,
        }

    # Everything in-between
    return {
        "label": "maybe",
        "reason": "Acceptable metrics, but not outstanding; verify assumptions.",
        "cash_on_cash_return": coc,
        "dscr": dscr,
        "breakeven_occupancy_pct": breakeven,
    }


# ============================================================================
# score_property: risk-adjusted scalar for ranking candidates (/top-deals)
# ============================================================================

def score_property(
    finance: dict,
    arv_q: Optional[Dict[str, float]] = None,
    rent_q: Optional[Dict[str, float]] = None,
    dom: Optional[float] = None,
    strategy: str = "hold",
) -> dict:
    """
    Risk-aware ranking score for an investment candidate.

    Inputs:
      finance:
        Output from analyze_property_financials(...).
        Uses:
          - cashflow_monthly_after_debt
          - dscr
          - cash_on_cash_return
          - breakeven_occupancy_pct
      arv_q:
        ARV quantiles, e.g. {"q10":..., "q50":..., "q90":...} (optional)
      rent_q:
        Rent quantiles, e.g. {"q10":..., "q50":...} (optional)
      dom:
        Predicted or observed days-on-market (optional)
      strategy:
        "hold" or "flip"

    Returns:
      {
        "rank_score": float,  # scalar for sorting
        "label": "buy|maybe|pass",
        "reason": str,
        "dscr": float,
        "cash_on_cash_return": float,
        "downside_coc": float,
        "breakeven_occupancy_pct": float,
        "moe_arv": float | None,
      }
    """

    cashflow = float(finance.get("cashflow_monthly_after_debt", 0.0))
    dscr = float(finance.get("dscr", 0.0))
    coc = float(finance.get("cash_on_cash_return", 0.0))
    breakeven = float(finance.get("breakeven_occupancy_pct", 1.0))

    # --- Uncertainty from ARV quantiles (margin of error ratio) ---
    moe = None
    if (
        arv_q
        and "q10" in arv_q
        and "q90" in arv_q
        and "q50" in arv_q
        and arv_q["q50"] > 0
    ):
        moe = (arv_q["q90"] - arv_q["q10"]) / max(arv_q["q50"], 1.0)

    # --- Downside CoC from rent quantiles ---
    downside_coc = coc
    if (
        rent_q
        and "q10" in rent_q
        and "q50" in rent_q
        and rent_q["q50"] > 0
    ):
        downside_coc = coc * (rent_q["q10"] / rent_q["q50"])

    # --- Base score assembly ---
    score = 0.0

    # 1) Reward CoC up to 20% → max ~80 pts
    score += max(0.0, min(coc, 0.20)) * 400.0

    # 2) Reward DSCR between 1.0 and 2.0 → max 40 pts
    if dscr >= 1.0:
        score += min(dscr - 1.0, 1.0) * 40.0

    # 3) Penalize weak downside (bad in q10 scenario)
    if downside_coc < 0.0:
        score -= 40.0
    elif downside_coc < 0.06:
        score -= 10.0

    # 4) Penalize fragile breakeven occupancy
    if breakeven > 0.90:
        score -= 15.0
    elif breakeven > 0.80:
        score -= 5.0

    # 5) Penalize uncertainty from ARV spread
    if moe is not None:
        # moe in [0, 0.5] → up to -40 pts
        score -= max(0.0, min(moe, 0.5)) * 40.0

    # 6) Penalize long DOM (optional)
    if dom is not None:
        if dom > 90:
            score -= 15.0
        elif dom > 60:
            score -= 5.0

    # 7) Flips are harsher on downside
    if strategy == "flip" and downside_coc < 0.0:
        score -= 20.0

    # --- Turn score into label/reason (compatible with score_deal logic) ---

    label = "maybe"
    reason = "borderline; review details."

    if score < 0 or cashflow < 0 or dscr < 1.0:
        label = "pass"
        reason = "fails safety checks: negative cashflow and/or DSCR < 1.0."
    elif score >= 80 and dscr >= 1.25 and downside_coc >= 0.08:
        label = "buy"
        reason = "strong returns, resilient downside, lender-friendly coverage."
    elif score >= 50:
        label = "maybe (negotiate)"
        reason = "workable, but only at better price/terms."

    return {
        "rank_score": round(score, 2),
        "label": label,
        "reason": reason,
        "dscr": dscr,
        "cash_on_cash_return": coc,
        "downside_coc": round(downside_coc, 4),
        "breakeven_occupancy_pct": breakeven,
        "moe_arv": moe,
    }
