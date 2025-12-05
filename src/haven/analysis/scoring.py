# src/haven/analysis/scoring.py
from __future__ import annotations

from typing import Dict, Mapping, Optional


# =====================================================================
# Legacy/simple scoring used by some tests or callers
# =====================================================================


def score_deal(finance: dict) -> dict:
    """
    Classify a deal using basic safety & return metrics.

    Expected keys in `finance`:
      - cashflow_monthly_after_debt
      - cash_on_cash_return
      - dscr
      - breakeven_occupancy_pct
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

    # Positive but thin
    if coc < 0.05 or dscr < 1.15:
        return {
            "label": "maybe",
            "reason": "Marginal safety/returns. Needs deeper underwriting.",
            "cash_on_cash_return": coc,
            "dscr": dscr,
            "breakeven_occupancy_pct": breakeven,
        }

    return {
        "label": "buy",
        "reason": "Strong cashflow and coverage relative to risk.",
        "cash_on_cash_return": coc,
        "dscr": dscr,
        "breakeven_occupancy_pct": breakeven,
    }


# =====================================================================
# Risk-adjusted rank scoring for /top-deals
# =====================================================================


def _coalesce_quantile(q: Mapping[str, float] | None, key: str, default: float = 0.0) -> float:
    if not q:
        return default
    v = q.get(key)
    return float(v) if v is not None else default


def _label_from_score(
    score: float,
    dscr: float,
    coc: float,
    cashflow: float,
    tiny_unit_flag: bool,
) -> tuple[str, str]:
    # Hard fails first
    if cashflow < 0:
        return "pass", "Negative cashflow in base case."
    if dscr < 1.0:
        return "pass", "DSCR < 1.0; cannot safely service debt."
    if tiny_unit_flag:
        return "pass", "Unit is extremely small; likely illiquid and operationally fragile."

    # Interpret by score bands
    if score >= 40:
        return "buy", "High risk-adjusted score with strong coverage and returns."
    if score >= 15:
        return "buy", "Attractive profile; meets target safety and return thresholds."
    if score >= 0:
        return "maybe", "Workable but requires deeper underwriting or better terms."
    return "pass", "Risk/return profile is not compelling versus alternatives."


def score_property(
    finance: Mapping[str, float],
    arv_q: Optional[Mapping[str, float]] = None,
    rent_q: Optional[Mapping[str, float]] = None,
    dom: float | None = None,
    strategy: str = "hold",
    flip_p_good: float | None = None,
    sqft: float | None = None,
    year_built: float | None = None,
) -> Dict[str, object]:
    """
    Main scoring function used by /top-deals.

    Inputs:
      - finance: output from analyze_property_financials
      - arv_q: dict with ARV quantiles (q10/q50/q90) if available
      - rent_q: dict with rent quantiles (q10/q50/q90) if available
      - dom: days on market
      - strategy: "hold" or "flip"
      - flip_p_good: optional probability from a flip classifier (0-1)
      - sqft: building square footage (optional, but recommended)
      - year_built: year the property was built (optional, but recommended)

    Output:
      {
        "rank_score": float in [-100, 100],
        "label": "buy" | "maybe" | "pass",
        "reason": str,
      }
    """
    cashflow = float(finance.get("cashflow_monthly_after_debt", 0.0))
    coc = float(finance.get("cash_on_cash_return", 0.0))
    dscr = float(finance.get("dscr", 0.0))
    breakeven = float(finance.get("breakeven_occupancy_pct", 1.0))

    dom = float(dom or finance.get("days_on_market", 0.0) or 0.0)

    size = float(sqft or finance.get("sqft", 0.0) or 0.0)
    year = float(year_built or finance.get("year_built", 0.0) or 0.0)

    # Downside signals from quantiles (if present)
    rent_q10 = _coalesce_quantile(rent_q, "q10", default=0.0)
    arv_q10 = _coalesce_quantile(arv_q, "q10", default=0.0)
    arv_q50 = _coalesce_quantile(arv_q, "q50", default=0.0)

    # ---------------- Base components ----------------

    # CoC: treat each percentage point as one score unit up to 40%, then clamp.
    coc_pct = coc * 100.0
    coc_component = max(min(coc_pct, 40.0), -40.0)

    # DSCR: reward strength above 1.0, with diminishing returns after ~2.0
    if dscr <= 0:
        dscr_component = -40.0
    elif dscr < 1.0:
        dscr_component = -30.0
    else:
        dscr_component = (dscr - 1.0) * 25.0  # DSCR 1.4 → +10
        dscr_component = max(min(dscr_component, 25.0), -30.0)

    # Breakeven occupancy: punish fragile deals
    if breakeven <= 0:
        breakeven_component = -10.0
    else:
        breakeven_component = -max((breakeven - 0.90) * 200.0, 0.0)
        breakeven_component = max(breakeven_component, -20.0)

    # DOM: stale listings might hide issues
    dom_component = 0.0
    if dom > 45:
        dom_component = -(min(dom - 45.0, 180.0) * 0.10)  # up to about -13.5

    # For flips, we care more about liquidity – increase DOM penalty
    if strategy == "flip" and dom_component < 0.0:
        dom_component *= 1.5

    # Tiny square footage penalties
    size_component = 0.0
    tiny_unit_flag = False
    if size > 0:
        if size < 450:
            size_component -= 40.0
            tiny_unit_flag = True
        elif size < 600:
            size_component -= 25.0

    # Old homes penalty – more likely to have capex and surprise rehab
    age_component = 0.0
    if year > 0 and year < 1960:
        age_component = -15.0

    # ---------------- Downside risk adjustments ----------------

    downside_component = 0.0

    # If ARV downside (q10) is far below median, increase caution.
    if arv_q10 > 0 and arv_q50 > 0:
        downside_ratio = arv_q10 / max(arv_q50, 1e-9)
        if downside_ratio < 0.9:
            downside_component -= (0.9 - downside_ratio) * 40.0  # up to about -40

    # If rent downside is very weak, also penalize
    if rent_q10 > 0:
        # Crude: if q10 rent would not cover op ex + debt → big penalty.
        # We don't recompute full mortgage here; this is a soft heuristic.
        if cashflow < 0 and coc < 0.03:
            downside_component -= 15.0

    # For flips, we care more about downside on ARV & rent
    if strategy == "flip" and downside_component < 0.0:
        downside_component *= 1.3

    # ---------------- Flip classifier overlay (optional) ----------------

    flip_component = 0.0
    if flip_p_good is not None:
        # Center at 0.5 -> neutral; more confident good/bad moves the score.
        flip_component = (float(flip_p_good) - 0.5) * 40.0
        # Only strongly applied if strategy hints "flip"
        if strategy != "flip":
            flip_component *= 0.4  # dampen for hold scenarios

    # ---------------- Aggregate & clamp ----------------

    rank_score = (
        coc_component
        + dscr_component
        + breakeven_component
        + dom_component
        + size_component
        + age_component
        + downside_component
        + flip_component
    )

    # Hard overrides from cashflow / DSCR / tiny units
    if cashflow < 0:
        rank_score = min(rank_score, -25.0)
    if dscr < 1.0:
        rank_score = min(rank_score, -25.0)
    if tiny_unit_flag:
        rank_score = min(rank_score, -25.0)

    # Clamp for stability
    rank_score = max(min(rank_score, 100.0), -100.0)

    label, reason = _label_from_score(
        score=rank_score,
        dscr=dscr,
        coc=coc,
        cashflow=cashflow,
        tiny_unit_flag=tiny_unit_flag,
    )

    return {
        "rank_score": float(rank_score),
        "label": label,
        "reason": reason,
    }
