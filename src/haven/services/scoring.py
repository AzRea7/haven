from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# ARV quantile helpers (existing behavior)
# ---------------------------------------------------------------------------


def load_arv_bundle(model_dir: str = "models") -> dict[str, Any]:
    """
    Load pre-trained ARV quantile models (q10, q50, q90) from joblib artifacts.
    """
    q10 = joblib.load(f"{model_dir}/arv_q10.joblib")
    q50 = joblib.load(f"{model_dir}/arv_q50.joblib")
    q90 = joblib.load(f"{model_dir}/arv_q90.joblib")
    return {"q10": q10, "q50": q50, "q90": q90}


def score_arv(models: Mapping[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """
    Run the ARV models on a feature matrix X and return a DataFrame
    with columns 'q10', 'q50', 'q90'.
    """
    preds: dict[str, Any] = {}
    for k, m in models.items():
        preds[k] = m.predict(X)
    return pd.DataFrame(preds, index=X.index)


def compute_profit_and_mao(cands: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """
    Flip analysis helper.

    Requires `cands` with:
        rehab, hold_costs, closing_costs, selling_cost_rate,
        buy_cost_rate, desired_profit, offer_price (optional).

    MAO (Max Allowable Offer) is computed per quantile as:

      MAO = ARV * (1 - selling_cost_rate)
            - rehab
            - hold_costs
            - desired_profit
            - ARV * buy_cost_rate
    """
    df = cands.join(preds)

    for k in ["q10", "q50", "q90"]:
        arv = df[k]
        selling_costs = arv * df["selling_cost_rate"]
        buy_costs = arv * df["buy_cost_rate"]
        net = arv - selling_costs - df["rehab"] - df["hold_costs"] - buy_costs

        df[f"profit_{k}"] = net - df.get("offer_price", 0.0)
        df[f"mao_{k}"] = (
            arv * (1 - df["selling_cost_rate"])
            - df["rehab"]
            - df["hold_costs"]
            - df["desired_profit"]
            - arv * df["buy_cost_rate"]
        )

    return df


# ---------------------------------------------------------------------------
# HOLD-deal scoring used by both /analyze and /top-deals
# ---------------------------------------------------------------------------


def _extract_finance_metric(finance: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(finance.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def score_deal(finance: Mapping[str, Any]) -> dict[str, Any]:
    """
    Legacy/low-level scoring for a single hold deal.

    Takes a `finance` dict from analyze_property_financials and returns:

      {
        "label": "buy" | "maybe" | "pass",
        "reason": "...",
        "dscr": ...,
        "cash_on_cash_return": ...,
        "breakeven_occupancy_pct": ...,
        "rank_score": float
      }

    Heuristic is intentionally simple but monotonic:
      - Higher DSCR & CoC -> higher rank_score
      - Lower breakeven occupancy -> higher rank_score
    """
    dscr = _extract_finance_metric(finance, "dscr", 0.0)
    coc = _extract_finance_metric(finance, "cash_on_cash_return", 0.0)
    breakeven = _extract_finance_metric(finance, "breakeven_occupancy_pct", 1.0)
    cashflow = _extract_finance_metric(finance, "cashflow_monthly_after_debt", 0.0)

    label = "pass"
    reason = "Negative cashflow or weak coverage relative to risk."

    # Solid "buy": strong coverage, strong CoC, positive cashflow, decent breakeven
    if dscr >= 1.35 and coc >= 0.08 and breakeven <= 0.90 and cashflow > 0:
        label = "buy"
        reason = "Strong cashflow, healthy DSCR, and resilient breakeven."
    # Workable, but needs negotiation or more diligence
    elif dscr >= 1.1 and coc >= 0.04 and cashflow > 0:
        label = "maybe"
        reason = "Workable deal but requires better terms or deeper underwriting."

    # Rank score: monotonic in dscr/coc, penalized by breakeven.
    # (Weights chosen just to separate good vs bad deals clearly.)
    rank_score = (
        dscr * 10.0
        + coc * 50.0
        - breakeven * 10.0
    )

    return {
        "label": label,
        "reason": reason,
        "dscr": dscr,
        "cash_on_cash_return": coc,
        "breakeven_occupancy_pct": breakeven,
        "rank_score": rank_score,
    }


def score_property(
    finance: Mapping[str, Any],
    arv_q: Mapping[str, Any] | None = None,
    rent_q: Mapping[str, Any] | None = None,
    dom: float | int | None = None,
    strategy: str = "hold",
    flip_p_good: float | None = None,
) -> dict[str, Any]:
    """
    Higher-level scoring used by:
      - services.deal_analyzer.analyze_deal
      - api.http.top_deals

    Starts from `score_deal(finance)` and then nudges `rank_score`
    using uncertainty (ARV/rent quantiles), days-on-market, and optional
    flip probability.
    """
    base = score_deal(finance)
    rank = float(base.get("rank_score", 0.0))

    # Days on market penalty: more DOM = slightly lower rank.
    if dom is not None:
        try:
            dom_f = float(dom)
            # Up to ~6 points penalty over 365 days.
            rank -= min(max(dom_f, 0.0) / 60.0, 6.0)
        except (TypeError, ValueError):
            pass

    # Uncertainty penalties from ARV quantiles
    if arv_q is not None:
        try:
            q10 = float(arv_q.get("q10", 0.0) or 0.0)
            q50 = float(arv_q.get("q50", 0.0) or 0.0)
            q90 = float(arv_q.get("q90", 0.0) or 0.0)
            spread = max(q90 - q10, 0.0)
            if q50 > 0.0 and spread > 0.0:
                rel_spread = spread / q50
                # Up to 5 points penalty for very uncertain ARV
                rank -= min(rel_spread * 5.0, 5.0)
        except (TypeError, ValueError):
            pass

    # Uncertainty penalties from rent quantiles
    if rent_q is not None:
        try:
            rq10 = float(rent_q.get("q10", 0.0) or 0.0)
            rq50 = float(rent_q.get("q50", 0.0) or 0.0)
            rq90 = float(rent_q.get("q90", 0.0) or 0.0)
            rspread = max(rq90 - rq10, 0.0)
            if rq50 > 0.0 and rspread > 0.0:
                rel_rspread = rspread / rq50
                # Up to 3 points penalty for very uncertain rent
                rank -= min(rel_rspread * 3.0, 3.0)
        except (TypeError, ValueError):
            pass

    # Optional flip probability bump (if model is present)
    if flip_p_good is not None:
        try:
            p = float(flip_p_good)
            # centered at 0.5, small bump (±5 points across full range)
            rank += (p - 0.5) * 10.0
        except (TypeError, ValueError):
            pass

    base["rank_score"] = rank
    base["strategy"] = strategy

    return base
