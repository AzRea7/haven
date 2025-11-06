

def score_deal(finance: dict[str, float]) -> dict[str, str | float | bool]:
    """
    Convert finance metrics into a recommendation.
    """
    cashflow = finance["cashflow_monthly_after_debt"]
    dscr = finance["dscr"]
    coc = finance["cash_on_cash_return"]
    breakeven = finance["breakeven_occupancy_pct"]

    suggestion = "maybe negotiate"
    risk_level = "medium"
    lender_friendly = dscr >= 1.20

    if cashflow < 0:
        suggestion = "pass"
        risk_level = "high"
        lender_friendly = False
    elif dscr < 1.20:
        suggestion = "maybe (low DSCR)"
        risk_level = "medium"
        lender_friendly = False
    elif coc >= 0.12 and breakeven <= 0.90:
        suggestion = "buy"
        risk_level = "low"
        lender_friendly = True

    return {
        "suggestion": suggestion,                # "buy" | "maybe negotiate" | "maybe (low DSCR)" | "pass"
        "risk_level": risk_level,                # "low" | "medium" | "high"
        "lender_friendly": lender_friendly,      # True/False
        "dscr": dscr,
        "cash_on_cash_return": coc,
        "breakeven_occupancy_pct": breakeven,
        "monthly_cashflow_after_debt": cashflow,
    }
