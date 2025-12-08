from haven.domain.underwriting import DealEvaluation

def apply_rules(eval: DealEvaluation, config) -> DealEvaluation:
    warnings = []
    hard_flags = []

    base = eval.base
    down = eval.downside

    # 1. Basic sanity
    if eval.list_price <= 0:
        hard_flags.append("Non-positive list price")
    if eval.base.arv < eval.list_price * 0.8:
        warnings.append("ARV is unusually low relative to list price (possible data issue).")
    if eval.base.arv > eval.list_price * 2.0:
        warnings.append("ARV is > 2x list price; check if this is realistic or a model artifact.")

    # 2. Rental safety (downside first)
    if down.dscr < 1.0:
        hard_flags.append(f"Downside DSCR < 1.0 ({down.dscr:.2f})")
    if down.monthly_cashflow < 0:
        warnings.append(f"Negative downside cashflow: ${down.monthly_cashflow:,.0f}/mo")

    # 3. Base performance thresholds
    meets_buy_base = (
        base.dscr >= config.min_dscr_buy and
        base.coc  >= config.min_coc_buy  and
        down.dscr >= config.min_dscr_downside and
        down.coc  >= config.min_coc_downside
    )

    # 4. Confidence from uncertainty
    arv_spread_rel = (eval.arv_quantiles["p90"] - eval.arv_quantiles["p10"]) / max(eval.arv_quantiles["p50"], 1.0)
    rent_spread_rel = (eval.rent_quantiles["p90"] - eval.rent_quantiles["p10"]) / max(eval.rent_quantiles["p50"], 1.0)

    uncertainty_penalty = max(arv_spread_rel, rent_spread_rel)
    confidence = max(0.0, 1.0 - uncertainty_penalty * config.uncertainty_weight)

    # 5. Final label
    if hard_flags:
        label = "pass"
        risk_tier = "high"
    elif meets_buy_base and confidence >= config.min_confidence_for_buy:
        label = "buy"
        risk_tier = "low"
    elif base.dscr >= config.min_dscr_maybe and base.coc >= config.min_coc_maybe:
        label = "maybe"
        risk_tier = "medium"
    else:
        label = "pass"
        risk_tier = "medium"

    eval.label = label
    eval.risk_tier = risk_tier
    eval.confidence = confidence
    eval.warnings = warnings
    eval.hard_flags = hard_flags
    return eval
