# tests/fixtures/deals.py

from haven.domain.underwriting import DealEvaluation, ScenarioMetrics


def _base_eval_template() -> dict:
    return dict(
        address="123 Test St",
        city="Testville",
        state="MI",
        zipcode="48009",
        list_price=0.0,
        strategy="rental",
        arv_quantiles={"p10": 0.0, "p50": 0.0, "p90": 0.0},
        rent_quantiles={"p10": 0.0, "p50": 0.0, "p90": 0.0},
        model_versions={},
        label="maybe",
        risk_tier="medium",
        confidence=0.0,
        warnings=[],
        hard_flags=[],
    )


def cheap_cashflow_beast() -> DealEvaluation:
    """
    150k house, ~2200 rent, strong DSCR & CoC, low uncertainty.
    Expected: label 'buy', low risk, no hard_flags (after rules).
    """
    downside = ScenarioMetrics(
        arv=180_000.0,
        rent=2000.0,
        noi=16_000.0,
        dscr=1.20,
        coc=0.12,
        cap_rate=0.09,
        breakeven_occ=0.70,
        monthly_cashflow=300.0,
    )
    base = ScenarioMetrics(
        arv=190_000.0,
        rent=2200.0,
        noi=18_000.0,
        dscr=1.35,
        coc=0.16,
        cap_rate=0.10,
        breakeven_occ=0.65,
        monthly_cashflow=500.0,
    )
    upside = ScenarioMetrics(
        arv=200_000.0,
        rent=2400.0,
        noi=20_000.0,
        dscr=1.50,
        coc=0.18,
        cap_rate=0.11,
        breakeven_occ=0.60,
        monthly_cashflow=650.0,
    )

    base_fields = _base_eval_template()
    base_fields.update(
        dict(
            list_price=150_000.0,
            arv_quantiles={"p10": 180_000.0, "p50": 190_000.0, "p90": 200_000.0},
            rent_quantiles={"p10": 2000.0, "p50": 2200.0, "p90": 2400.0},
        )
    )

    return DealEvaluation(
        downside=downside,
        base=base,
        upside=upside,
        **base_fields,
    )


def bad_negative_cashflow() -> DealEvaluation:
    """
    400k house, 1800 rent, negative downside cashflow and DSCR < 1.0.
    Expected: 'pass' with a hard_flag about downside DSCR.
    """
    downside = ScenarioMetrics(
        arv=380_000.0,
        rent=1700.0,
        noi=5_000.0,
        dscr=0.85,
        coc=0.01,
        cap_rate=0.01,
        breakeven_occ=0.95,
        monthly_cashflow=-300.0,
    )
    base = ScenarioMetrics(
        arv=390_000.0,
        rent=1800.0,
        noi=7_000.0,
        dscr=0.95,
        coc=0.03,
        cap_rate=0.02,
        breakeven_occ=0.90,
        monthly_cashflow=-150.0,
    )
    upside = ScenarioMetrics(
        arv=400_000.0,
        rent=1900.0,
        noi=9_000.0,
        dscr=1.05,
        coc=0.04,
        cap_rate=0.02,
        breakeven_occ=0.85,
        monthly_cashflow=-50.0,
    )

    base_fields = _base_eval_template()
    base_fields.update(
        dict(
            list_price=400_000.0,
            arv_quantiles={"p10": 380_000.0, "p50": 390_000.0, "p90": 400_000.0},
            rent_quantiles={"p10": 1700.0, "p50": 1800.0, "p90": 1900.0},
        )
    )

    return DealEvaluation(
        downside=downside,
        base=base,
        upside=upside,
        **base_fields,
    )


def borderline_maybe() -> DealEvaluation:
    """
    Middle-of-the-road deal:
    DSCR ~1.15, CoC ~7% at base; downside is okay-ish but not stellar.
    Expected: 'maybe', medium risk.
    """
    downside = ScenarioMetrics(
        arv=260_000.0,
        rent=1800.0,
        noi=11_000.0,
        dscr=1.05,
        coc=0.05,
        cap_rate=0.05,
        breakeven_occ=0.85,
        monthly_cashflow=50.0,
    )
    base = ScenarioMetrics(
        arv=270_000.0,
        rent=1900.0,
        noi=12_000.0,
        dscr=1.15,
        coc=0.07,
        cap_rate=0.06,
        breakeven_occ=0.80,
        monthly_cashflow=150.0,
    )
    upside = ScenarioMetrics(
        arv=280_000.0,
        rent=2000.0,
        noi=13_000.0,
        dscr=1.25,
        coc=0.08,
        cap_rate=0.07,
        breakeven_occ=0.75,
        monthly_cashflow=250.0,
    )

    base_fields = _base_eval_template()
    base_fields.update(
        dict(
            list_price=260_000.0,
            arv_quantiles={"p10": 260_000.0, "p50": 270_000.0, "p90": 280_000.0},
            rent_quantiles={"p10": 1800.0, "p50": 1900.0, "p90": 2000.0},
        )
    )

    return DealEvaluation(
        downside=downside,
        base=base,
        upside=upside,
        **base_fields,
    )
