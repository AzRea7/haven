from typing import Dict
from haven.domain.property import Property
from haven.domain.assumptions import UnderwritingAssumptions


def _monthly_mortgage_payment(principal: float, annual_rate: float, years: int) -> float:
    """
    Standard fixed-rate amortization formula:
    M = P * [ r(1+r)^n / ((1+r)^n - 1) ]
    P = loan principal
    r = monthly interest rate
    n = number of payments (months)
    """
    r = annual_rate / 12.0
    n = years * 12

    if r == 0:
        return principal / n

    numerator = r * (1 + r) ** n
    denom = (1 + r) ** n - 1
    return principal * (numerator / denom)


def _aggregate_rent(property: Property) -> float:
    """
    Determine total gross scheduled rent per month.
    - If units[] exists, sum unit.market_rent.
    - Else fall back to est_market_rent (single door).
    """
    if property.units:
        return sum((u.market_rent or 0.0) for u in property.units)
    return property.est_market_rent or 0.0


def _effective_rent(gross_rent_monthly: float, assumptions: UnderwritingAssumptions) -> Dict[str, float]:
    """
    Apply vacancy to gross rent using assumptions.vacancy_rate.
    """
    vacancy_loss = gross_rent_monthly * assumptions.vacancy_rate
    effective = gross_rent_monthly - vacancy_loss
    return {
        "gross_rent_monthly": gross_rent_monthly,
        "vacancy_loss_monthly": vacancy_loss,
        "effective_rent_monthly": effective,
    }


def _operating_expenses_monthly(
    property: Property,
    effective_rent_monthly: float,
    assumptions: UnderwritingAssumptions
) -> Dict[str, float]:
    """
    Operating expenses do NOT include mortgage. (Mortgage is financing, not operations.)
    Industry-standard buckets:
    - Maintenance
    - Property management
    - CapEx reserves (roofs, HVAC, parking lot, etc.)
    - Taxes
    - Insurance
    - HOA (if any)
    """
    maint = effective_rent_monthly * assumptions.maintenance_rate
    mgmt  = effective_rent_monthly * assumptions.property_mgmt_rate
    capex = effective_rent_monthly * assumptions.capex_rate

    taxes_monthly = property.taxes_annual / 12.0
    ins_monthly   = property.insurance_annual / 12.0
    hoa_monthly   = property.hoa_monthly

    total_op = maint + mgmt + capex + taxes_monthly + ins_monthly + hoa_monthly

    return {
        "maintenance_monthly": maint,
        "mgmt_monthly": mgmt,
        "capex_monthly": capex,
        "taxes_monthly": taxes_monthly,
        "insurance_monthly": ins_monthly,
        "hoa_monthly": hoa_monthly,
        "total_operating_monthly": total_op,
    }


def analyze_property_financials(
    property: Property,
    assumptions: UnderwritingAssumptions
) -> Dict[str, float]:
    """
    Core underwriting brain.
    Returns metrics that investors and lenders actually care about.
    """

    # --- financing basics ---
    purchase_price = property.list_price
    down_payment = purchase_price * property.down_payment_pct
    loan_amount = purchase_price - down_payment

    mortgage_monthly = _monthly_mortgage_payment(
        principal=loan_amount,
        annual_rate=property.interest_rate_annual,
        years=property.loan_term_years,
    )

    # --- income side ---
    gross_rent_monthly = _aggregate_rent(property)
    rent_info = _effective_rent(gross_rent_monthly, assumptions)

    # --- operating expenses ---
    opx = _operating_expenses_monthly(
        property,
        rent_info["effective_rent_monthly"],
        assumptions,
    )

    # --- NOI (Net Operating Income) ---
    # NOI is income after vacancy + operating expenses, BEFORE debt.
    noi_monthly = rent_info["effective_rent_monthly"] - opx["total_operating_monthly"]
    noi_annual = noi_monthly * 12.0

    # --- Debt Service ---
    annual_debt_service = mortgage_monthly * 12.0

    # --- DSCR (Debt Service Coverage Ratio) ---
    # Lenders love DSCR >= ~1.20 for small multifamily/commercial.
    dscr = 0.0
    if annual_debt_service > 0:
        dscr = noi_annual / annual_debt_service

    # --- Cap Rate ---
    # Cap rate = NOI / Purchase Price. Used to value income-producing property.
    cap_rate = 0.0
    if purchase_price > 0:
        cap_rate = noi_annual / purchase_price

    # --- Cash Flow After Debt ---
    cashflow_monthly_after_debt = (
        rent_info["effective_rent_monthly"]
        - opx["total_operating_monthly"]
        - mortgage_monthly
    )

    # --- Cash on Cash Return ---
    # Annualized cash flow / initial cash invested (down payment + closing costs).
    est_closing_costs = purchase_price * assumptions.closing_cost_pct
    total_cash_in = down_payment + est_closing_costs

    cash_on_cash = 0.0
    if total_cash_in > 0:
        cash_on_cash = (cashflow_monthly_after_debt * 12.0) / total_cash_in

    # --- Breakeven Occupancy ---
    # Breakeven occupancy = % of gross rent you must collect so you don't lose money
    breakeven_occ = 0.0
    if gross_rent_monthly > 0:
        breakeven_occ = (
            (opx["total_operating_monthly"] + mortgage_monthly)
            / gross_rent_monthly
        )

    result = {
        "purchase_price": purchase_price,
        "down_payment": down_payment,
        "loan_amount": loan_amount,

        "mortgage_monthly": mortgage_monthly,

        "gross_rent_monthly": rent_info["gross_rent_monthly"],
        "effective_rent_monthly": rent_info["effective_rent_monthly"],
        "vacancy_loss_monthly": rent_info["vacancy_loss_monthly"],

        "operating_expenses_monthly": opx["total_operating_monthly"],
        "noi_monthly": noi_monthly,
        "noi_annual": noi_annual,

        "cap_rate": cap_rate,
        "dscr": dscr,
        "cashflow_monthly_after_debt": cashflow_monthly_after_debt,
        "cash_on_cash_return": cash_on_cash,
        "breakeven_occupancy_pct": breakeven_occ,

        "meets_lender_dscr_threshold": dscr >= assumptions.min_dscr_good,
    }

    return result
