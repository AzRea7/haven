import math
from haven.domain.underwriting import ScenarioMetrics
from dataclasses import dataclass

@dataclass
class FinanceConfig:
    interest_rate: float      # annual, e.g. 0.075
    amort_years: int          # 30
    ltv: float                # loan-to-value, e.g. 0.80
    closing_cost_rate: float  # % of purchase price
    rehab_contingency: float  # bump rehab estimates, e.g. +10%
    taxes_rate: float         # heuristics if not given
    insurance_rate: float     # heuristics if not given
    vacancy_rate: float       # e.g. 0.05
    maintenance_rate: float   # as % of rent or value
    property_management_rate: float

def annuity_payment(rate_monthly: float, n_months: int, principal: float) -> float:
    r = rate_monthly
    if r == 0:
        return principal / n_months
    return principal * (r * (1 + r) ** n_months) / ((1 + r) ** n_months - 1)

def build_scenario_metrics(norm, arv: float, rent: float, config: FinanceConfig) -> ScenarioMetrics:
    purchase_price = norm.list_price
    loan_amount = purchase_price * config.ltv
    down_payment = purchase_price - loan_amount

    # Costs
    closing_costs = purchase_price * config.closing_cost_rate
    total_equity_in = down_payment + closing_costs + norm.rehab_budget * (1 + config.rehab_contingency)

    # Income / expenses
    gross_rent_annual = rent * 12
    vacancy_loss = gross_rent_annual * config.vacancy_rate
    taxes = purchase_price * config.taxes_rate
    insurance = purchase_price * config.insurance_rate
    maintenance = gross_rent_annual * config.maintenance_rate
    mgmt_fees = gross_rent_annual * config.property_management_rate

    # Debt service
    m_rate = config.interest_rate / 12
    n_months = config.amort_years * 12
    monthly_pi = annuity_payment(m_rate, n_months, loan_amount)
    annual_debt = monthly_pi * 12

    # NOI
    operating_expenses = taxes + insurance + maintenance + mgmt_fees + vacancy_loss
    noi = gross_rent_annual - operating_expenses

    dscr = noi / annual_debt if annual_debt > 0 else float("inf")

    cap_rate = noi / purchase_price if purchase_price > 0 else 0.0
    coc = noi / total_equity_in if total_equity_in > 0 else 0.0

    monthly_cashflow = (noi - annual_debt) / 12

    # Breakeven occupancy (very useful for risk)
    # Solve for occupancy where NOI = debt service
    if gross_rent_annual > 0:
        fixed_expenses = taxes + insurance + maintenance + mgmt_fees
        required_income = annual_debt + fixed_expenses
        breakeven_occ = required_income / gross_rent_annual
    else:
        breakeven_occ = 1.0

    return ScenarioMetrics(
        arv=arv,
        rent=rent,
        noi=noi,
        dscr=dscr,
        coc=coc,
        cap_rate=cap_rate,
        breakeven_occ=breakeven_occ,
        monthly_cashflow=monthly_cashflow,
    )
