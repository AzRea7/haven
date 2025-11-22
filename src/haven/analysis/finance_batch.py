# src/haven/analysis/finance_batch.py

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from haven.domain.assumptions import UnderwritingAssumptions


@dataclass
class BatchFinanceResult:
    dscr: np.ndarray
    cash_on_cash_return: np.ndarray
    breakeven_occupancy_pct: np.ndarray
    noi_annual: np.ndarray
    cap_rate: np.ndarray


def compute_financial_metrics_df(
    df: pd.DataFrame,
    assumptions: UnderwritingAssumptions,
    *,
    down_payment_pct: float,
    interest_rate_annual: float,
    loan_term_years: int,
) -> BatchFinanceResult:
    """
    Vectorized DSCR / CoC computation over a DataFrame.

    Expected columns on df:
      - purchase_price
      - est_rent (gross monthly)
      - taxes_annual
      - insurance_annual
      - hoa_monthly
    """

    purchase_price = df["purchase_price"].to_numpy(dtype=float)
    gross_rent_monthly = df["est_rent"].to_numpy(dtype=float)

    taxes_annual = df["taxes_annual"].to_numpy(dtype=float)
    insurance_annual = df["insurance_annual"].to_numpy(dtype=float)
    hoa_monthly = df["hoa_monthly"].to_numpy(dtype=float)

    # --- Financing assumptions ---
    down_payment = purchase_price * down_payment_pct
    loan_amount = purchase_price - down_payment

    r_monthly = interest_rate_annual / 12.0
    n_months = loan_term_years * 12

    mortgage_monthly = np.zeros_like(purchase_price, dtype=float)
    # Standard mortgage payment formula, vectorized
    mask_nonzero = (loan_amount > 0) & (r_monthly > 0)
    if mask_nonzero.any():
        la = loan_amount[mask_nonzero]
        mortgage_monthly[mask_nonzero] = (
            la * (r_monthly) / (1.0 - (1.0 + r_monthly) ** (-n_months))
        )

    # --- Income / vacancy ---
    vacancy_loss = gross_rent_monthly * assumptions.vacancy_rate
    effective_rent_monthly = gross_rent_monthly - vacancy_loss

    # --- Operating expenses (very simplified, but still vectorized) ---
    taxes_monthly = taxes_annual / 12.0
    insurance_monthly = insurance_annual / 12.0

    maintenance = gross_rent_monthly * assumptions.maintenance_rate
    mgmt = gross_rent_monthly * assumptions.property_mgmt_rate
    capex = gross_rent_monthly * assumptions.capex_rate

    total_operating_monthly = (
        taxes_monthly
        + insurance_monthly
        + hoa_monthly
        + maintenance
        + mgmt
        + capex
    )

    # --- NOI ---
    noi_monthly = effective_rent_monthly - total_operating_monthly
    noi_annual = noi_monthly * 12.0

    # --- Debt service ---
    annual_debt_service = mortgage_monthly * 12.0

    dscr = np.zeros_like(purchase_price, dtype=float)
    mask_debt = annual_debt_service > 0
    dscr[mask_debt] = noi_annual[mask_debt] / annual_debt_service[mask_debt]

    # --- Cap rate ---
    cap_rate = np.zeros_like(purchase_price, dtype=float)
    mask_price = purchase_price > 0
    cap_rate[mask_price] = noi_annual[mask_price] / purchase_price[mask_price]

    # --- Cash flow & CoC ---
    cashflow_monthly_after_debt = (
        effective_rent_monthly - total_operating_monthly - mortgage_monthly
    )

    est_closing_costs = purchase_price * assumptions.closing_cost_pct
    total_cash_in = down_payment + est_closing_costs

    cash_on_cash = np.zeros_like(purchase_price, dtype=float)
    mask_cash_in = total_cash_in > 0
    cash_on_cash[mask_cash_in] = (
        cashflow_monthly_after_debt[mask_cash_in] * 12.0
    ) / total_cash_in[mask_cash_in]

    # --- Breakeven occupancy ---
    breakeven_occ = np.zeros_like(purchase_price, dtype=float)
    mask_rent = gross_rent_monthly > 0
    breakeven_occ[mask_rent] = (
        total_operating_monthly[mask_rent] + mortgage_monthly[mask_rent]
    ) / gross_rent_monthly[mask_rent]

    return BatchFinanceResult(
        dscr=dscr,
        cash_on_cash_return=cash_on_cash,
        breakeven_occupancy_pct=breakeven_occ,
        noi_annual=noi_annual,
        cap_rate=cap_rate,
    )
