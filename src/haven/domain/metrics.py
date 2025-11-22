from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PortfolioMetrics:
    """
    Aggregated DSCR / CoC stats across a batch of properties.

    This is the 'reduction' result of a map-style per-property computation.
    """
    n_properties: int
    mean_dscr: float
    p5_dscr: float
    p50_dscr: float
    p95_dscr: float
    mean_coc: float
    p5_coc: float
    p50_coc: float
    p95_coc: float


def compute_dscr(noi: np.ndarray, annual_debt_service: np.ndarray) -> np.ndarray:
    """
    DSCR = NOI / Annual Debt Service.

    We guard against division by zero to keep the vector op well-defined.
    """
    noi = np.asarray(noi, dtype=float)
    debt = np.asarray(annual_debt_service, dtype=float)
    # Avoid division by zero: where debt == 0, we treat DSCR as +inf.
    with np.errstate(divide="ignore", invalid="ignore"):
        dscr = noi / debt
        dscr[debt == 0.0] = np.inf
    return dscr


def compute_cash_on_cash_return(
    annual_cash_flow: np.ndarray,
    total_cash_invested: np.ndarray,
) -> np.ndarray:
    """
    Cash-on-cash return = Annual Cash Flow / Total Cash Invested.

    Again we guard against division by zero.
    """
    cf = np.asarray(annual_cash_flow, dtype=float)
    invested = np.asarray(total_cash_invested, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        coc = cf / invested
        coc[invested == 0.0] = np.nan
    return coc


def summarize_portfolio(
    dscr: np.ndarray,
    coc: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> PortfolioMetrics:
    """
    Reduction step: take per-property DSCR / CoC arrays and collapse them into
    summary statistics for the whole portfolio.
    """
    dscr = np.asarray(dscr, dtype=float)
    coc = np.asarray(coc, dtype=float)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        dscr = dscr[mask]
        coc = coc[mask]

    n = int(dscr.shape[0])

    if n == 0:
        # Degenerate case: empty portfolio.
        return PortfolioMetrics(
            n_properties=0,
            mean_dscr=float("nan"),
            p5_dscr=float("nan"),
            p50_dscr=float("nan"),
            p95_dscr=float("nan"),
            mean_coc=float("nan"),
            p5_coc=float("nan"),
            p50_coc=float("nan"),
            p95_coc=float("nan"),
        )

    def _nanquantile(x: np.ndarray, q: float) -> float:
        return float(np.nanquantile(x, q))

    return PortfolioMetrics(
        n_properties=n,
        mean_dscr=float(np.nanmean(dscr)),
        p5_dscr=_nanquantile(dscr, 0.05),
        p50_dscr=_nanquantile(dscr, 0.50),
        p95_dscr=_nanquantile(dscr, 0.95),
        mean_coc=float(np.nanmean(coc)),
        p5_coc=_nanquantile(coc, 0.05),
        p50_coc=_nanquantile(coc, 0.50),
        p95_coc=_nanquantile(coc, 0.95),
    )
