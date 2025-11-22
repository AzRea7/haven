import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from haven.adapters.config import config
from haven.analysis.finance_batch import compute_financial_metrics_df
from haven.domain.assumptions import UnderwritingAssumptions

OUT_DIR = Path("data/derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_properties_df(limit: int = 100_000) -> pd.DataFrame:
    engine = create_engine(config.DB_URI)
    query = text(
        """
        SELECT
            id,
            list_price AS purchase_price,
            COALESCE(est_rent, list_price * 0.008) AS est_rent,
            COALESCE(taxes_annual, 0) AS taxes_annual,
            COALESCE(insurance_annual, 0) AS insurance_annual,
            COALESCE(hoa_monthly, 0) AS hoa_monthly
        FROM properties
        WHERE list_price IS NOT NULL
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})
    return df


def main() -> None:
    df = load_properties_df()

    assumptions = UnderwritingAssumptions(
        vacancy_rate=config.VACANCY_RATE,
        maintenance_rate=config.MAINTENANCE_RATE,
        property_mgmt_rate=config.PROPERTY_MGMT_RATE,
        capex_rate=config.CAPEX_RATE,
        closing_cost_pct=config.DEFAULT_CLOSING_COST_PCT,
        min_dscr_good=config.MIN_DSCR_GOOD,
    )

    print(f"Batch-scoring {len(df)} properties...")

    t0 = time.perf_counter()
    result = compute_financial_metrics_df(
        df,
        assumptions,
        down_payment_pct=0.25,            # or from config
        interest_rate_annual=0.065,
        loan_term_years=30,
    )
    dt = time.perf_counter() - t0

    df["dscr"] = result.dscr
    df["cash_on_cash_return"] = result.cash_on_cash_return
    df["breakeven_occupancy_pct"] = result.breakeven_occupancy_pct
    df["noi_annual"] = result.noi_annual
    df["cap_rate"] = result.cap_rate

    print(f"Computed metrics in {dt:.3f}s for {len(df)} properties "
          f"({len(df)/dt:.1f} props/s).")

    out_path = OUT_DIR / "property_finance_metrics.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
