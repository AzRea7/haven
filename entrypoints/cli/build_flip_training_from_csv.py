# entrypoints/cli/build_flip_training_from_csv.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


def build_flip_training(
    csv_path: Path,
    out_parquet: Path,
    roi_threshold: float,
) -> None:
    """
    Build flip_training.parquet from a CSV of historical flip deals.

    EXPECTED COLUMNS (minimal):

      dscr                      (numeric)
      cash_on_cash_return       (numeric)
      breakeven_occupancy_pct   (numeric)
      list_price                (numeric)
      sqft                      (numeric)
      days_on_market            (numeric)
      actual_roi                (numeric, realized ROI, e.g. 0.22 for 22%)

    You can add extra fields; they'll be carried through.

    This script will compute:

      is_good_flip = (actual_roi >= roi_threshold)

    and save everything to data/processed/flip_training.parquet, which
    is consumed by entrypoints/cli/audit_flip_classifier.py
    """
    if not csv_path.exists():
        raise SystemExit(f"Flip deals CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required: List[str] = [
        "dscr",
        "cash_on_cash_return",
        "breakeven_occupancy_pct",
        "list_price",
        "sqft",
        "days_on_market",
        "actual_roi",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Flip deals CSV missing required columns: {missing}")

    df = df.copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid ROI rows
    mask = df["actual_roi"].notnull() & np.isfinite(df["actual_roi"])
    df = df.loc[mask].reset_index(drop=True)

    if df.empty:
        raise SystemExit("No valid flip rows with actual_roi after cleaning; nothing to train on.")

    df["is_good_flip"] = df["actual_roi"] >= float(roi_threshold)

    logger.info(
        "flip_training_built",
        extra={
            "rows": len(df),
            "good_flips": int(df["is_good_flip"].sum()),
            "roi_threshold": roi_threshold,
        },
    )

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    print(
        f"Wrote flip_training parquet to {out_parquet} "
        f"(rows={len(df)}, good_flips={int(df['is_good_flip'].sum())})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build flip_training.parquet from historical flip deals CSV."
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/flip_deals.csv"),
        help="Input CSV with flip deals.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/flip_training.parquet"),
        help="Output parquet for flip classifier training.",
    )
    ap.add_argument(
        "--roi-threshold",
        type=float,
        default=0.18,
        help="ROI threshold above which a flip is considered 'good'.",
    )
    args = ap.parse_args()

    build_flip_training(args.csv, args.out, args.roi_threshold)


if __name__ == "__main__":
    main()
