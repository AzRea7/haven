# entrypoints/cli/build_rent_training_from_csv.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


def build_rent_training(csv_path: Path, out_parquet: Path) -> None:
    """
    Build rent_training.parquet from a simple CSV of rent observations.

    Expected CSV columns:

      address            (string, optional but useful)
      zipcode            (string or int, REQUIRED)
      sqft               (numeric, REQUIRED)
      bedrooms           (numeric, REQUIRED)
      bathrooms          (numeric, REQUIRED)
      property_type      (string, optional, defaults to 'single_family')
      target_rent        (numeric, REQUIRED) -- achieved / market rent

    Example:

      address,zipcode,sqft,bedrooms,bathrooms,property_type,target_rent
      123 Main St,48009,950,2,1,condo_townhome,1900
      456 Oak St,48009,1200,3,2,single_family,2300
    """
    if not csv_path.exists():
        raise SystemExit(f"Rent observations CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["zipcode", "sqft", "bedrooms", "bathrooms", "target_rent"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Rent observations CSV missing required columns: {missing}")

    df = df.copy()

    # Coerce numeric fields
    for col in ["sqft", "bedrooms", "bathrooms", "target_rent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize zipcode, property_type
    df["zipcode"] = df["zipcode"].astype(str).str.strip().str.zfill(5)
    if "property_type" not in df.columns:
        df["property_type"] = "single_family"
    else:
        df["property_type"] = df["property_type"].astype(str).str.strip()

    # Drop rows with missing target_rent or zipcode
    mask = (
        df["target_rent"].notnull()
        & np.isfinite(df["target_rent"])
        & df["zipcode"].notnull()
        & (df["zipcode"].str.len() > 0)
    )
    df = df.loc[mask].reset_index(drop=True)

    if df.empty:
        raise SystemExit("No valid rent observations after cleaning; nothing to train on.")

    logger.info(
        "rent_training_built",
        extra={"rows": len(df), "cols": list(df.columns)},
    )

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    print(
        f"Wrote rent_training parquet to {out_parquet} "
        f"(rows={len(df)}, cols={len(df.columns)})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build rent_training.parquet from rent observations CSV."
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/rent_observations.csv"),
        help="Input CSV with rent observations.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/rent_training.parquet"),
        help="Output parquet for rent training.",
    )
    args = ap.parse_args()

    build_rent_training(args.csv, args.out)


if __name__ == "__main__":
    main()
