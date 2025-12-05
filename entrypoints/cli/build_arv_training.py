# entrypoints/cli/build_arv_training.py
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import argparse
import pandas as pd

from haven.adapters.logging_utils import get_logger
from haven.adapters.storage import read_df, write_df

log = get_logger(__name__)


def _choose_price_col(df: pd.DataFrame) -> str:
    # Prefer explicit after-rehab sale price if present
    if "sale_price_after_rehab" in df.columns:
        return "sale_price_after_rehab"
    if "sale_price" in df.columns:
        return "sale_price"
    if "sold_price" in df.columns:
        return "sold_price"
    raise SystemExit(
        "Could not find a sale price column. Expected one of: "
        "'sale_price_after_rehab', 'sale_price', 'sold_price'. "
        "Inspect data/raw/sold_properties.parquet and adjust build_arv_training.py."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sold",
        type=str,
        default="data/raw/sold_properties.parquet",
        help="Path to raw sold properties parquet file.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="data/curated/arv_training.parquet",
        help="Output ARV training frame.",
    )
    args = ap.parse_args()

    sold_path = Path(args.sold)
    out_path = Path(args.out)

    if not sold_path.exists():
        raise SystemExit(
            f"Sold properties file not found at {sold_path}. "
            "Make sure you have exported sold comps to this path."
        )

    log.info("Reading sold properties from %s", sold_path)
    df = read_df(str(sold_path))

    price_col = _choose_price_col(df)
    log.info("Using %s as ARV target", price_col)

    # Normalize ZIP column
    if "zip" in df.columns:
        df["zip"] = df["zip"].astype(str).str.zfill(5)
    elif "zipcode" in df.columns:
        df["zip"] = df["zipcode"].astype(str).str.zfill(5)
    else:
        raise SystemExit(
            "Sold properties are missing 'zip' or 'zipcode' column. "
            "Add a ZIP field or adjust build_arv_training.py."
        )

    # Basic required features
    for col in ["bedrooms", "bathrooms", "sqft"]:
        if col not in df.columns:
            raise SystemExit(f"Sold properties are missing required column: {col}")

    # Optional features
    if "year_built" not in df.columns:
        df["year_built"] = 0

    # Filter out junk rows
    df = df[df[price_col].notna()]
    df = df[df["sqft"].notna() & (df["sqft"] > 0)]

    if df.empty:
        raise SystemExit("No valid rows left after filtering sold properties for ARV training.")

    df = df.copy()
    df["target_arv"] = df[price_col].astype(float)
    df["psf"] = df["target_arv"] / df["sqft"]

    # Optional: time features if available
    if "sold_date" in df.columns:
        s = pd.to_datetime(df["sold_date"], errors="coerce")
        df["sold_year"] = s.dt.year.fillna(0).astype(int)
        df["sold_month"] = s.dt.month.fillna(0).astype(int)

    # Keep a clean set of columns
    keep_cols = [
        "zip",
        "bedrooms",
        "bathrooms",
        "sqft",
        "year_built",
        "psf",
        "target_arv",
    ]
    for extra in ["sold_year", "sold_month"]:
        if extra in df.columns:
            keep_cols.append(extra)

    df_out = df[keep_cols].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_df(df_out, str(out_path))

    log.info(
        "Wrote ARV training frame -> %s rows=%d",
        out_path,
        len(df_out),
    )
    print(f"wrote ARV training -> {out_path} rows={len(df_out)}")


if __name__ == "__main__":
    main()
