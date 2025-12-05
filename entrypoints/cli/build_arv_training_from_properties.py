# entrypoints/cli/build_arv_training_from_properties.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=str,
        default="data/curated/properties.parquet",
        help="Input properties parquet (current listings).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/curated/arv_training.parquet",
        help="Output ARV training parquet.",
    )
    args = parser.parse_args()

    base_path = Path(args.base)
    out_path = Path(args.out)

    if not base_path.exists():
        raise SystemExit(
            f"ERROR: base properties file not found at {base_path}. "
            "Run your ingest + build_features_parallel pipeline first."
        )

    df = pd.read_parquet(base_path)

    # Required columns
    required = ["list_price", "bedrooms", "bathrooms", "sqft"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(
                f"ERROR: properties.parquet missing required column '{col}'. "
                "Inspect data/curated/properties.parquet and adjust this script."
            )

    # Normalize ZIP-ish column if present
    if "zip" in df.columns:
        df["zip"] = df["zip"].astype(str).str.zfill(5)
    elif "zipcode" in df.columns:
        df["zip"] = df["zipcode"].astype(str).str.zfill(5)
    # If neither exists, we just proceed without ZIP as a feature.

    # Year built optional
    if "year_built" not in df.columns:
        df["year_built"] = 0

    # Filter out garbage rows
    df = df[
        df["list_price"].notna()
        & df["sqft"].notna()
        & (df["sqft"] > 0)
    ]

    if df.empty:
        raise SystemExit(
            "ERROR: No valid rows in properties.parquet for ARV training "
            "(list_price / sqft missing or zero)."
        )

    df = df.copy()
    # Use list_price as a proxy for ARV target
    df["target_arv"] = df["list_price"].astype(float)
    df["psf"] = df["target_arv"] / df["sqft"]

    # Keep a clean set of columns: numeric features + target_arv
    keep_cols = []

    for c in df.columns:
        if c == "target_arv":
            keep_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            keep_cols.append(c)

    df_out = df[keep_cols].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)

    print(
        f"wrote ARV training → {out_path} rows={len(df_out)} "
        f"with numeric features={ [c for c in keep_cols if c != 'target_arv'] }"
    )


if __name__ == "__main__":
    main()
