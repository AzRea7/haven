from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from haven.features.common_features import build_property_features


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Redfin column names to a simple, predictable form:
    'Sold Date' -> 'solddate', 'Square Feet' -> 'squarefeet', etc.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    return df


def map_redfin_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common Redfin export columns to your canonical fields:

    - sold_price
    - sold_date
    - bedrooms
    - bathrooms
    - sqft
    - year_built
    - zipcode
    - property_type
    """
    df = normalize_columns(df)

    col_map: Dict[str, str] = {}

    # price
    for cand in ["price", "soldprice"]:
        if cand in df.columns:
            col_map[cand] = "sold_price"
            break

    # sold date
    for cand in ["solddate", "saledate"]:
        if cand in df.columns:
            col_map[cand] = "sold_date"
            break

    # bedrooms
    for cand in ["beds", "bedrooms"]:
        if cand in df.columns:
            col_map[cand] = "bedrooms"
            break

    # bathrooms
    for cand in ["baths", "bathrooms"]:
        if cand in df.columns:
            col_map[cand] = "bathrooms"
            break

    # square feet
    for cand in ["squarefeet", "finishedsqft", "sqft", "livingarea"]:
        if cand in df.columns:
            col_map[cand] = "sqft"
            break

    # year built
    for cand in ["yearbuilt"]:
        if cand in df.columns:
            col_map[cand] = "year_built"
            break

    # zipcode
    for cand in ["zip", "zipcode"]:
        if cand in df.columns:
            col_map[cand] = "zipcode"
            break

    # property type
    for cand in ["propertytype", "proptype"]:
        if cand in df.columns:
            col_map[cand] = "property_type"
            break

    if "sold_price" not in col_map.values():
        raise SystemExit(
            "Could not identify sold price column in Redfin export. "
            "Inspect the CSV and update map_redfin_to_canonical."
        )

    out = df[list(col_map.keys())].rename(columns=col_map)

    # Basic cleaning
    out["sold_price"] = (
        out["sold_price"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    # Try to convert sold_date to datetime; keep as string
    if "sold_date" in out.columns:
        out["sold_date"] = pd.to_datetime(out["sold_date"], errors="coerce").dt.date.astype(str)

    for col in ["bedrooms", "bathrooms", "sqft", "year_built"]:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace(r"[^0-9.]", "", regex=True)
                .replace("", np.nan)
            )
            if col == "bathrooms":
                out[col] = out[col].astype(float)
            else:
                out[col] = out[col].astype(float)

    if "zipcode" in out.columns:
        out["zipcode"] = out["zipcode"].astype(str).str.strip()

    if "property_type" in out.columns:
        out["property_type"] = out["property_type"].astype(str).str.strip().str.lower()
    else:
        out["property_type"] = "unknown"

    return out


def build_training_from_redfin(
    inputs: List[Path],
    out_parquet: Path,
) -> None:
    frames: List[pd.DataFrame] = []

    for p in inputs:
        if not p.exists():
            print(f"WARNING: Redfin CSV not found, skipping: {p}")
            continue
        print(f"Reading Redfin sold export: {p}")
        raw = pd.read_csv(p)
        mapped = map_redfin_to_canonical(raw)
        frames.append(mapped)

    if not frames:
        raise SystemExit("No valid Redfin CSVs provided or all missing.")

    sold = pd.concat(frames, ignore_index=True)

    # Drop rows missing sold_price or sqft (we need price per sqft)
    sold = sold.dropna(subset=["sold_price", "sqft"])

    # Use sold_price as ARV target
    sold["target_arv"] = sold["sold_price"].astype(float)

    # For feature builder, we can treat sold_price as list_price as well
    sold["list_price"] = sold["sold_price"].astype(float)

    feat = build_property_features(sold)

    # Combine features + target
    out_df = feat.X.copy()
    out_df["target_arv"] = sold["target_arv"].astype(float)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    print(
        f"Wrote ARV training parquet to {out_parquet} with "
        f"{len(out_df)} rows and {len(feat.feature_names)} features."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ARV training dataset directly from raw Redfin sold CSVs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more raw Redfin 'recently sold' CSV exports.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/arv_training_from_sold.parquet"),
        help="Where to write the ARV training parquet.",
    )
    args = parser.parse_args()

    build_training_from_redfin(args.inputs, args.out)


if __name__ == "__main__":
    main()
