# entrypoints/cli/build_arv_training_from_sold.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from haven.features.common_features import build_property_features
from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


def build_training(
    sold_parquet: Path,
    out_parquet: Path,
) -> None:
    """
    Build ARV training dataset directly from ATTOM-sourced sold_properties.parquet.

    Input (sold_parquet) is expected to come from:
      entrypoints/cli/fetch_sold_from_api.py

    Required columns:
      - sold_price
      - zipcode

    Optional but helpful:
      - bedrooms
      - bathrooms
      - sqft
      - year_built
      - property_type
    """
    if not sold_parquet.exists():
        raise SystemExit(f"Sold comps parquet not found: {sold_parquet}")

    df = pd.read_parquet(sold_parquet)

    if "sold_price" not in df.columns:
        raise SystemExit("sold_properties.parquet must contain 'sold_price' column")

    if "zipcode" not in df.columns:
        raise SystemExit("sold_properties.parquet must contain 'zipcode' column")

    # Work on a copy to avoid weird view issues
    df = df.copy()

    # Define ARV target as sold_price
    df["target_arv"] = df["sold_price"].astype(float)

    # For consistency, create a list_price column if missing
    if "list_price" not in df.columns:
        df["list_price"] = df["sold_price"].astype(float)

    # Normalize zipcode as zero-padded string
    df["zipcode"] = df["zipcode"].astype(str).str.strip().str.zfill(5)

    # Property type: default to single_family if missing
    if "property_type" not in df.columns:
        df["property_type"] = "single_family"
    else:
        df["property_type"] = df["property_type"].astype(str).str.strip()

    logger.info(
        "build_arv_training_from_sold_input",
        extra={
            "rows": len(df),
            "cols": list(df.columns),
        },
    )

    # Build features using the shared feature pipeline.
    feat = build_property_features(df)

    # Use the feature matrix as a plain DataFrame
    out = feat.X.copy()
    out["target_arv"] = df["target_arv"].astype(float)

    # ALSO keep zipcode for per-ZIP evaluation (non-numeric, training will ignore it)
    out["zipcode"] = df["zipcode"].astype(str)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    logger.info(
        "build_arv_training_from_sold_done",
        extra={
            "out": str(out_parquet),
            "rows": len(out),
            "cols": list(out.columns),
        },
    )
    print(
        f"Wrote ARV training parquet to {out_parquet} "
        f"with {len(out)} rows and {len(out.columns) - 2} numeric features."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ARV training dataset from ATTOM sold properties parquet."
    )
    parser.add_argument(
        "--sold-path",
        type=Path,
        default=Path("data/raw/sold_properties.parquet"),
        help="Input Parquet with sold comps.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/arv_training_from_sold.parquet"),
        help="Where to write the ARV training parquet.",
    )
    args = parser.parse_args()

    build_training(args.sold_path, args.out)


if __name__ == "__main__":
    main()
