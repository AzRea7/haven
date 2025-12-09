# entrypoints/cli/build_arv_training_from_sold.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from haven.features.common_features import build_property_features
from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)

# Keywords we want to EXCLUDE from comps (multi-family/condo/etc.)
EXCLUDED_TYPE_KEYWORDS = (
    "condo",
    "condominium",
    "townhome",
    "townhouse",
    "apartment",
    "multi-family",
    "multi family",
    "duplex",
    "triplex",
    "quadplex",
    "manufactured",
    "mobile",
    "trailer",
)

# Optional positive SFR hints if you want to tighten further later
SFR_HINT_KEYWORDS = (
    "single_family",
    "single family",
    "sfr",
    "house",
    "detached",
)


def _infer_combined_type(df: pd.DataFrame) -> pd.Series:
    """
    Build a combined type string per row using:

      - df['property_type'] if present
      - + any Zillow-style home-type metadata column if present

    This lets us filter out condos/townhomes/apartments robustly.
    """
    # Normalize property_type
    if "property_type" not in df.columns:
        df["property_type"] = ""
    prop_type = df["property_type"].astype(str).str.strip().str.lower()

    # Try to find a Zillow-style home type column
    home_type_col: str | None = None
    for cand in ("zillow_home_type", "home_type", "homeType"):
        if cand in df.columns:
            home_type_col = cand
            break

    if home_type_col is not None:
        home_type = df[home_type_col].astype(str).str.strip().str.lower()
    else:
        home_type = pd.Series([""] * len(df), index=df.index)

    combined = (prop_type.fillna("") + " " + home_type.fillna("")).str.strip()

    return combined


def build_training(
    sold_parquet: Path,
    out_parquet: Path,
) -> None:
    """
    Build ARV training frame from ATTOM/Zillow-style sold properties parquet.

    Expects sold_parquet to contain at least:
      - sold_price
      - zipcode
      - (optional) list_price
      - (optional) property_type
      - (optional) Zillow home type columns:
           zillow_home_type / home_type / homeType

    Output:
      - out_parquet with numeric feature matrix + 'target_arv' column.
    """
    if not sold_parquet.exists():
        raise SystemExit(f"Sold comps parquet not found: {sold_parquet}")

    df = pd.read_parquet(sold_parquet)

    if "sold_price" not in df.columns:
        raise SystemExit("sold_properties.parquet must contain 'sold_price' column")

    if "zipcode" not in df.columns:
        raise SystemExit("sold_properties.parquet must contain 'zipcode' column")

    # Work on a copy to avoid view issues
    df = df.copy()

    # Define ARV target as sold_price
    df["target_arv"] = df["sold_price"].astype(float)

    # For consistency, create a list_price column if missing
    if "list_price" not in df.columns:
        df["list_price"] = df["sold_price"].astype(float)

    # Normalize zipcode as zero-padded string
    df["zipcode"] = df["zipcode"].astype(str).str.strip().str.zfill(5)

    # Normalize property_type if present
    if "property_type" not in df.columns:
        df["property_type"] = ""
    else:
        df["property_type"] = df["property_type"].astype(str).str.strip().str.lower()

    # --- Use Zillow metadata to filter to SFR-like comps -------------------
    combined_type = _infer_combined_type(df)

    before_rows = len(df)

    # 1) Drop obvious non-SFR types (condos, townhomes, apartments, etc.)
    excl_mask = combined_type.str.contains("|".join(EXCLUDED_TYPE_KEYWORDS), na=False)
    df = df.loc[~excl_mask].copy()

    after_excl_rows = len(df)

    # 2) (Optional) If we find enough rows with SFR hints, restrict to those.
    sfr_mask = combined_type.str.contains("|".join(SFR_HINT_KEYWORDS), na=False)
    sfr_rows = sfr_mask.sum()

    # Only apply positive SFR filter if it doesn't nuke the dataset
    if sfr_rows > 0 and sfr_rows >= 0.3 * len(df):
        df = df.loc[sfr_mask].copy()

    after_sfr_rows = len(df)

    logger.info(
        "build_arv_training_from_sold_input",
        extra={
            "rows_initial": before_rows,
            "rows_after_exclude": after_excl_rows,
            "rows_after_sfr_filter": after_sfr_rows,
            "cols": list(df.columns),
        },
    )

    # --- Build features using the shared pipeline --------------------------
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
            "out_path": str(out_parquet),
            "rows": len(out),
            "cols": list(out.columns),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ARV training dataset from ATTOM/Zillow sold properties parquet."
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
