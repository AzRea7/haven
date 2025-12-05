from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from haven.features.common_features import build_property_features


def build_training(
    sold_csv: Path,
    out_parquet: Path,
) -> None:
    if not sold_csv.exists():
        raise SystemExit(f"Sold comps CSV not found: {sold_csv}")

    df = pd.read_csv(sold_csv)

    if "sold_price" not in df.columns:
        raise SystemExit("sold_properties.csv must contain 'sold_price' column")

    # Define ARV target as sold_price
    df["target_arv"] = df["sold_price"].astype(float)

    # For consistency, create a list_price column if missing
    if "list_price" not in df.columns:
        df["list_price"] = df["sold_price"].astype(float)

    feat = build_property_features(df)

    # Save features + target_arv in a single frame
    out = feat.X.copy()
    out["target_arv"] = df["target_arv"].astype(float)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    print(f"Wrote ARV training parquet to {out_parquet}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ARV training dataset from sold properties."
    )
    parser.add_argument(
        "--sold-csv",
        type=Path,
        default=Path("data/raw/sold_properties.csv"),
        help="Input CSV with sold comps.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/arv_training_from_sold.parquet"),
        help="Where to write the ARV training parquet.",
    )
    args = parser.parse_args()
    build_training(args.sold_csv, args.out)


if __name__ == "__main__":
    main()
