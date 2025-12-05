# scripts/eval_rent_model.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator

DATA_PATH = Path("data/curated/rent_training.parquet")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.maximum(y_true, 1.0)
    return float(np.mean(np.abs(y_true - y_pred) / y_true))


def run(sample_size: int = 3000) -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing rent training data at {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    required_cols = ["bedrooms", "bathrooms", "sqft", "zipcode", "rent"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in rent_training.parquet: {missing}")

    # Basic cleaning
    work = df.dropna(subset=required_cols).copy()
    work = work[work["sqft"] > 0]
    work = work[work["rent"] > 0]

    if work.empty:
        raise SystemExit("No valid rows in rent_training.parquet after cleaning.")

    if len(work) > sample_size:
        work = work.sample(sample_size, random_state=42)

    est = LightGBMRentEstimator()

    preds: list[float] = []
    targets: list[float] = []

    for _, row in work.iterrows():
        rent_hat = est.predict_unit_rent(
            bedrooms=row["bedrooms"],
            bathrooms=row["bathrooms"],
            sqft=row["sqft"],
            zipcode=str(row["zipcode"]),
            property_type=str(row.get("property_type", "single_family")),
        )
        preds.append(rent_hat)
        targets.append(row["rent"])

    y_true = np.asarray(targets, dtype=float)
    y_pred = np.asarray(preds, dtype=float)

    overall_mae = mae(y_true, y_pred)
    overall_mape = mape(y_true, y_pred)

    print("=== Rent Model Evaluation ===")
    print(f"n_samples   : {len(y_true)}")
    print(f"MAE         : ${overall_mae:,.0f}")
    print(f"MAPE        : {overall_mape * 100:.1f}%")
    print()

    # By-ZIP breakdown for the largest few zips
    work = work.assign(y_true=y_true, y_pred=y_pred)
    work["abs_err"] = (work["y_true"] - work["y_pred"]).abs()
    work["ape"] = (work["abs_err"] / work["y_true"]).clip(0, 5)

    grouped = (
        work.groupby("zipcode")
        .agg(
            n=("rent", "size"),
            mae_zip=("abs_err", "mean"),
            mape_zip=("ape", "mean"),
        )
        .sort_values("n", ascending=False)
        .head(10)
    )

    print("Top ZIPs by sample count:")
    for zip_code, row in grouped.iterrows():
        print(
            f"  ZIP {zip_code}: n={int(row['n'])}, "
            f"MAE=${row['mae_zip']:.0f}, MAPE={row['mape_zip'] * 100:.1f}%"
        )


if __name__ == "__main__":
    run()
