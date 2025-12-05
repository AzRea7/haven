from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from haven.features.common_features import build_property_features


def load_rent_training(
    rent_parquet: Path,
    neighborhood_csv: Path,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if not rent_parquet.exists():
        raise SystemExit(f"Rent training parquet not found: {rent_parquet}")
    if not neighborhood_csv.exists():
        raise SystemExit(f"Neighborhood csv not found: {neighborhood_csv}")

    df = pd.read_parquet(rent_parquet)

    if "target_rent" not in df.columns:
        raise SystemExit("Rent training parquet must contain 'target_rent' column.")

    neigh = pd.read_csv(neighborhood_csv)

    if "zipcode" not in df.columns:
        raise SystemExit("Rent training data must have a 'zipcode' column to merge.")

    df = df.merge(
        neigh,
        on="zipcode",
        how="left",
        suffixes=("", "_neigh"),
    )

    y = df["target_rent"].astype(float).to_numpy()

    feat = build_property_features(
        df,
        extra_numeric_cols=["walk_score", "school_score", "crime_index", "rent_demand_index"],
    )

    return feat.X, y, feat.feature_names


def train_rent_quantiles(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
) -> Dict[float, lgb.LGBMRegressor]:
    models: Dict[float, lgb.LGBMRegressor] = {}
    for q in quantiles:
        print(f"Training rent quantile model for alpha={q}...")
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            verbose=50,
        )
        models[q] = model
    return models


def evaluate(y_true: np.ndarray, y_pred_median: np.ndarray) -> None:
    mae = mean_absolute_error(y_true, y_pred_median)
    mape = mean_absolute_percentage_error(y_true, y_pred_median)
    print(f"Rent MAE:  {mae:,.0f}")
    print(f"Rent MAPE: {mape * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain rent quantile model with neighborhood features.")
    parser.add_argument(
        "--rent-parquet",
        type=Path,
        default=Path("data/processed/rent_training.parquet"),
    )
    parser.add_argument(
        "--neighborhood-csv",
        type=Path,
        default=Path("data/raw/neighborhood_scores.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/rent_quantiles_with_neighborhood.joblib"),
    )
    args = parser.parse_args()

    X, y, feature_names = load_rent_training(args.rent_parquet, args.neighborhood_csv)

    X_train, X_val, y_train, y_val = train_test_split(
        X.to_numpy(),
        y,
        test_size=0.2,
        random_state=42,
    )

    models = train_rent_quantiles(X_train, y_train, X_val, y_val)
    y_pred_median = models[0.5].predict(X_val)
    evaluate(y_val, y_pred_median)

    payload = {
        "models": models,
        "feature_names": feature_names,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.out)
    print(f"Saved rent quantile models with neighborhood features to {args.out}")


if __name__ == "__main__":
    main()
