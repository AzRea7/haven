# entrypoints/cli/retrain_rent_with_neighborhood.py
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


ALPHAS: List[float] = [0.1, 0.5, 0.9]


def load_rent_and_neighborhood(
    rent_path: Path, neighborhood_csv: Path
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if not rent_path.exists():
        raise SystemExit(f"Rent training parquet not found: {rent_path}")

    df = pd.read_parquet(rent_path)

    if "target_rent" not in df.columns:
        raise SystemExit("Rent training parquet must contain 'target_rent' column.")

    if "zipcode" not in df.columns:
        raise SystemExit("Rent training parquet must contain 'zipcode' column.")

    df["zipcode"] = df["zipcode"].astype(str).str.strip().str.zfill(5)

    if not neighborhood_csv.exists():
        raise SystemExit(f"Neighborhood CSV not found: {neighborhood_csv}")

    nb = pd.read_csv(neighborhood_csv)
    if "zipcode" not in nb.columns:
        raise SystemExit("Neighborhood CSV must contain 'zipcode' column.")

    nb["zipcode"] = nb["zipcode"].astype(str).str.strip().str.zfill(5)

    merged = df.merge(nb, on="zipcode", how="left")

    # Drop rows with missing target_rent
    y_raw = merged["target_rent"].astype(float)
    mask_finite = np.isfinite(y_raw.to_numpy())
    merged = merged.loc[mask_finite].reset_index(drop=True)
    y = merged["target_rent"].astype(float).to_numpy()

    # Numeric feature columns (exclude target)
    numeric = merged.select_dtypes(include=[np.number])
    if "target_rent" in numeric.columns:
        X = numeric.drop(columns=["target_rent"])
    else:
        X = numeric

    feature_names = list(X.columns)
    if not feature_names:
        raise SystemExit("No numeric features found for rent model.")

    return X, y, feature_names


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Dict[float, lgb.LGBMRegressor]:
    models: Dict[float, lgb.LGBMRegressor] = {}
    base_params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    for alpha in ALPHAS:
        print(f"Training rent quantile model alpha={alpha}...")
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **base_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",
        )
        models[alpha] = model

    return models


def evaluate_rent(y_true: np.ndarray, y_pred_med: np.ndarray) -> None:
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

    y_true = np.asarray(y_true, dtype=float)
    y_pred_med = np.asarray(y_pred_med, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred_med)
    if mask.sum() == 0:
        print("WARNING: No valid rent pairs for eval.")
        return

    y_t = y_true[mask]
    y_p = y_pred_med[mask]

    mae = mean_absolute_error(y_t, y_p)
    mape = mean_absolute_percentage_error(y_t, y_p)
    print(f"Rent MAE (median):  {mae:,.2f}")
    print(f"Rent MAPE (median): {100 * mape:,.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Retrain rent quantile model with neighborhood features."
    )
    ap.add_argument(
        "--rent-parquet",
        type=Path,
        default=Path("data/processed/rent_training.parquet"),
    )
    ap.add_argument(
        "--neighborhood-csv",
        type=Path,
        default=Path("data/raw/neighborhood_scores.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("models/rent_quantiles_with_neighborhood.joblib"),
    )
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    X, y, feature_names = load_rent_and_neighborhood(
        args.rent_parquet, args.neighborhood_csv
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    models = train_quantile_models(X_train, y_train, X_val, y_val)
    y_val_med = models[0.5].predict(X_val)
    evaluate_rent(y_val, y_val_med)

    bundle = {"alphas": ALPHAS, "feature_names": feature_names, "models": models}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"Saved rent quantile bundle to {args.out}")


if __name__ == "__main__":
    main()
