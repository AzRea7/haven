# entrypoints/cli/train_arv_quantiles.py
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


def load_training(path: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load ARV training frame and return:
      X (features), y (target_arv), feature_names.

    Assumes the parquet was built by build_arv_training_from_sold.py:
      - numeric feature columns
      - target_arv column (float)
    """
    if not path.exists():
        raise SystemExit(
            f"ERROR: ARV training file not found at {path}. "
            "Run entrypoints/cli/build_arv_training_from_sold.py first."
        )

    df = pd.read_parquet(path)

    if "target_arv" not in df.columns:
        raise SystemExit("ERROR: ARV training frame must contain 'target_arv' column.")

    df = df.copy()

    # Drop rows with NaN / inf targets
    y_raw = df["target_arv"].astype(float)
    mask_finite = np.isfinite(y_raw.to_numpy())
    df = df.loc[mask_finite].reset_index(drop=True)
    y = df["target_arv"].astype(float).to_numpy()

    # Use all numeric columns except target_arv as features
    numeric_df = df.select_dtypes(include=[np.number])
    if "target_arv" in numeric_df.columns:
        X = numeric_df.drop(columns=["target_arv"])
    else:
        # Fallback if something weird happens
        X = numeric_df

    feature_names = list(X.columns)

    if X.empty or len(feature_names) == 0:
        raise SystemExit("ERROR: No numeric feature columns found for ARV training.")

    if len(y) != len(X):
        raise SystemExit(
            f"ERROR: Feature/target length mismatch: X={len(X)}, y={len(y)}."
        )

    return X, y, feature_names


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Dict[float, lgb.LGBMRegressor]:
    """
    Train one LightGBM quantile regressor per alpha.
    """
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
        print(f"Training quantile model for alpha={alpha}...")

        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            **base_params,
        )

        # LightGBM python package changed APIs; avoid deprecated 'verbose' kw
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",
        )

        models[alpha] = model

    return models


def evaluate_basic(y_true: np.ndarray, y_pred_dict: Dict[float, np.ndarray]) -> None:
    """
    Basic evaluation:
      - MAE and MAPE for the median (alpha=0.5)
      - Warn and skip if NaNs sneak into y_true or predictions.
    """
    if 0.5 not in y_pred_dict:
        print("WARNING: No alpha=0.5 prediction found; skipping basic metrics.")
        return

    y_pred_median = np.asarray(y_pred_dict[0.5], dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if y_true.shape[0] != y_pred_median.shape[0]:
        min_len = min(len(y_true), len(y_pred_median))
        y_true = y_true[:min_len]
        y_pred_median = y_pred_median[:min_len]

    # Filter out NaNs / infs from both
    mask = np.isfinite(y_true) & np.isfinite(y_pred_median)
    if mask.sum() == 0:
        print("WARNING: No finite pairs in validation set; cannot compute metrics.")
        return

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred_median[mask]

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)

    print(f"Validation MAE (median):  {mae:,.2f}")
    print(f"Validation MAPE (median): {100 * mape:,.2f}%")

    # A tiny bit of extra introspection
    errors = y_pred_clean - y_true_clean
    print(
        "Error quantiles (pred - true) [p10, p50, p90]:",
        np.percentile(errors, [10, 50, 90]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM quantile ARV models from training parquet."
    )
    parser.add_argument(
        "--training-path",
        type=Path,
        default=Path("data/processed/arv_training_from_sold.parquet"),
        help="Parquet file with features + target_arv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/arv_quantiles.joblib"),
        help="Output path for quantile model bundle.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for validation.",
    )
    args = parser.parse_args()

    X, y, feature_names = load_training(args.training_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
    )

    models = train_quantile_models(X_train, y_train, X_val, y_val)

    # Predictions on validation set for evaluation
    y_val_pred: Dict[float, np.ndarray] = {}
    for alpha, model in models.items():
        y_val_pred[alpha] = model.predict(X_val)

    evaluate_basic(y_val, y_val_pred)

    # Bundle the models + metadata in a way arv_quantile_bundle can use
    bundle = {
        "alphas": ALPHAS,
        "feature_names": feature_names,
        "models": models,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)

    print(f"Saved ARV quantile bundle to {args.out}")


if __name__ == "__main__":
    main()
