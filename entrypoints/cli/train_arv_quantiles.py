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


def load_training(path: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if not path.exists():
        raise SystemExit(
            f"ERROR: ARV training file not found at {path}. "
            "Run entrypoints/cli/build_arv_training_from_redfin.py first."
        )

    df = pd.read_parquet(path)

    if "target_arv" not in df.columns:
        raise SystemExit("ERROR: ARV training frame must contain 'target_arv' column.")

    y = df["target_arv"].astype(float).to_numpy()

    # Select all numeric columns except target
    numeric_cols: list[str] = []
    for c in df.columns:
        if c == "target_arv":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    X = df[numeric_cols].astype(float)
    return X, y, numeric_cols


def train_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
) -> Dict[float, lgb.LGBMRegressor]:

    models: Dict[float, lgb.LGBMRegressor] = {}

    for q in quantiles:
        print(f"Training quantile model for alpha={q}...")

        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
        )

        # FIX: LightGBM >= 4.0 expects callbacks instead of verbose
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[
                lgb.log_evaluation(period=50),  # print evaluation every 50 iterations
            ],
        )

        models[q] = model

    return models


def evaluate_basic(y_true: np.ndarray, y_pred_median: np.ndarray) -> None:
    mae = mean_absolute_error(y_true, y_pred_median)
    mape = mean_absolute_percentage_error(y_true, y_pred_median)

    print(f"Validation MAE:  {mae:,.0f}")
    print(f"Validation MAPE: {mape * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ARV quantile models.")
    parser.add_argument(
        "--training-path",
        type=Path,
        default=Path("data/processed/arv_training_from_sold.parquet"),
        help="Parquet with numeric features and target_arv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/arv_quantiles.joblib"),
        help="Where to save the trained quantile models.",
    )
    args = parser.parse_args()

    X, y, feature_names = load_training(args.training_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X.to_numpy(),
        y,
        test_size=0.2,
        random_state=42,
    )

    models = train_quantile_models(X_train, y_train, X_val, y_val)

    # Evaluate median quantile model
    y_pred_median = models[0.5].predict(X_val)
    evaluate_basic(y_val, y_pred_median)

    payload = {"models": models, "feature_names": feature_names}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.out)
    print(f"Saved ARV quantile models to {args.out}")


if __name__ == "__main__":
    main()
