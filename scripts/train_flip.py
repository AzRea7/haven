# scripts/train_flip.py
"""
Train a calibrated flip classifier from curated deal history.

Expected input:
  data/curated/flip_training.parquet with columns:
    - is_good: 1 if outcome was acceptable (profit/return threshold hit)
    - numeric features used for screening/scoring

Outputs:
  models/flip_logit_calibrated.joblib

This is intentionally simple & explainable:
  - StandardScaler + LogisticRegression
  - Calibrated via cross-validation (sigmoid)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = Path("data/curated/flip_training.parquet")
DEFAULT_OUTPUT = Path("models/flip_logit_calibrated.joblib")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "You must build data/curated/flip_training.parquet from your historical deals."
        )
    df = pd.read_parquet(path)
    if "is_good" not in df.columns:
        raise ValueError("flip_training dataset must contain 'is_good' column.")
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    y = df["is_good"].astype(int).to_numpy()

    # Heuristic: use all numeric columns except target.
    numeric_cols = [
        c
        for c in df.columns
        if c != "is_good" and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found for flip model.")

    X = df[numeric_cols].fillna(0.0)

    return X, y, numeric_cols


def train_model(X: pd.DataFrame, y: np.ndarray) -> CalibratedClassifierCV:
    base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=1000)),
        ]
    )
    clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=5)
    clf.fit(X, y)
    return clf


def main(input_path: Path, output_path: Path) -> None:
    df = load_data(input_path)
    X, y, cols = select_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = train_model(X_train, y_train)

    # Evaluate
    p_val = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p_val)
    print(f"[flip] Validation AUC: {auc:.3f}")

    # Attach feature names for deterministic inference
    setattr(clf, "feature_names_in_", np.array(cols))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"[flip] Saved model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to flip_training.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output model path",
    )
    args = parser.parse_args()
    main(args.input, args.output)
