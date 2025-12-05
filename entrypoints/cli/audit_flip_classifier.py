from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from haven.features.common_features import build_property_features


def load_flip_training(path: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if not path.exists():
        raise SystemExit(f"Flip training parquet not found: {path}")

    df = pd.read_parquet(path)

    if "is_good_flip" not in df.columns:
        raise SystemExit("Flip training frame must contain 'is_good_flip' column (0/1).")

    y = df["is_good_flip"].astype(int).to_numpy()

    feat = build_property_features(df)
    return feat.X, y, feat.feature_names


def train_flip_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        objective="binary",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        verbose=50,
    )
    return model


def audit_model(
    model: lgb.LGBMClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_val, proba)
    ap = average_precision_score(y_val, proba)

    print(f"Validation ROC AUC: {roc:.3f}")
    print(f"Validation Average Precision: {ap:.3f}")
    print("Classification report (threshold=0.5):")
    print(classification_report(y_val, preds))

    # Calibration bucket check
    frac_pos, mean_pred = calibration_curve(y_val, proba, n_bins=10)
    print("Calibration (mean_pred, frac_positive) bins:")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"  p̂={mp:.2f} -> freq={fp:.2f}")


def cross_val_roc(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> None:
    print(f"Running {n_splits}-fold CV ROC AUC...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        objective="binary",
    )
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    print(f"CV ROC AUC: mean={scores.mean():.3f}, std={scores.std():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and retrain flip classifier.")
    parser.add_argument(
        "--training-path",
        type=Path,
        default=Path("data/processed/flip_training.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/flip_classifier_lgb.joblib"),
    )
    args = parser.parse_args()

    X, y, feature_names = load_flip_training(args.training_path)
    print(f"Loaded flip training with {len(y)} rows and {len(feature_names)} features.")

    cross_val_roc(X.to_numpy(), y)

    X_train, X_val, y_train, y_val = train_test_split(
        X.to_numpy(),
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = train_flip_classifier(X_train, y_train, X_val, y_val)
    audit_model(model, X_val, y_val)

    payload = {
        "model": model,
        "feature_names": feature_names,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.out)
    print(f"Saved flip classifier to {args.out}")


if __name__ == "__main__":
    main()
