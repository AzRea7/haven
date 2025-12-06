# entrypoints/cli/audit_flip_classifier.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split

from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)

# Features MUST match what deal_analyzer/_compute_flip_probability uses
FEATURE_COLS: List[str] = [
    "dscr",
    "cash_on_cash_return",
    "breakeven_occupancy_pct",
    "list_price",
    "sqft",
    "days_on_market",
]


def load_flip_training(path: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load flip_training.parquet built by build_flip_training_from_csv.py.

    Expected columns:
      - FEATURE_COLS (dscr, cash_on_cash_return, breakeven_occupancy_pct, list_price, sqft, days_on_market)
      - is_good_flip (bool/int label)
      - actual_roi (optional, for calibration sanity checks)
    """
    if not path.exists():
        raise SystemExit(
            f"Flip training parquet not found at {path}. "
            "Run entrypoints/cli/build_flip_training_from_csv.py first."
        )

    df = pd.read_parquet(path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Flip training frame missing required feature cols: {missing}")

    if "is_good_flip" not in df.columns:
        raise SystemExit("Flip training frame must contain 'is_good_flip' column.")

    df = df.copy()

    # Coerce features to numeric
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Label
    y = df["is_good_flip"].astype(int).to_numpy()

    # Features
    X = df[FEATURE_COLS].to_numpy(dtype=float)

    # Drop rows with NaNs in features or labels
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    if X.shape[0] == 0:
        raise SystemExit("No valid rows after cleaning flip training data.")

    logger.info(
        "flip_training_loaded",
        extra={
            "rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "positive_rate": float(y.mean()),
        },
    )

    return pd.DataFrame(X, columns=FEATURE_COLS), y, FEATURE_COLS


def train_flip_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier for "good flip" vs "not good".
    """

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
    )

    return model


def evaluate_classifier(
    model: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ROC AUC, precision/recall/F1, and calibration-ish metrics.
    """
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics: Dict[str, float] = {}

    # ROC AUC
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_val, proba))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    # Precision, recall, F1 for positive class
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, preds, average="binary", zero_division=0
    )
    metrics["precision"] = float(prec)
    metrics["recall"] = float(rec)
    metrics["f1"] = float(f1)

    # Brier score for calibration-ish sense
    try:
        metrics["brier"] = float(brier_score_loss(y_val, proba))
    except ValueError:
        metrics["brier"] = float("nan")

    # Simple calibration: bucketed predicted probs vs actual rate
    bins = np.linspace(0.0, 1.0, 6)  # 0,0.2,...,1.0
    bucket_ids = np.digitize(proba, bins) - 1
    calib_rows: List[Dict[str, float]] = []
    for b in range(len(bins)):
        mask = bucket_ids == b
        if mask.sum() < 5:
            continue
        avg_pred = float(proba[mask].mean())
        avg_true = float(y_val[mask].mean())
        calib_rows.append(
            {
                "bin_low": float(bins[b] if b < len(bins) else 1.0),
                "n": int(mask.sum()),
                "avg_pred": avg_pred,
                "avg_true": avg_true,
            }
        )

    logger.info("flip_classifier_calibration", extra={"bins": calib_rows})

    print("Flip classifier metrics:")
    for k, v in metrics.items():
        if isinstance(v, float) and np.isfinite(v):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if calib_rows:
        print("\nCalibration buckets (avg_pred vs avg_true):")
        for row in calib_rows:
            print(
                f"  bin≥{row['bin_low']:.1f}: n={row['n']}, "
                f"avg_pred={row['avg_pred']:.3f}, avg_true={row['avg_true']:.3f}"
            )

    return metrics


def _split_train_val(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Robust train/val splitter that handles tiny datasets.

    - If n_samples < 5: use the entire dataset as both train and val (just to get a model).
    - Else:
        * Only use stratify when we have enough samples per class.
        * Fall back to non-stratified split otherwise.
    """
    n_samples = len(y)
    unique, counts = np.unique(y, return_counts=True)
    n_classes = len(unique)

    if n_samples < 5:
        # Too tiny to split meaningfully; train and eval on all data.
        print(
            f"WARNING: flip_training has only {n_samples} samples; "
            "using full data for both train and validation."
        )
        return X, X, y, y

    # target size in samples
    raw_test = int(round(test_size * n_samples))
    test_n = max(1, raw_test)

    # Check whether stratification is feasible:
    # need at least 2 classes, and enough samples per class
    # so that each class can appear in both train and test.
    can_stratify = False
    if n_classes > 1:
        # minimal per-class count
        min_count = int(counts.min())
        # naive condition: min_count >= 2 and test_n >= n_classes
        # (so test set can hold at least one sample of each class)
        if min_count >= 2 and test_n >= n_classes:
            can_stratify = True

    stratify_arg = y if can_stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_arg,
    )

    if not can_stratify:
        print(
            "NOTE: Not enough samples per class for stratified split; "
            "using non-stratified train/val split."
        )

    return X_train, X_val, y_train, y_val


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train and audit LightGBM flip classifier on flip_training.parquet."
    )
    ap.add_argument(
        "--training-path",
        type=Path,
        default=Path("data/processed/flip_training.parquet"),
        help="Flip training parquet.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("models/flip_classifier_lgb.joblib"),
        help="Output path for flip classifier bundle.",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout fraction for validation (0-1).",
    )
    args = ap.parse_args()

    X, y, feature_names = load_flip_training(args.training_path)

    X_train, X_val, y_train, y_val = _split_train_val(
        X, y, test_size=args.test_size
    )

    model = train_flip_classifier(X_train, y_train, X_val, y_val)

    evaluate_classifier(model, X_val, y_val)

    bundle = {
        "feature_names": feature_names,
        "model": model,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)

    logger.info(
        "flip_classifier_saved",
        extra={
            "path": str(args.out),
            "n_features": len(feature_names),
        },
    )
    print(f"\nSaved flip classifier bundle to {args.out}")


if __name__ == "__main__":
    main()
