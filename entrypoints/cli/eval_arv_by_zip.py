# entrypoints/cli/eval_arv_by_zip.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


def load_bundle(model_path: Path) -> Dict:
    if not model_path.exists():
        raise SystemExit(f"ARV quantile bundle not found at {model_path}")
    bundle = joblib.load(model_path)
    if "feature_names" not in bundle or "models" not in bundle:
        raise SystemExit("ARV bundle missing 'feature_names' or 'models' keys.")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate ARV quantile model per ZIP (MAE/MAPE/error quantiles)."
    )
    ap.add_argument(
        "--training-path",
        type=Path,
        default=Path("data/processed/arv_training_from_sold.parquet"),
        help="ARV training parquet with features + target_arv + zipcode.",
    )
    ap.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/arv_quantiles.joblib"),
        help="ARV quantile bundle path.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports/arv_eval_by_zip.parquet"),
        help="Output parquet for per-ZIP metrics.",
    )
    args = ap.parse_args()

    if not args.training_path.exists():
        raise SystemExit(f"Training parquet not found: {args.training_path}")

    df = pd.read_parquet(args.training_path)

    if "target_arv" not in df.columns:
        raise SystemExit("Training parquet must contain 'target_arv'.")
    if "zipcode" not in df.columns:
        raise SystemExit("Training parquet must contain 'zipcode' column.")

    bundle = load_bundle(args.model_path)
    feature_names = bundle["feature_names"]
    models = bundle["models"]

    # pick median model
    if 0.5 in models:
        model = models[0.5]
        alpha_used = 0.5
    else:
        # fallback to "middle" alpha if 0.5 not present
        alphas = sorted(models.keys())
        model = models[alphas[len(alphas) // 2]]
        alpha_used = alphas[len(alphas) // 2]

    logger.info(
        "eval_arv_by_zip_start",
        extra={
            "rows": len(df),
            "alpha_used": alpha_used,
            "feature_names": feature_names,
        },
    )

    X = df[feature_names]
    y_true = df["target_arv"].astype(float).to_numpy()
    zipcodes = df["zipcode"].astype(str).to_numpy()

    y_pred = model.predict(X)

    eval_df = pd.DataFrame(
        {
            "zipcode": zipcodes,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    eval_df["error"] = eval_df["y_pred"] - eval_df["y_true"]
    eval_df["abs_error"] = eval_df["error"].abs()

    # Filter non-finite for metrics
    finite_mask = np.isfinite(eval_df["y_true"]) & np.isfinite(eval_df["y_pred"])
    eval_df = eval_df.loc[finite_mask].reset_index(drop=True)

    if eval_df.empty:
        raise SystemExit("No finite y_true / y_pred pairs after filtering; cannot evaluate.")

    grouped = eval_df.groupby("zipcode")

    rows = []
    for z, g in grouped:
        if len(g) < 5:
            # Skip tiny groups; they are too noisy
            continue
        mae = mean_absolute_error(g["y_true"], g["y_pred"])
        mape = mean_absolute_percentage_error(g["y_true"], g["y_pred"])
        errors = g["error"].to_numpy()

        rows.append(
            {
                "zipcode": z,
                "n": len(g),
                "mae": mae,
                "mape_pct": float(mape * 100.0),
                "p10_error": np.percentile(errors, 10),
                "p50_error": np.percentile(errors, 50),
                "p90_error": np.percentile(errors, 90),
                "pct_under": float((errors < 0).mean() * 100.0),
                "pct_over": float((errors > 0).mean() * 100.0),
            }
        )

    if not rows:
        raise SystemExit("No ZIPs with enough samples to evaluate (need at least 5 per ZIP).")

    out_df = pd.DataFrame(rows).sort_values("mae")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    print(f"Wrote ARV eval by ZIP to {args.out}")
    print("Top 10 ZIPs by MAE:")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
