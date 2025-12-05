# entrypoints/cli/train_arv_quantiles.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb


def load_training(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not path.exists():
        raise SystemExit(
            f"ERROR: ARV training file not found at {path}. "
            "Run entrypoints/cli/build_arv_training_from_properties.py first."
        )

    df = pd.read_parquet(path)

    if "target_arv" not in df.columns:
        raise SystemExit("ERROR: ARV training frame must contain 'target_arv' column.")

    y = df["target_arv"].astype(float).to_numpy()

    # Use all numeric columns except target and (optionally) raw list_price
    numeric_cols: list[str] = []
    for c in df.columns:
        if c == "target_arv":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    # Optionally drop list_price so ARV isn't trivially equal to ask
    if "list_price" in numeric_cols:
        numeric_cols.remove("list_price")

    if not numeric_cols:
        raise SystemExit("ERROR: No numeric feature columns found for ARV training.")

    X = df[numeric_cols].fillna(0.0)

    print(f"Using ARV features: {numeric_cols}")
    return X, y, numeric_cols


def make_quantile_model(alpha: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        type=str,
        default="data/curated/arv_training.parquet",
        help="Input ARV training parquet.",
        dest="in_path",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="models",
        help="Directory to write arv_q10/50/90.joblib.",
    )
    args = parser.parse_args()
    in_path = Path(args.in_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, cols = load_training(in_path)

    quantiles = {
        "arv_q10.joblib": 0.10,
        "arv_q50.joblib": 0.50,
        "arv_q90.joblib": 0.90,
    }

    for fname, alpha in quantiles.items():
        print(f"Training LightGBM quantile model alpha={alpha} → {fname}")
        model = make_quantile_model(alpha)

        # Fit on a pandas DataFrame so LightGBM can infer feature names itself
        model.fit(X, y)

        out_path = outdir / fname
        joblib.dump(model, out_path)
        print(f"Saved ARV model alpha={alpha} to {out_path}")

    print("ARV quantile training complete.")


if __name__ == "__main__":
    main()
