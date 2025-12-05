# entrypoints/cli/score_candidates.py
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import joblib
import numpy as np
import pandas as pd

from haven.adapters.rehab_estimator import RehabEstimator

# Basic flip financial assumptions
FIN = {
    "target_margin": 0.12,      # desired profit margin on ARV (e.g., 12%)
    "buy_closing_rate": 0.02,   # buy-side closing costs as % of purchase
    "sell_closing_rate": 0.07,  # sell-side closing costs as % of ARV
}


def _ensure_series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    """Return a Series for col, filling missing with default if col doesn't exist."""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index)


def _ensure_feature_columns_for_arv(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Make sure all features expected by the ARV model exist in df.
    We reconstruct simple ones like 'psf' and 'year_built' if missing.
    """
    needed = set(feature_names)
    have = set(df.columns)

    missing = needed - have

    # Rebuild psf if the model expects it
    if "psf" in missing:
        # Safely compute price-per-sqft from list_price and sqft
        if "list_price" in df.columns and "sqft" in df.columns:
            sqft = df["sqft"].replace(0, np.nan)
            df["psf"] = (df["list_price"] / sqft).fillna(0.0)
            missing.discard("psf")
        else:
            raise SystemExit(
                "ARV model expects 'psf' but properties.parquet "
                "does not have list_price and sqft to compute it."
            )

    # Provide dummy year_built if needed (matching training, where we used 0)
    if "year_built" in missing:
        df["year_built"] = 0
        missing.discard("year_built")

    # If any other features are still missing, bail loudly
    if missing:
        raise SystemExit(
            f"ARV model expects features not present in properties.parquet: {sorted(missing)}"
        )

    return df


def _ensure_feature_columns_for_flip(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Same idea as ARV, but for the flip classifier. We reuse the same
    reconstruction logic for psf/year_built.
    """
    needed = set(feature_names)
    have = set(df.columns)

    missing = needed - have

    if "psf" in missing:
        if "list_price" in df.columns and "sqft" in df.columns:
            sqft = df["sqft"].replace(0, np.nan)
            df["psf"] = (df["list_price"] / sqft).fillna(0.0)
            missing.discard("psf")
        else:
            raise SystemExit(
                "Flip model expects 'psf' but properties.parquet "
                "does not have list_price and sqft to compute it."
            )

    if "year_built" in missing:
        df["year_built"] = 0
        missing.discard("year_built")

    if missing:
        raise SystemExit(
            f"Flip model expects features not present in properties.parquet: {sorted(missing)}"
        )

    return df


def main() -> None:
    # ----------------------------------------------------------------------
    # Load base properties
    # ----------------------------------------------------------------------
    props_path = Path("data/curated/properties.parquet")
    if not props_path.exists():
        raise SystemExit(
            f"ERROR: properties file not found at {props_path}. "
            "Run your ingest + build_features pipeline first."
        )

    df = pd.read_parquet(props_path)

    # ----------------------------------------------------------------------
    # 1. ARV quantiles using LightGBM models
    # ----------------------------------------------------------------------
    q10_model = joblib.load("models/arv_q10.joblib")
    q50_model = joblib.load("models/arv_q50.joblib")
    q90_model = joblib.load("models/arv_q90.joblib")

    # LightGBM stores trained feature names in model.feature_name_
    if hasattr(q50_model, "feature_name_"):
        arv_feature_cols = list(q50_model.feature_name_)
    else:
        # Fallback: all numeric columns except obvious targets
        numeric_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in ["list_price", "sale_price_after_rehab", "flip_success"]
        ]
        arv_feature_cols = numeric_cols

    # Ensure props df has all features the model expects (build psf/year_built if needed)
    df = _ensure_feature_columns_for_arv(df, arv_feature_cols)

    X_arv = df[arv_feature_cols].fillna(0.0)

    arv_p10 = q10_model.predict(X_arv)
    arv_p50 = q50_model.predict(X_arv)
    arv_p90 = q90_model.predict(X_arv)

    # ----------------------------------------------------------------------
    # 2. Flip probability (p_success) from flip_logit_calibrated
    # ----------------------------------------------------------------------
    flip_model_path = Path("models/flip_logit_calibrated.joblib")
    if not flip_model_path.exists():
        raise SystemExit(
            "Flip model not found at models/flip_logit_calibrated.joblib. "
            "Train it with scripts/train_flip.py first."
        )

    flip_model = joblib.load(flip_model_path)

    if hasattr(flip_model, "feature_names_in_"):
        flip_feature_cols = list(flip_model.feature_names_in_)
        df = _ensure_feature_columns_for_flip(df, flip_feature_cols)
        X_flip = df[flip_feature_cols].fillna(0.0)
    else:
        # Fallback: reuse ARV feature set
        X_flip = X_arv

    p_succ = flip_model.predict_proba(X_flip)[:, 1]

    # ----------------------------------------------------------------------
    # 3. Rehab estimate (model-based, with override)
    # ----------------------------------------------------------------------
    if "list_price" not in df.columns:
        raise SystemExit(
            "ERROR: properties.parquet must contain 'list_price' for flip analysis."
        )

    buy = df["list_price"].astype(float).to_numpy()

    estimator = RehabEstimator()

    def _row_rehab(row: pd.Series) -> float:
        # Use manual rehab estimate if present and positive
        if "rehab_est" in row and pd.notna(row["rehab_est"]) and row["rehab_est"] > 0:
            return float(row["rehab_est"])

        sqft = float(row.get("sqft") or 0.0)
        year_built_val = row.get("year_built")
        year_built = int(year_built_val) if pd.notna(year_built_val) else None

        return estimator.estimate(sqft=sqft, year_built=year_built)

    rehab = df.apply(_row_rehab, axis=1).to_numpy()

    # ----------------------------------------------------------------------
    # 4. Holding / closing costs
    # ----------------------------------------------------------------------
    hold = _ensure_series(df, "HoldCost", 0.0).to_numpy()

    buy_rate = _ensure_series(df, "buy_closing_rate", FIN["buy_closing_rate"]).to_numpy()
    sell_rate = _ensure_series(df, "sell_closing_rate", FIN["sell_closing_rate"]).to_numpy()

    buy_fee = buy * buy_rate

    def profit(arv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sell_fee = arv * sell_rate
        total = buy + rehab + hold + buy_fee + sell_fee
        return arv - total, total

    p10, cost10 = profit(arv_p10)
    p50, cost50 = profit(arv_p50)
    p90, cost90 = profit(arv_p90)

    # ----------------------------------------------------------------------
    # 5. Maximum Allowable Offer (MAO) from p50 ARV
    # ----------------------------------------------------------------------
    mao = (
        arv_p50 * (1 - FIN["target_margin"] - sell_rate)
        - rehab
        - hold
        - buy_fee
    )

    # ----------------------------------------------------------------------
    # 6. Assemble output cheat sheet
    # ----------------------------------------------------------------------
    out = df.copy()
    out["arv_p10"] = arv_p10
    out["arv_p50"] = arv_p50
    out["arv_p90"] = arv_p90

    out["flip_p_good"] = p_succ
    out["rehab_model_est"] = rehab

    out["profit_p10"] = p10
    out["profit_p50"] = p50
    out["profit_p90"] = p90

    out["total_cost_p10"] = cost10
    out["total_cost_p50"] = cost50
    out["total_cost_p90"] = cost90

    out["mao_p50"] = mao

    out_path = Path("data/curated/properties_flips_scored.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
