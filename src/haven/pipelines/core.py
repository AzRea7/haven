# src/haven/pipelines/core.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Any

import json
import subprocess
import sys

from loguru import logger


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
MODELS_DIR = Path("models")


# ---------------------------
# 1. DATA REFRESH (stub for now)
# ---------------------------

def refresh_data(
    zipcodes: Sequence[str],
    max_price: float | None = None,
    workers: int = 2,
) -> None:
    """
    End-to-end data refresh for listings.

    For now this is a thin wrapper around the existing CLI script
    `scripts/ingest_properties_parallel.py`.

    This keeps imports simple (no missing haven.services.ingest),
    and lets the pipeline call the same thing you'd type by hand.
    """
    logger.info(
        "Starting data refresh via scripts/ingest_properties_parallel.py",
        zipcodes=list(zipcodes),
        max_price=max_price,
        workers=workers,
    )

    # Build CLI command: python scripts/ingest_properties_parallel.py --zip ... --workers ...
    cmd = [
        sys.executable,
        "scripts/ingest_properties_parallel.py",
        "--workers",
        str(workers),
    ]

    if max_price is not None:
        cmd += ["--max-price", str(max_price)]

    # In the script, --zip takes one or more values; we just append them.
    cmd += ["--zip", *zipcodes]

    logger.info("Running ingest command", cmd=" ".join(cmd))
    subprocess.run(cmd, check=True)

    logger.info("Data refresh completed", zipcodes=list(zipcodes))


# ---------------------------
# 2. MODEL TRAINING
# ---------------------------

def train_arv_models(force: bool = False) -> Path:
    """
    Train ARV LightGBM quantile models.

    This does **not** try to fetch sold data itself; it assumes you have already:
      - Fetched/normalized sold data (e.g. via fetch_hasdata_zillow + normalize_hasdata_sold)
      - Built data/processed/arv_training_from_sold.parquet via
        `python -m entrypoints.cli.build_arv_training_from_sold`

    It then calls the existing CLI:

        python -m entrypoints.cli.train_arv_quantiles \
            --training-path data/processed/arv_training_from_sold.parquet \
            --output-model-path models/arv_quantiles.joblib
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    training_path = PROCESSED_DIR / "arv_training_from_sold.parquet"
    model_path = MODELS_DIR / "arv_quantiles.joblib"

    if not training_path.exists():
        raise FileNotFoundError(
            f"{training_path} not found. Build it first with "
            "`python -m entrypoints.cli.build_arv_training_from_sold`."
        )

    logger.info(
        "Training ARV quantile models via CLI",
        training_path=str(training_path),
        model_path=str(model_path),
    )

    cmd = [
        sys.executable,
        "-m",
        "entrypoints.cli.train_arv_quantiles",
        "--training-path",
        str(training_path),
        "--out",
        str(model_path),
    ]
    subprocess.run(cmd, check=True)

    logger.info("ARV pipeline completed", model_path=str(model_path))
    return model_path


def train_rent_models() -> Path:
    """
    Train rent quantile model with neighborhood features.

    Thin wrapper around:

        python -m entrypoints.cli.retrain_rent_with_neighborhood \
            --output-model-path models/rent_quantiles_with_neighborhood.joblib

    Assumes data/processed/rent_training.parquet already exists.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rent_quantiles_with_neighborhood.joblib"

    logger.info(
        "Training rent model via CLI (with neighborhood features)",
        model_path=str(model_path),
    )

    cmd = [
        sys.executable,
        "-m",
        "entrypoints.cli.retrain_rent_with_neighborhood",
        "--out",
        str(model_path),
    ]
    subprocess.run(cmd, check=True)

    logger.info("Rent model retrain completed", model_path=str(model_path))
    return model_path


def train_flip_classifier() -> Path:
    """
    End-to-end flip classifier training via the existing CLI entrypoints:

    1. build_flip_training_from_csv -> data/processed/flip_training.parquet
    2. audit_flip_classifier -> models/flip_classifier_lgb.joblib
    """
    import subprocess
    import sys

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    flip_training_path = PROCESSED_DIR / "flip_training.parquet"
    flip_model_path = MODELS_DIR / "flip_classifier_lgb.joblib"

    # 1) Build flip training frame
    logger.info(
        "Building flip training frame via CLI",
        training_path=str(flip_training_path),
    )
    cmd_build = [
        sys.executable,
        "-m",
        "entrypoints.cli.build_flip_training_from_csv",
        "--out",  # <-- script expects --out, NOT --output-path
        str(flip_training_path),
    ]
    subprocess.run(cmd_build, check=True)

    # 2) Train / audit flip classifier
    logger.info(
        "Training flip classifier via CLI",
        model_path=str(flip_model_path),
    )
    cmd_train = [
        sys.executable,
        "-m",
        "entrypoints.cli.audit_flip_classifier",
        "--training-path",
        str(flip_training_path),
        "--out",  # keep arg names consistent with other CLIs
        str(flip_model_path),
    ]
    subprocess.run(cmd_train, check=True)

    logger.info("Flip classifier training completed", model_path=str(flip_model_path))
    return flip_model_path



def train_all_models(force_arv_fetch: bool = False) -> None:
    """
    High-level orchestrator: trains ARV, rent, and flip models.

    Note: `force_arv_fetch` is kept for API compatibility but currently
    unused because the ARV pipeline relies on pre-built training data.
    """
    logger.info("Starting full model training pipeline")

    train_arv_models(force=force_arv_fetch)
    train_rent_models()
    train_flip_classifier()

    logger.info("Full model training pipeline completed")


# ---------------------------
# 3. EVALUATION / TRUST
# ---------------------------

def eval_arv_models() -> Path:
    """
    Evaluate ARV quantile model and write per-ZIP MAE/MAPE.

    Reads:
      - data/processed/arv_training_from_sold.parquet
      - models/arv_quantiles.joblib

    Writes:
      - data/reports/arv_eval_by_zip.csv
    """
    import joblib
    import numpy as np
    import pandas as pd

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    training_path = PROCESSED_DIR / "arv_training_from_sold.parquet"
    model_path = MODELS_DIR / "arv_quantiles.joblib"
    report_path = REPORTS_DIR / "arv_eval_by_zip.csv"

    logger.info(
        "Evaluating ARV model",
        training_path=str(training_path),
        model_path=str(model_path),
        report_path=str(report_path),
    )

    if not training_path.exists():
        raise FileNotFoundError(
            f"ARV training data not found at {training_path}. "
            "Run train_arv_models() first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"ARV model bundle not found at {model_path}. "
            "Run train_arv_models() first."
        )

    df = pd.read_parquet(training_path)
    df = df.copy().dropna(subset=["target_arv", "zipcode"])

    bundle: Dict[str, Any] = joblib.load(model_path)

    models = bundle.get("models") or bundle.get("models_by_quantile")
    if models is None:
        raise KeyError("ARV bundle missing 'models' key")

    # We evaluate using the median (0.5 quantile) model.
    median_model = (
        models.get("p50")
        or models.get("0.5")
        or models.get(0.5)
    )
    if median_model is None:
        raise KeyError("ARV bundle missing median quantile model (0.5/p50)")


    feature_names = bundle.get("feature_names")
    if not feature_names:
        raise KeyError("ARV bundle missing 'feature_names' key")

    # Let LightGBM handle NaNs; only require target/zipcode to be present
    df_features = df[feature_names].copy()

    if df_features.empty:
        raise RuntimeError("No feature rows available for ARV eval.")

    X = df_features.to_numpy(dtype=float)
    y_true = df["target_arv"].to_numpy(dtype=float)


    y_pred = median_model.predict(X)

    abs_err = (y_pred - y_true).__abs__()
    pct_err = abs_err / np.clip(y_true, 1.0, None)

    df_eval = df[["zipcode"]].copy()
    df_eval["abs_err"] = abs_err
    df_eval["pct_err"] = pct_err

    grouped = (
        df_eval.groupby("zipcode")
        .agg(
            count=("abs_err", "size"),
            mae=("abs_err", "mean"),
            mape=("pct_err", "mean"),
        )
        .reset_index()
        .rename(columns={"zipcode": "zip"})
    )

    overall = {
        "zip": "ALL",
        "count": int(df_eval.shape[0]),
        "mae": float(abs_err.mean()),
        "mape": float(pct_err.mean()),
    }

    grouped = pd.concat([grouped, pd.DataFrame([overall])], ignore_index=True)

    grouped.to_csv(report_path, index=False)
    logger.info("ARV evaluation report written", report_path=str(report_path))
    return report_path


def eval_rent_models() -> Path:
    """
    Evaluate rent model and write per-ZIP MAE/MAPE.

    Reads:
      - data/processed/rent_training.parquet
      - models/rent_quantiles_with_neighborhood.joblib

    Writes:
      - data/reports/rent_eval_by_zip.csv
    """
    import joblib
    import numpy as np
    import pandas as pd

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    training_path = PROCESSED_DIR / "rent_training.parquet"
    model_path = MODELS_DIR / "rent_quantiles_with_neighborhood.joblib"
    report_path = REPORTS_DIR / "rent_eval_by_zip.csv"

    logger.info(
        "Evaluating rent model",
        training_path=str(training_path),
        model_path=str(model_path),
        report_path=str(report_path),
    )

    if not training_path.exists():
        raise FileNotFoundError(
            f"Rent training data not found at {training_path}. "
            "You need to build rent_training.parquet from leases / market data."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Rent model bundle not found at {model_path}. "
            "Run train_rent_models() first."
        )

    df = pd.read_parquet(training_path)
    df = df.copy().dropna(subset=["target_rent", "zipcode"])

    bundle: Dict[str, Any] = joblib.load(model_path)

    # Models dict can be keyed by strings ("p50", "0.5") or floats (0.5)
    models = bundle.get("models") or bundle.get("models_by_quantile")
    if models is None:
        raise KeyError("Rent bundle missing 'models' key")

    median_model = (
        models.get("p50")
        or models.get("0.5")
        or models.get(0.5)
    )

    if median_model is None:
        if isinstance(models, dict) and len(models) > 0:
            sorted_keys = sorted(models.keys())
            median_key = sorted_keys[len(sorted_keys) // 2]
            median_model = models[median_key]
        else:
            raise KeyError(
                "Rent bundle missing median quantile model "
                "('p50', '0.5', or 0.5) and could not infer from keys."
            )

    feature_names = bundle.get("feature_names")
    if not feature_names:
        raise KeyError("Rent bundle missing 'feature_names' key")

    df = pd.read_parquet(training_path)

    # 🔧 NEW: ensure all required feature columns exist
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        logger.warning(
            "rent_eval_missing_features_adding_defaults",
            extra={"context": {"missing": missing}},
        )
        # For now, just fill with 0.0 so evaluation can run
        for c in missing:
            df[c] = 0.0

    df_features = df[feature_names].copy()

    if df_features.empty:
        raise RuntimeError("No feature rows available for rent eval.")


    X = df_features.to_numpy(dtype=float)
    y_true = df["target_rent"].to_numpy(dtype=float)

    y_pred = median_model.predict(X)

    abs_err = (y_pred - y_true).__abs__()
    pct_err = abs_err / np.clip(y_true, 1.0, None)

    df_eval = df[["zipcode"]].copy()
    df_eval["abs_err"] = abs_err
    df_eval["pct_err"] = pct_err

    grouped = (
        df_eval.groupby("zipcode")
        .agg(
            count=("abs_err", "size"),
            mae=("abs_err", "mean"),
            mape=("pct_err", "mean"),
        )
        .reset_index()
        .rename(columns={"zipcode": "zip"})
    )

    overall = {
        "zip": "ALL",
        "count": int(df_eval.shape[0]),
        "mae": float(abs_err.mean()),
        "mape": float(pct_err.mean()),
    }
    grouped = pd.concat([grouped, pd.DataFrame([overall])], ignore_index=True)

    grouped.to_csv(report_path, index=False)
    logger.info("Rent evaluation report written", report_path=str(report_path))
    return report_path


def eval_flip_classifier() -> Path:
    """
    Evaluate flip classifier and write global metrics to JSON.

    Reads:
      - data/processed/flip_training.parquet
      - models/flip_classifier_lgb.joblib

    Writes:
      - data/reports/flip_eval.json
    """
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    training_path = PROCESSED_DIR / "flip_training.parquet"
    model_path = MODELS_DIR / "flip_classifier_lgb.joblib"
    report_path = REPORTS_DIR / "flip_eval.json"

    logger.info(
        "Evaluating flip classifier",
        training_path=str(training_path),
        model_path=str(model_path),
        report_path=str(report_path),
    )

    if not training_path.exists():
        raise FileNotFoundError(
            f"Flip training data not found at {training_path}. "
            "Run train_flip_classifier() first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Flip classifier model not found at {model_path}. "
            "Run train_flip_classifier() first."
        )

    df = pd.read_parquet(training_path).copy()

    if "is_good_flip" not in df.columns:
        raise KeyError("flip_training.parquet must contain 'is_good_flip' label column.")

    y_true = df["is_good_flip"].astype(int).to_numpy()

    drop_cols = {
        "is_good_flip",
        "actual_roi",
        "deal_id",
        "address",
        "city",
        "state",
        "zipcode",
        "notes",
    }
    feature_cols = [
        c
        for c in df.columns
        if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found for flip classifier eval.")

    X = df[feature_cols].to_numpy(dtype=float)

        # Load classifier; newer training code saves a bundle dict
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        model = bundle.get("model")
        if model is None:
            raise KeyError("Flip classifier bundle missing 'model' key")
        # Optionally sanity check feature names, but don't hard-fail:
        bundle_features = bundle.get("feature_names")
        if bundle_features:
            # You *could* assert equality or log mismatch here; for now just log.
            if list(bundle_features) != feature_cols:
                logger.warning(
                    "flip_eval_feature_mismatch",
                    extra={
                        "context": {
                            "bundle_feature_names": list(bundle_features),
                            "eval_feature_cols": feature_cols,
                        }
                    },
                )
    else:
        # Backward-compat: older models may be saved as bare LGBMClassifier
        model = bundle

    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    metrics: Dict[str, Any] = {}
    metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    metrics["precision"] = float(precision_score(y_true, y_pred))
    metrics["recall"] = float(recall_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred))
    metrics["brier"] = float(brier_score_loss(y_true, proba))
    # ... rest of your code unchanged ...

    metrics["n_samples"] = int(len(y_true))

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        metrics["roc_auc"] = None

    metrics["pr_auc"] = float(average_precision_score(y_true, proba))
    metrics["brier"] = float(brier_score_loss(y_true, proba))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    df_eval = df.copy()
    df_eval["y_true"] = y_true
    df_eval["p_good"] = proba
    df_eval = df_eval.sort_values("p_good", ascending=False).reset_index(drop=True)

    precision_at_k: Dict[str, float] = {}
    for k in (10, 20, 50):
        if len(df_eval) >= k:
            top_k = df_eval.iloc[:k]
            precision_at_k[str(k)] = float(top_k["y_true"].mean())
        else:
            precision_at_k[str(k)] = float(df_eval["y_true"].mean())
    metrics["precision_at_k"] = precision_at_k

    report_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Flip classifier evaluation report written", report_path=str(report_path))
    return report_path


def eval_all_models() -> None:
    """
    High-level evaluation pipeline (ARV, rent, flip).
    """
    logger.info("Starting model evaluation pipeline")

    arv_report = eval_arv_models()
    rent_report = eval_rent_models()
    flip_report = eval_flip_classifier()

    logger.info(
        "Model evaluation pipeline completed",
        arv_report=str(arv_report),
        rent_report=str(rent_report),
        flip_report=str(flip_report),
    )


def collect_metrics_snapshot() -> Path:
    """
    Aggregate key metrics from the various reports into a single JSON snapshot.

    Reads:
      - data/reports/arv_eval_by_zip.csv
      - data/reports/rent_eval_by_zip.csv
      - data/reports/flip_eval.json

    Writes:
      - data/reports/metrics_snapshot.json
    """
    import pandas as pd

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = REPORTS_DIR / "metrics_snapshot.json"

    snapshot: Dict[str, Any] = {}

    arv_path = REPORTS_DIR / "arv_eval_by_zip.csv"
    rent_path = REPORTS_DIR / "rent_eval_by_zip.csv"
    flip_path = REPORTS_DIR / "flip_eval.json"

    if arv_path.exists():
        df_arv = pd.read_csv(arv_path)
        all_row = df_arv.loc[df_arv["zip"] == "ALL"].to_dict(orient="records")
        snapshot["arv"] = all_row[0] if all_row else {}
    else:
        snapshot["arv"] = {}

    if rent_path.exists():
        df_rent = pd.read_csv(rent_path)
        all_row = df_rent.loc[df_rent["zip"] == "ALL"].to_dict(orient="records")
        snapshot["rent"] = all_row[0] if all_row else {}
    else:
        snapshot["rent"] = {}

    if flip_path.exists():
        snapshot["flip"] = json.loads(flip_path.read_text())
    else:
        snapshot["flip"] = {}

    snapshot_path.write_text(json.dumps(snapshot, indent=2))
    logger.info("Metrics snapshot written", snapshot_path=str(snapshot_path))
    return snapshot_path


# ---------------------------
# 4. BACKTESTING
# ---------------------------

def backtest_engine(
    backtest_csv: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Run engine backtest over historical deals.

    Assumes:
      - data/raw/historical_deals.csv exists (or backtest_csv is provided).
      - haven.services.backtest.run_backtest(backtest_csv, output_path) is implemented.
    """
    from haven.services.backtest import run_backtest

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    backtest_csv = backtest_csv or RAW_DIR / "historical_deals.csv"
    output_path = output_path or REPORTS_DIR / "backtest_summary.json"

    logger.info(
        "Running backtest engine",
        backtest_csv=str(backtest_csv),
        output_path=str(output_path),
    )

    run_backtest(backtest_csv=backtest_csv, output_path=output_path)

    logger.info("Backtest completed", output_path=str(output_path))
    return output_path


# ---------------------------
# 5. FULL PIPELINE
# ---------------------------

def full_refresh(
    zipcodes: Sequence[str],
    max_price: float | None = None,
    workers: int = 2,
    force_arv_fetch: bool = False,
) -> None:
    """
    One-command full pipeline (when you want it):

    1. Refresh listings for ZIPs.
    2. Train all models.
    3. Evaluate models and write reports.
    """
    logger.info(
        "Starting FULL PIPELINE",
        zipcodes=list(zipcodes),
        max_price=max_price,
        workers=workers,
        force_arv_fetch=force_arv_fetch,
    )

    refresh_data(zipcodes=zipcodes, max_price=max_price, workers=workers)
    train_all_models(force_arv_fetch=force_arv_fetch)
    eval_all_models()

    logger.info("FULL PIPELINE completed")
