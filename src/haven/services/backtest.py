# haven/services/backtest.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from haven.services.deal_analyzer import analyze_deal_with_defaults


def _row_to_payload(row: pd.Series) -> Dict[str, Any]:
    """
    Convert a backtest CSV row into an analyze_deal payload.

    Adjust this mapping as needed to match your engine's expectations.
    """
    payload: Dict[str, Any] = {
        "address": row.get("address", ""),
        "city": row.get("city", ""),
        "state": row.get("state", ""),
        "zipcode": str(row.get("zipcode", "")).strip(),
        "list_price": float(row.get("list_price", 0.0)),
        "sqft": float(row.get("sqft", 0.0)) if not pd.isna(row.get("sqft")) else None,
        "bedrooms": float(row.get("bedrooms", 0.0)) if not pd.isna(row.get("bedrooms")) else None,
        "bathrooms": float(row.get("bathrooms", 0.0)) if not pd.isna(row.get("bathrooms")) else None,
        "property_type": row.get("property_type", "single_family"),
    }

    # Optional stuff
    if "year_built" in row:
        payload["year_built"] = (
            int(row["year_built"]) if not pd.isna(row["year_built"]) else None
        )
    if "lat" in row and "lon" in row:
        payload["lat"] = float(row["lat"]) if not pd.isna(row["lat"]) else None
        payload["lon"] = float(row["lon"]) if not pd.isna(row["lon"]) else None

    return payload


def run_backtest(backtest_csv: Path, output_path: Path) -> None:
    """
    Run the engine over historical deals and compare predictions to realized ROI.

    backtest_csv must contain at least:
      - columns needed by analyze_deal_with_defaults
      - 'actual_roi'  (float, e.g., 0.20 for 20%)

    Writes:
      - data/reports/backtest_summary.json
      - data/reports/backtest_details.csv
    """
    import json

    logger.info(
        "Starting engine backtest",
        backtest_csv=str(backtest_csv),
        output_path=str(output_path),
    )

    if not backtest_csv.exists():
        raise FileNotFoundError(
            f"Backtest CSV not found at {backtest_csv}. "
            "Create data/raw/historical_deals.csv first."
        )

    df = pd.read_csv(backtest_csv)
    if "actual_roi" not in df.columns:
        raise KeyError(
            "Backtest CSV must contain an 'actual_roi' column with realized ROI."
        )

    df = df.copy()
    df["actual_roi"] = df["actual_roi"].astype(float)

    engine_labels: List[str] = []
    engine_rank_scores: List[float] = []
    engine_coc: List[float] = []

    for idx, row in df.iterrows():
        payload = _row_to_payload(row)
        try:
            res = analyze_deal_with_defaults(payload)
        except Exception as exc:  # log and continue; bad rows shouldn't kill backtest
            logger.exception("Error analyzing row in backtest", idx=idx, exc=exc)
            engine_labels.append("error")
            engine_rank_scores.append(float("nan"))
            engine_coc.append(float("nan"))
            continue

        # Label / suggestion
        label = res.get("label")
        if label is None:
            label = res.get("score", {}).get("suggestion", "unknown")

        # Rank score
        rank_score = res.get("score", {}).get("rank_score", float("nan"))

        # CoC as a proxy for predicted ROI
        finance = res.get("finance", {})
        coc = finance.get("cash_on_cash_return", float("nan"))

        engine_labels.append(label)
        engine_rank_scores.append(rank_score)
        engine_coc.append(coc)

    df["engine_label"] = engine_labels
    df["engine_rank_score"] = engine_rank_scores
    df["engine_cash_on_cash_return"] = engine_coc

    # Basic ROI comparisons
    overall_mean_roi = float(df["actual_roi"].mean())

    buy_mask = df["engine_label"] == "buy"
    maybe_mask = df["engine_label"].str.contains("maybe", case=False, na=False)
    pass_mask = df["engine_label"] == "pass"

    def safe_mean(mask) -> float | None:
        filtered = df.loc[mask, "actual_roi"]
        return float(filtered.mean()) if len(filtered) > 0 else None

    mean_roi_by_label = {
        "buy": safe_mean(buy_mask),
        "maybe": safe_mean(maybe_mask),
        "pass": safe_mean(pass_mask),
        "error_or_other": safe_mean(
            ~(buy_mask | maybe_mask | pass_mask)
        ),
    }

    # Top-K by rank_score (best deals)
    df_sorted = df.dropna(subset=["engine_rank_score"]).sort_values(
        "engine_rank_score", ascending=False
    )
    top_k_stats: Dict[str, Any] = {}
    for k in (5, 10, 20, 50):
        if len(df_sorted) == 0:
            top_k_stats[str(k)] = {"n": 0, "mean_actual_roi": None}
            continue
        k_eff = min(k, len(df_sorted))
        subset = df_sorted.head(k_eff)
        top_k_stats[str(k)] = {
            "n": int(k_eff),
            "mean_actual_roi": float(subset["actual_roi"].mean()),
        }

    summary = {
        "n_deals": int(len(df)),
        "overall_mean_actual_roi": overall_mean_roi,
        "mean_actual_roi_by_label": mean_roi_by_label,
        "top_k_by_rank_score": top_k_stats,
    }

    # Write summary JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    # Also write enriched per-deal CSV next to it
    details_path = output_path.with_name("backtest_details.csv")
    df.to_csv(details_path, index=False)

    logger.info(
        "Backtest completed",
        summary_path=str(output_path),
        details_path=str(details_path),
    )
