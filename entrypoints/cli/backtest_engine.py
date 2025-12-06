# entrypoints/cli/backtest_engine.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from haven.adapters.logging_utils import get_logger
from haven.services.deal_analyzer import analyze_deal_with_defaults

logger = get_logger(__name__)


def load_historical(path: Path) -> pd.DataFrame:
    """
    Load historical deals from CSV or Parquet.

    EXPECTED MINIMUM COLUMNS:

      address
      city
      state
      zipcode
      list_price
      sqft
      bedrooms
      bathrooms
      property_type
      year_built          (optional)
      days_on_market      (optional)
      strategy            ('flip' or 'hold', optional but useful)
      realized_roi        (for flips, numeric, e.g. 0.22)
      realized_rent       (for holds, numeric monthly)

    You can add extra columns; they will be carried through.
    """
    if not path.exists():
        raise SystemExit(f"Historical deals file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    required_base = ["address", "city", "state", "zipcode", "list_price"]
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise SystemExit(f"Historical deals missing required columns: {missing}")

    return df.copy()


def build_payload(row: pd.Series) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "address": row.get("address"),
        "city": row.get("city"),
        "state": row.get("state"),
        "zipcode": str(row.get("zipcode")),
        "list_price": float(row.get("list_price") or 0.0),
        "sqft": float(row.get("sqft") or 0.0),
        "bedrooms": float(row.get("bedrooms") or 0.0),
        "bathrooms": float(row.get("bathrooms") or 0.0),
        "property_type": row.get("property_type", "single_family"),
    }

    year_built = row.get("year_built")
    if pd.notnull(year_built):
        payload["year_built"] = float(year_built)

    dom = row.get("days_on_market")
    if pd.notnull(dom):
        payload["days_on_market"] = float(dom)

    strategy = row.get("strategy")
    if isinstance(strategy, str) and strategy:
        payload["strategy"] = strategy

    return payload


def run_backtest(
    df: pd.DataFrame,
    top_n_per_zip: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run engine on each historical row and compute summary metrics.
    """
    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        payload = build_payload(row)

        try:
            result = analyze_deal_with_defaults(payload)
        except Exception as exc:
            logger.exception(
                "backtest_analyze_failed",
                extra={"row_index": idx, "error": str(exc)},
            )
            continue

        score = result.get("score", {}) or {}
        pricing = result.get("pricing", {}) or {}
        finance = result.get("finance", {}) or {}

        rec: Dict[str, Any] = {
            "address": payload["address"],
            "city": payload["city"],
            "state": payload["state"],
            "zipcode": payload["zipcode"],
            "list_price": payload["list_price"],
            "strategy": row.get("strategy"),
            "realized_roi": row.get("realized_roi"),
            "realized_rent": row.get("realized_rent"),
            "engine_label": score.get("label"),
            "engine_suggestion": score.get("suggestion"),
            "engine_rank_score": score.get("rank_score"),
            "flip_p_good": result.get("flip_p_good"),
            "profit_p50": pricing.get("profit_p50"),
            "profit_p10": pricing.get("profit_p10"),
            "dscr": finance.get("dscr"),
            "cash_on_cash_return": finance.get("cash_on_cash_return"),
        }

        records.append(rec)

    if not records:
        raise SystemExit("Backtest produced no records; check logs for errors.")

    out_df = pd.DataFrame(records)

    # ---- Summary metrics ----
    summaries: List[Dict[str, Any]] = []

    # 1) Flip ROI by engine label
    flips = out_df[
        out_df["realized_roi"].notnull()
        & np.isfinite(out_df["realized_roi"])
    ]
    if not flips.empty:
        for label in ["buy", "maybe", "pass"]:
            g = flips[flips["engine_label"] == label]
            if g.empty:
                continue
            summaries.append(
                {
                    "group": f"flip_label_{label}",
                    "n": len(g),
                    "mean_roi": float(g["realized_roi"].mean()),
                    "median_roi": float(g["realized_roi"].median()),
                }
            )

        # baseline: all flips
        summaries.append(
            {
                "group": "flip_baseline_all",
                "n": len(flips),
                "mean_roi": float(flips["realized_roi"].mean()),
                "median_roi": float(flips["realized_roi"].median()),
            }
        )

    # 2) Top-N per ZIP flips vs baseline
    if not flips.empty:
        for z, g in flips.groupby("zipcode"):
            g = g.sort_values("engine_rank_score", ascending=False)
            top = g.head(top_n_per_zip)
            baseline_mean = float(g["realized_roi"].mean())
            top_mean = float(top["realized_roi"].mean())
            summaries.append(
                {
                    "group": f"flip_top_{top_n_per_zip}_zip_{z}",
                    "n": len(top),
                    "mean_roi": top_mean,
                    "baseline_mean_roi": baseline_mean,
                }
            )

    summary_df = pd.DataFrame(summaries)

    return out_df, summary_df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backtest Haven engine against historical deals."
    )
    ap.add_argument(
        "--historical-path",
        type=Path,
        default=Path("data/raw/historical_deals.parquet"),
        help="CSV or Parquet with historical deals and realized outcomes.",
    )
    ap.add_argument(
        "--out-detailed",
        type=Path,
        default=Path("data/reports/backtest_detailed.parquet"),
        help="Detailed backtest output.",
    )
    ap.add_argument(
        "--out-summary",
        type=Path,
        default=Path("data/reports/backtest_summary.parquet"),
        help="Backtest summary metrics output.",
    )
    ap.add_argument(
        "--top-n-per-zip",
        type=int,
        default=5,
        help="Number of top-ranked deals per ZIP for ROI comparison.",
    )
    args = ap.parse_args()

    df_hist = load_historical(args.historical_path)

    detailed, summary = run_backtest(
        df_hist,
        top_n_per_zip=args.top_n_per_zip,
    )

    args.out_detailed.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_parquet(args.out_detailed, index=False)
    summary.to_parquet(args.out_summary, index=False)

    print(f"Wrote detailed backtest to {args.out_detailed} (rows={len(detailed)})")
    print("Backtest summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
