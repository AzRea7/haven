"""
Parallel scoring of all candidate properties.

This script:
  1. Pulls properties from the SQL repository (or filters by ZIP / price).
  2. Uses haven.services.deal_analyzer.analyze_deal_with_defaults to compute
     DSCR, cash-on-cash, and rank_score per property.
  3. Runs the scoring in parallel across CPU cores with joblib.
  4. Saves per-property metrics to data/processed/scored_properties.parquet.

This is the "parallel map" part of the DSCR / CoC computation. The reduction
step (portfolio-level stats) can be done with haven.domain.metrics.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from haven.adapters.sql_repo import SqlPropertyRepository
from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator
from haven.services.deal_analyzer import analyze_deal_with_defaults
from haven.adapters.config import config  # for DB URL
from haven.domain.metrics import summarize_portfolio  # optional nice summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel DSCR / CoC scoring for all properties."
    )
    parser.add_argument(
        "--zip",
        dest="zipcodes",
        nargs="*",
        default=None,
        help="Optional list of ZIP codes to restrict the scoring set.",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Optional maximum list price filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of properties to score.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for scoring (joblib n_jobs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/scored_properties.parquet",
        help="Path to save scored properties (parquet).",
    )
    return parser.parse_args()


def _score_single_property(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function run in parallel.

    Each call:
      - builds a payload for analyze_deal_with_defaults
      - computes finance + score
      - returns a flat dict with DSCR, CoC, rank_score, label, etc.
    """
    # Each worker gets its own estimator instance to avoid cross-process issues.
    rent_estimator = LightGBMRentEstimator()

    payload = {
        "address": record.get("address"),
        "city": record.get("city"),
        "state": record.get("state"),
        "zipcode": record.get("zipcode"),
        "list_price": float(record.get("list_price"))
        if record.get("list_price") is not None
        else None,
        "sqft": record.get("sqft"),
        "bedrooms": record.get("bedrooms"),
        "bathrooms": record.get("bathrooms"),
        "property_type": record.get("property_type"),
        # Strategy is currently fixed; you can parameterize this later.
        "strategy": "hold",
    }

    # Ask the core Haven logic to analyze the deal.
    result = analyze_deal_with_defaults(
        payload,
        rent_estimator=rent_estimator,
        repo=None,  # we don't need repo inside the analyzer for this call
    )

    finance = result.get("finance", {})
    score = result.get("score", {})

    return {
        "external_id": record.get("external_id"),
        "source": record.get("source"),
        "address": record.get("address"),
        "city": record.get("city"),
        "state": record.get("state"),
        "zipcode": record.get("zipcode"),
        "list_price": record.get("list_price"),
        "sqft": record.get("sqft"),
        "bedrooms": record.get("bedrooms"),
        "bathrooms": record.get("bathrooms"),
        "property_type": record.get("property_type"),
        # Finance metrics
        "dscr": finance.get("dscr"),
        "cash_on_cash_return": finance.get("cash_on_cash_return"),
        "net_operating_income": finance.get("net_operating_income"),
        "annual_debt_service": finance.get("annual_debt_service"),
        # Score outputs
        "rank_score": score.get("rank_score"),
        "label": score.get("label"),
        "reason": score.get("reason"),
    }


def _load_properties(
    zipcodes: Optional[List[str]],
    max_price: Optional[float],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Pull candidate properties from the SQL repository.

    This is intentionally simple and can be evolved later (e.g., to read from
    data/curated parquet instead of the DB).
    """
    repo = SqlPropertyRepository(config.database_url)

    if zipcodes:
        all_records: List[Dict[str, Any]] = []
        for z in zipcodes:
            records = repo.search(
                zipcode=z,
                max_price=max_price,
                limit=limit,
            )
            all_records.extend(records)
    else:
        all_records = repo.search(
            zipcode=None,
            max_price=max_price,
            limit=limit,
        )

    return all_records


def main() -> None:
    args = parse_args()

    records = _load_properties(
        zipcodes=args.zipcodes,
        max_price=args.max_price,
        limit=args.limit,
    )

    if not records:
        print("No properties found to score. Did you run ingest + build_features?")
        return

    print(f"Scoring {len(records)} properties in parallel...")

    scored: List[Dict[str, Any]] = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(_score_single_property)(rec) for rec in records
    )

    df = pd.DataFrame(scored)

    # Save to your data pipeline layout: data/processed/...
    output_path = args.output
    df.to_parquet(output_path, index=False)
    print(f"Saved scored properties to {output_path}")

    # Optional: run a portfolio-level reduction to show "reduction" metrics
    dscr = df["dscr"].to_numpy(dtype=float)
    coc = df["cash_on_cash_return"].to_numpy(dtype=float)

    summary = summarize_portfolio(dscr, coc)
    print("Portfolio metrics:")
    print(f"  n_properties: {summary.n_properties}")
    print(f"  mean DSCR   : {summary.mean_dscr:.3f}")
    print(f"  p5 / p50 / p95 DSCR: {summary.p5_dscr:.3f}, "
          f"{summary.p50_dscr:.3f}, {summary.p95_dscr:.3f}")
    print(f"  mean CoC    : {summary.mean_coc:.3f}")
    print(f"  p5 / p50 / p95 CoC : {summary.p5_coc:.3f}, "
          f"{summary.p50_coc:.3f}, {summary.p95_coc:.3f}")


if __name__ == "__main__":
    main()
