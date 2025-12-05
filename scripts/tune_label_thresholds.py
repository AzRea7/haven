# scripts/tune_label_thresholds.py

"""
Explore different DSCR / CoC thresholds for the 'buy' label.

This does NOT change your engine. It helps you pick realistic,
business-aligned thresholds before you go edit scoring logic.

Usage (from repo root):

    python scripts/tune_label_thresholds.py
    python scripts/tune_label_thresholds.py --limit 2000
"""

from __future__ import annotations

import argparse
from typing import List, Sequence

import numpy as np

from haven.adapters.config import config
from haven.adapters.sql_repo import SqlDealRepository, DealRow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep DSCR/CoC thresholds for 'buy' label tuning.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of most recent deals to include (default: 1000).",
    )
    return p.parse_args()


def load_deal_metrics(limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load recent deals and extract DSCR, CoC, rank_score arrays.
    """
    repo = SqlDealRepository(uri=config.DB_URI)
    rows: Sequence[DealRow] = repo.list_recent(limit=limit)

    dscr_vals: List[float] = []
    coc_vals: List[float] = []
    rank_vals: List[float] = []

    for row in rows:
        result = row.result or {}
        finance = result.get("finance") or {}
        score = result.get("score") or {}

        try:
            dscr = float(finance.get("dscr"))
            coc = float(finance.get("cash_on_cash_return"))
            rank = float(score.get("rank_score", 0.0))
        except (TypeError, ValueError):
            continue

        dscr_vals.append(dscr)
        coc_vals.append(coc)
        rank_vals.append(rank)

    if not dscr_vals:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    return (
        np.asarray(dscr_vals, dtype=float),
        np.asarray(coc_vals, dtype=float),
        np.asarray(rank_vals, dtype=float),
    )


def main() -> None:
    args = parse_args()

    dscr, coc, rank = load_deal_metrics(limit=args.limit)

    n_total = dscr.size
    if n_total == 0:
        print("No deals available; run some analyses first.")
        return

    print(f"Loaded {n_total} deals for threshold tuning.\n")

    # Candidate thresholds to explore.
    # You can modify these as you like.
    dscr_thresholds = [1.10, 1.20, 1.25, 1.30, 1.40, 1.50]
    coc_thresholds = [0.04, 0.06, 0.08, 0.10, 0.12]

    print("Threshold grid (DSCR ≥ x, CoC ≥ y):\n")
    print(
        f"{'DSCR':>6}  {'CoC%':>6}  "
        f"{'#buys':>7}  {'%universe':>9}  "
        f"{'mean DSCR':>9}  {'mean CoC%':>10}  {'mean rank':>10}"
    )
    print("-" * 70)

    for d_thr in dscr_thresholds:
        for c_thr in coc_thresholds:
            mask = (dscr >= d_thr) & (coc >= c_thr)
            n_buy = int(mask.sum())
            if n_buy == 0:
                # Still print, but keep stats blank-ish
                pct = 0.0
                mean_d = float("nan")
                mean_c = float("nan")
                mean_r = float("nan")
            else:
                pct = 100.0 * n_buy / n_total
                mean_d = float(dscr[mask].mean())
                mean_c = float(coc[mask].mean() * 100.0)
                mean_r = float(rank[mask].mean())

            print(
                f"{d_thr:6.2f}  {c_thr*100:6.1f}  "
                f"{n_buy:7d}  {pct:9.2f}  "
                f"{mean_d:9.2f}  {mean_c:10.1f}  {mean_r:10.2f}"
            )


if __name__ == "__main__":
    main()
