# scripts/evaluate_engine.py

"""
Quick CLI for evaluating the Haven deal engine.

Usage (from repo root):

    python scripts/evaluate_engine.py
    python scripts/evaluate_engine.py --limit 500
    python scripts/evaluate_engine.py --dscr-target 1.25 --coc-target 0.10
    python scripts/evaluate_engine.py --as-json

This script:
  - Pulls recent deals from the deals table.
  - Computes portfolio-level DSCR / CoC metrics.
  - Breaks out behavior by label ("buy", "maybe", "pass").
  - Checks whether rank_score correlates with DSCR / CoC.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from haven.adapters.config import config
from haven.adapters.sql_repo import SqlDealRepository
from haven.services.engine_evaluation import evaluate_engine, format_report_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Haven deal engine using recent deals.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of most recent deals to include in the evaluation (default: 1000).",
    )
    parser.add_argument(
        "--dscr-target",
        type=float,
        default=None,
        help=(
            "Target DSCR threshold. Defaults to HAVEN_MIN_DSCR_GOOD "
            f"(currently {config.MIN_DSCR_GOOD})."
        ),
    )
    parser.add_argument(
        "--coc-target",
        type=float,
        default=None,
        help=(
            "Target cash-on-cash return as a decimal (e.g. 0.08 = 8%%). "
            "Defaults to 0.08."
        ),
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Emit a JSON blob instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo = SqlDealRepository(uri=config.DB_URI)

    report = evaluate_engine(
        repo=repo,
        limit=args.limit,
        dscr_target=args.dscr_target,
        coc_target=args.coc_target,
    )

    if args.as_json:
        # JSON-friendly form for logging / dashboards
        blob: dict[str, Any] = report.to_dict()
        print(json.dumps(blob, indent=2, sort_keys=True))
    else:
        text = format_report_text(report)
        print(text)


if __name__ == "__main__":
    main()
