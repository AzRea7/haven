from __future__ import annotations

from typing import List, Optional

import typer

from haven.pipelines.core import (
    refresh_data,
    train_all_models,
    eval_all_models,
    backtest_engine,
    full_refresh,
    collect_metrics_snapshot,
)

app = typer.Typer(help="Haven automation pipeline (data, models, eval, backtest).")


@app.command("refresh-data")
def refresh_data_cmd(
    zip: List[str] = typer.Option(..., "--zip", help="ZIP code(s) to refresh"),
    max_price: Optional[float] = typer.Option(
        None, help="Optional max list price for ingest"
    ),
    workers: int = typer.Option(2, help="Number of ingest workers"),
) -> None:
    """
    Refresh listing data for one or more ZIP codes.
    """
    refresh_data(zipcodes=zip, max_price=max_price, workers=workers)


@app.command("train-models")
def train_models(
    force_arv_fetch: bool = typer.Option(
        False,
        "--force-arv-fetch",
        help="Force re-fetch of sold comps from ATTOM even if cached.",
    )
) -> None:
    """
    Train ARV, rent, and flip models.
    """
    train_all_models(force_arv_fetch=force_arv_fetch)


@app.command("eval-models")
def eval_models() -> None:
    """
    Evaluate ARV, rent, and flip models and write reports to data/reports.
    """
    eval_all_models()


@app.command("metrics-snapshot")
def metrics_snapshot() -> None:
    """
    Aggregate ARV, rent, and flip metrics into data/reports/metrics_snapshot.json.
    """
    collect_metrics_snapshot()


@app.command()
def backtest(
    backtest_csv: Optional[str] = typer.Option(
        None,
        help="Path to historical deals CSV (default: data/raw/historical_deals.csv)",
    ),
    output: Optional[str] = typer.Option(
        None,
        help="Path to write backtest summary (default: data/reports/backtest_summary.json)",
    ),
) -> None:
    """
    Run engine backtest over historical deals.
    """
    from pathlib import Path

    backtest_engine(
        backtest_csv=Path(backtest_csv) if backtest_csv else None,
        output_path=Path(output) if output else None,
    )


@app.command("full-refresh")
def full_refresh_cmd(
    zip: List[str] = typer.Option(..., "--zip", help="ZIP code(s) to refresh"),
    max_price: Optional[float] = typer.Option(
        None, help="Optional max list price for ingest"
    ),
    workers: int = typer.Option(2, help="Number of ingest workers"),
    force_arv_fetch: bool = typer.Option(
        False,
        "--force-arv-fetch",
        help="Force re-fetch of sold comps from ATTOM even if cached.",
    ),
) -> None:
    """
    End-to-end pipeline:

    - refresh-data
    - train-models
    - eval-models
    """
    full_refresh(
        zipcodes=zip,
        max_price=max_price,
        workers=workers,
        force_arv_fetch=force_arv_fetch,
    )


if __name__ == "__main__":
    app()
