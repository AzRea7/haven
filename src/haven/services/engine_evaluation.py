# src/haven/services/engine_evaluation.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from haven.adapters.config import config
from haven.adapters.sql_repo import SqlDealRepository, DealRow
from haven.domain.metrics import PortfolioMetrics, summarize_portfolio


@dataclass
class LabelBucketMetrics:
    """
    Aggregated metrics for a slice of deals that share the same label
    (e.g. "buy", "maybe", "pass").

    This is where you can see whether your scoring logic is "honest":
    - Are 'buy' deals actually stronger on DSCR / CoC than 'maybe' or 'pass'?
    - Are they consistently meeting your target thresholds?
    """

    label: str
    n_deals: int
    portfolio: PortfolioMetrics

    avg_rank_score: float
    median_rank_score: float

    # Percent of deals in this bucket that meet business targets
    pct_meeting_dscr_target: float
    pct_meeting_coc_target: float


@dataclass
class EngineEvaluationReport:
    """
    Top-level evaluation artifact for the scoring engine.

    This is intentionally JSON-friendly (see .to_dict()) so you can:
    - Log it to stdout for observability
    - Store it as a structured record
    - Feed it into dashboards later (Grafana/Metabase/etc.)
    """

    total_deals: int
    dscr_target: float
    coc_target: float

    overall_portfolio: PortfolioMetrics
    buckets: List[LabelBucketMetrics]

    # Simple technical sanity checks: does rank_score correlate with fundamentals?
    corr_rank_dscr: Optional[float]
    corr_rank_coc: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_deals": self.total_deals,
            "dscr_target": self.dscr_target,
            "coc_target": self.coc_target,
            "overall_portfolio": asdict(self.overall_portfolio),
            "buckets": [
                {
                    "label": b.label,
                    "n_deals": b.n_deals,
                    "portfolio": asdict(b.portfolio),
                    "avg_rank_score": b.avg_rank_score,
                    "median_rank_score": b.median_rank_score,
                    "pct_meeting_dscr_target": b.pct_meeting_dscr_target,
                    "pct_meeting_coc_target": b.pct_meeting_coc_target,
                }
                for b in self.buckets
            ],
            "corr_rank_dscr": self.corr_rank_dscr,
            "corr_rank_coc": self.corr_rank_coc,
        }


def _load_recent_deal_metrics(
    repo: SqlDealRepository,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pull recent deals from the deals table and extract the key metrics needed
    for evaluation.

    Returns:
        dscr: array of DSCR values (float)
        coc: array of cash-on-cash return values (float)
        labels: array of string labels ("buy", "maybe", "pass", etc.)
        rank_scores: array of composite rank_score values (float)
    """
    rows: Sequence[DealRow] = repo.list_recent(limit=limit)

    dscr_vals: List[float] = []
    coc_vals: List[float] = []
    labels: List[str] = []
    rank_scores: List[float] = []

    for row in rows:
        result = row.result or {}
        finance = result.get("finance") or {}
        score = result.get("score") or {}

        dscr_raw = finance.get("dscr")
        coc_raw = finance.get("cash_on_cash_return")

        # Skip rows that don't have clean DSCR/CoC
        try:
            dscr = float(dscr_raw)
            coc = float(coc_raw)
        except (TypeError, ValueError):
            continue

        label = str(score.get("label", "unknown"))
        rank_score = float(score.get("rank_score", 0.0))

        dscr_vals.append(dscr)
        coc_vals.append(coc)
        labels.append(label)
        rank_scores.append(rank_score)

    if not dscr_vals:
        # Empty arrays, the caller will handle this as "no data"
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=str),
            np.asarray([], dtype=float),
        )

    return (
        np.asarray(dscr_vals, dtype=float),
        np.asarray(coc_vals, dtype=float),
        np.asarray(labels, dtype=str),
        np.asarray(rank_scores, dtype=float),
    )


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Compute Pearson correlation, guarding against degenerate cases.
    """
    if x.size < 2 or y.size < 2:
        return None

    # If all values are identical, correlation is undefined.
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None

    try:
        c = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return None

    if np.isnan(c):
        return None
    return c


def evaluate_engine(
    repo: Optional[SqlDealRepository] = None,
    *,
    limit: int = 1000,
    dscr_target: Optional[float] = None,
    coc_target: Optional[float] = None,
) -> EngineEvaluationReport:
    """
    Core evaluation entrypoint.

    This function answers questions like:
    - As a portfolio, what are my DSCR / CoC stats for recently analyzed deals?
    - Do 'buy' deals consistently meet or exceed my DSCR / CoC targets?
    - Is the composite rank_score actually aligned with DSCR / CoC?
    - How do 'maybe'/'pass' buckets compare?

    Args:
        repo: Deal repository; if None, uses SqlDealRepository(config.DB_URI).
        limit: Number of most recent deals to include in the evaluation.
        dscr_target: Target DSCR threshold (e.g. lender's 1.20). Defaults to config.MIN_DSCR_GOOD.
        coc_target: Target cash-on-cash threshold (e.g. 0.08 = 8%). Defaults to 0.08.

    Returns:
        EngineEvaluationReport with both business-style and technical metrics.
    """
    if repo is None:
        repo = SqlDealRepository(uri=config.DB_URI)

    if dscr_target is None:
        dscr_target = float(config.MIN_DSCR_GOOD)
    if coc_target is None:
        coc_target = 0.08  # 8% default CoC target; adjust if your fund uses something else.

    dscr, coc, labels, rank_scores = _load_recent_deal_metrics(repo, limit=limit)

    total = int(dscr.size)
    if total == 0:
        # No data yet; return an "empty" report.
        empty_portfolio = PortfolioMetrics(
            n_properties=0,
            mean_dscr=float("nan"),
            p5_dscr=float("nan"),
            p50_dscr=float("nan"),
            p95_dscr=float("nan"),
            mean_coc=float("nan"),
            p5_coc=float("nan"),
            p50_coc=float("nan"),
            p95_coc=float("nan"),
        )
        return EngineEvaluationReport(
            total_deals=0,
            dscr_target=dscr_target,
            coc_target=coc_target,
            overall_portfolio=empty_portfolio,
            buckets=[],
            corr_rank_dscr=None,
            corr_rank_coc=None,
        )

    # Overall portfolio stats using existing domain metrics logic
    overall = summarize_portfolio(dscr, coc)

    # Technical sanity: does rank_score move with DSCR/CoC?
    corr_rank_dscr = _safe_corr(rank_scores, dscr)
    corr_rank_coc = _safe_corr(rank_scores, coc)

    # Bucketed metrics by label ("buy", "maybe", "pass", etc.)
    unique_labels = sorted(set(labels.tolist()))
    buckets: List[LabelBucketMetrics] = []

    for label in unique_labels:
        mask = labels == label
        n = int(mask.sum())
        if n == 0:
            continue

        dscr_l = dscr[mask]
        coc_l = coc[mask]
        rank_l = rank_scores[mask]

        portfolio_l = summarize_portfolio(dscr_l, coc_l)

        # Business-style thresholds: "what fraction of these deals are actually acceptable?"
        pct_dscr = float((dscr_l >= dscr_target).mean() * 100.0)
        pct_coc = float((coc_l >= coc_target).mean() * 100.0)

        avg_rank = float(rank_l.mean())
        median_rank = float(np.median(rank_l))

        buckets.append(
            LabelBucketMetrics(
                label=label,
                n_deals=n,
                portfolio=portfolio_l,
                avg_rank_score=avg_rank,
                median_rank_score=median_rank,
                pct_meeting_dscr_target=pct_dscr,
                pct_meeting_coc_target=pct_coc,
            )
        )

    return EngineEvaluationReport(
        total_deals=total,
        dscr_target=dscr_target,
        coc_target=coc_target,
        overall_portfolio=overall,
        buckets=buckets,
        corr_rank_dscr=corr_rank_dscr,
        corr_rank_coc=corr_rank_coc,
    )


def format_report_text(report: EngineEvaluationReport) -> str:
    """
    Render a human-readable multi-line string summarizing the evaluation.

    This is intentionally "boardroom friendly": it explains whether the
    engine is behaving like a sane analyst, not just regurgitating numbers.
    """
    lines: List[str] = []

    lines.append("=== Haven Deal Engine Evaluation ===")
    lines.append(f"Total evaluated deals: {report.total_deals}")
    lines.append(f"Targets: DSCR ≥ {report.dscr_target:.2f}, CoC ≥ {report.coc_target*100:.1f}%")
    lines.append("")

    ov = report.overall_portfolio
    lines.append("Overall portfolio (recent deals):")
    lines.append(f"  n_properties        : {ov.n_properties}")
    lines.append(f"  DSCR mean / P50 / P5 / P95: {ov.mean_dscr:.2f} / {ov.p50_dscr:.2f} / {ov.p5_dscr:.2f} / {ov.p95_dscr:.2f}")
    lines.append(f"  CoC  mean / P50 / P5 / P95: {ov.mean_coc*100:.1f}% / {ov.p50_coc*100:.1f}% / {ov.p5_coc*100:.1f}% / {ov.p95_coc*100:.1f}%")
    lines.append("")

    lines.append("Label buckets (behavioral sanity check):")
    for b in sorted(report.buckets, key=lambda x: x.label):
        lines.append(f"- Label: {b.label!r}")
        lines.append(f"    n_deals                 : {b.n_deals}")
        lines.append(
            "    DSCR mean / P50 / P5 / P95: "
            f"{b.portfolio.mean_dscr:.2f} / {b.portfolio.p50_dscr:.2f} / "
            f"{b.portfolio.p5_dscr:.2f} / {b.portfolio.p95_dscr:.2f}"
        )
        lines.append(
            "    CoC  mean / P50 / P5 / P95: "
            f"{b.portfolio.mean_coc*100:.1f}% / {b.portfolio.p50_coc*100:.1f}% / "
            f"{b.portfolio.p5_coc*100:.1f}% / {b.portfolio.p95_coc*100:.1f}%"
        )
        lines.append(
            f"    % meeting DSCR target   : {b.pct_meeting_dscr_target:.1f}%"
        )
        lines.append(
            f"    % meeting CoC target    : {b.pct_meeting_coc_target:.1f}%"
        )
        lines.append(
            f"    rank_score mean / median: {b.avg_rank_score:.2f} / {b.median_rank_score:.2f}"
        )
        lines.append("")

    lines.append("Correlation sanity (scores vs fundamentals):")
    lines.append(
        f"  corr(rank_score, DSCR): "
        f"{report.corr_rank_dscr:.3f}" if report.corr_rank_dscr is not None else "  corr(rank_score, DSCR): n/a"
    )
    lines.append(
        f"  corr(rank_score, CoC) : "
        f"{report.corr_rank_coc:.3f}" if report.corr_rank_coc is not None else "  corr(rank_score, CoC) : n/a"
    )

    return "\n".join(lines)
