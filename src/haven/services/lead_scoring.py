from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LeadPreview:
    dscr: Optional[float]
    cash_on_cash_return: Optional[float]
    rank_score: Optional[float]
    label: Optional[str]
    reason: Optional[str]
    lead_score: float


def compute_lead_score(rank_score: Optional[float], dscr: Optional[float], coc: Optional[float]) -> float:
    """
    V1: lead_score = rank_score (simple, consistent with your ranking engine)

    Later you can blend:
      lead_score = 0.70*rank_score + 0.20*clip(dscr) + 0.10*clip(coc)
    """
    if rank_score is None:
        return 0.0
    return float(rank_score)


def preview_from_analysis_result(result: dict[str, Any]) -> LeadPreview:
    """
    Normalize whatever your analyzer returns into lead preview fields.
    This function is intentionally defensive: it won’t crash if keys differ.
    """
    # common names you’ve used in the project:
    dscr = result.get("dscr")
    coc = result.get("cash_on_cash_return") or result.get("coc")
    score = result.get("rank_score") or result.get("score") or result.get("rankScore")
    label = result.get("label")
    reason = result.get("reason")

    lead_score = compute_lead_score(score, dscr, coc)

    return LeadPreview(
      dscr=float(dscr) if dscr is not None else None,
      cash_on_cash_return=float(coc) if coc is not None else None,
      rank_score=float(score) if score is not None else None,
      label=str(label) if label is not None else None,
      reason=str(reason) if reason is not None else None,
      lead_score=float(lead_score),
    )
