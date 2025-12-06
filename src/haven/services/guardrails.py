# src/haven/services/guardrails.py
from __future__ import annotations

from typing import Any, Dict, List

from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def apply_guardrails(
    payload: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach simple, high-leverage sanity checks to the deal analysis result.

    Produces:
        result["guardrails"] = {
            "has_flags": bool,
            "flags": [
                {
                    "code": "ARV_TOO_HIGH",
                    "severity": "warning" | "error",
                    "message": "...human readable...",
                    "context": {...raw numbers...},
                },
                ...
            ],
        }

    These do *not* block anything; they just flag sketchy deals so the
    UI / caller can highlight them.
    """
    flags: List[Dict[str, Any]] = []

    finance = result.get("finance") or {}
    pricing = result.get("pricing") or {}

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------
    list_price = _safe_float(payload.get("list_price"), 0.0)

    # Try a few possible ARV fields; fall back to ask_price if nothing else
    arv_est = _safe_float(
        pricing.get("arv_p50")
        or pricing.get("arv_estimate")
        or pricing.get("arv")
        or pricing.get("arv_median")
        or pricing.get("ask_price"),  # worst-case, acts as a no-op
        0.0,
    )

    rehab_total = _safe_float(
        finance.get("rehab_total")
        or finance.get("rehab_budget")
        or payload.get("rehab_total")
        or payload.get("rehab_budget"),
        0.0,
    )

    profit_p50 = _safe_float(pricing.get("profit_p50"), 0.0)
    mao_p50 = _safe_float(pricing.get("mao_p50"), 0.0)
    dscr = _safe_float(finance.get("dscr"), 0.0)

    # ------------------------------------------------------------------
    # 1) Basic data sanity
    # ------------------------------------------------------------------
    if list_price <= 0:
        flags.append(
            {
                "code": "LIST_PRICE_MISSING",
                "severity": "warning",
                "message": "List price is missing or zero.",
                "context": {"list_price": list_price},
            }
        )

    # ------------------------------------------------------------------
    # 2) ARV vs list price sanity
    # ------------------------------------------------------------------
    if list_price > 0 and arv_est > 0:
        ratio = arv_est / list_price

        if ratio < 0.5:
            flags.append(
                {
                    "code": "ARV_TOO_LOW",
                    "severity": "warning",
                    "message": "ARV is less than 50% of list price. Likely a bad flip/hold.",
                    "context": {
                        "list_price": list_price,
                        "arv_est": arv_est,
                        "ratio": ratio,
                    },
                }
            )
        elif ratio > 3.0:
            flags.append(
                {
                    "code": "ARV_TOO_HIGH",
                    "severity": "warning",
                    "message": "ARV is more than 3× list price. Check comps / model.",
                    "context": {
                        "list_price": list_price,
                        "arv_est": arv_est,
                        "ratio": ratio,
                    },
                }
            )

    # ------------------------------------------------------------------
    # 3) Rehab vs ARV sanity
    # ------------------------------------------------------------------
    if arv_est > 0 and rehab_total > arv_est:
        flags.append(
            {
                "code": "REHAB_EXCEEDS_ARV",
                "severity": "error",
                "message": "Rehab budget exceeds ARV. Deal almost certainly does not pencil.",
                "context": {
                    "arv_est": arv_est,
                    "rehab_total": rehab_total,
                },
            }
        )

    # ------------------------------------------------------------------
    # 4) Profit & MAO sanity
    # ------------------------------------------------------------------
    if profit_p50 < 0:
        flags.append(
            {
                "code": "NEGATIVE_PROFIT_P50",
                "severity": "warning",
                "message": "Median profit (p50) is negative.",
                "context": {"profit_p50": profit_p50},
            }
        )

    if mao_p50 > 0 and list_price > mao_p50:
        flags.append(
            {
                "code": "LIST_ABOVE_MAO",
                "severity": "warning",
                "message": "List price is above MAO (p50). Negotiation or walk-away.",
                "context": {
                    "list_price": list_price,
                    "mao_p50": mao_p50,
                },
            }
        )

    # ------------------------------------------------------------------
    # 5) DSCR sanity
    # ------------------------------------------------------------------
    if dscr and dscr < 1.0:
        flags.append(
            {
                "code": "DSCR_BELOW_ONE",
                "severity": "warning",
                "message": "DSCR below 1.0 — property does not cover debt service from rent.",
                "context": {"dscr": dscr},
            }
        )

    # ------------------------------------------------------------------
    # Attach & log
    # ------------------------------------------------------------------
    result.setdefault("guardrails", {})
    result["guardrails"]["flags"] = flags
    result["guardrails"]["has_flags"] = bool(flags)

    if flags:
        logger.info("deal_guardrails_flags", extra={"context": {"flags": flags}})

    return result
