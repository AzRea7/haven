# src/haven/api/http.py
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from haven.adapters.config import config
from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator
from haven.adapters.rent_estimator_rentcast import RentCastRentEstimator
from haven.adapters.sql_repo import (
    DealRow,
    LeadRow,
    SqlDealRepository,
    SqlLeadRepository,
    SqlPropertyRepository,
)
from haven.services.deal_analyzer import analyze_deal
from .schemas import AnalyzeRequest, AnalyzeResponse, TopDealItem, LeadItem, LeadEventCreate

app = FastAPI()

_deal_repo = SqlDealRepository(config.DB_URI)
_property_repo = SqlPropertyRepository(config.DB_URI)
_lead_repo = SqlLeadRepository(config.DB_URI)

# -------------------------------------------------------------------
# Rent estimator selection (single init at startup)
# -------------------------------------------------------------------
try:
    if getattr(config, "RENTCAST_USE_FOR_RENT_ESTIMATES", False):
        _rent_estimator = RentCastRentEstimator()
    else:
        _rent_estimator = LightGBMRentEstimator()
except Exception:
    try:
        _rent_estimator = LightGBMRentEstimator()
    except Exception:
        _rent_estimator = None


@app.get("/deals", response_model=list[dict])
def list_deals(limit: int = 50) -> list[dict]:
    rows: Sequence[DealRow] = _deal_repo.list_recent(limit=limit)
    return [r.result | {"deal_id": r.id, "ts": r.ts.isoformat()} for r in rows]


@app.post("/analyze2", response_model=AnalyzeResponse)
def analyze_endpoint2(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    This is your normal "save to deals" analysis endpoint.
    """
    try:
        result = analyze_deal(
            raw_payload=payload.model_dump(),
            rent_estimator=_rent_estimator or LightGBMRentEstimator(),
            repo=_deal_repo,
            save=True,
        )
        return AnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# -----------------------------
# LEADS: core helpers
# -----------------------------
def _sigmoid(x: float) -> float:
    # Stable-ish sigmoid
    if x < -50:
        return 0.0
    if x > 50:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# Upstream types you said you do NOT want
_EXCLUDED_UPSTREAM_TYPES = {
    "condo",
    "condominium",
    "townhouse",
    "town home",
    "manufactured",
    "mobile",
    "mobile home",
    "land",
    "lot",
    "vacant land",
}


def _detect_excluded_property_type(prop_rec: dict[str, Any]) -> str | None:
    """
    Quick exclusion check BEFORE we call analyzer.
    This prevents wasted CPU and avoids pydantic literal validation errors.
    """
    raw = prop_rec.get("raw") or {}
    pt = prop_rec.get("property_type") or raw.get("propertyType") or raw.get("homeType") or raw.get("type") or ""
    t = str(pt).lower().strip()
    if not t:
        return None

    for bad in _EXCLUDED_UPSTREAM_TYPES:
        if bad in t:
            return str(pt)
    return None


def _compute_lead_preview(prop_rec: dict[str, Any], *, strategy: str = "rental") -> dict[str, Any]:
    """
    Compute underwriting preview ONCE per property and turn it into a lead score.

    CRITICAL FIXES:
    - Runs analyzer in PREVIEW mode: repo=None, save=False
      (prevents sqlite lock + makes bulk scoring fast)
    - Excludes unwanted property types up-front
    - Returns 'reason' always, so UI can explain why score is 0

    Returns fields that should be stored on the lead:
      lead_score, dscr, cash_on_cash_return, rank_score, label, reason
    """
    if not prop_rec.get("list_price"):
        return {"lead_score": 0.0, "reason": "missing list_price"}

    excluded = _detect_excluded_property_type(prop_rec)
    if excluded:
        return {"lead_score": 0.0, "reason": f"excluded property_type: {excluded}"}

    raw = prop_rec.get("raw") or {}
    dom_raw = raw.get("daysOnZillow") or raw.get("dom") or raw.get("days_on_market") or 0.0
    try:
        dom = float(dom_raw or 0.0)
    except Exception:
        dom = 0.0

    # Build analyzer payload
    payload: dict[str, Any] = {
        "address": prop_rec.get("address", "") or "",
        "city": prop_rec.get("city", "") or "",
        "state": prop_rec.get("state", "") or "",
        "zipcode": prop_rec.get("zipcode", "") or "",
        "list_price": float(prop_rec.get("list_price") or 0.0),

        # IMPORTANT:
        # feed whatever we have; deal_analyzer.py will normalize and/or reject.
        "property_type": prop_rec.get("property_type") or raw.get("propertyType") or raw.get("homeType") or "single_family",

        "sqft": float(prop_rec.get("sqft") or 0.0),
        "bedrooms": float(prop_rec.get("beds") or 0.0),
        "bathrooms": float(prop_rec.get("baths") or 0.0),
        "strategy": strategy,
        "days_on_market": dom,

        # pass-through raw so analyzer has more context if needed
        "raw": raw,
    }

    try:
        analysis = analyze_deal(
            raw_payload=payload,
            rent_estimator=_rent_estimator or LightGBMRentEstimator(),
            repo=None,       # PREVIEW MODE: do not write deals
            save=False,      # PREVIEW MODE
        )
    except Exception as e:
        return {"lead_score": 0.0, "reason": f"analyze_deal failed: {e}"}

    finance = analysis.get("finance", {}) or {}
    score = analysis.get("score", {}) or {}

    label = str(score.get("label") or "maybe").lower()

    try:
        rank_score = float(score.get("rank_score") or 0.0)
    except Exception:
        rank_score = 0.0

    try:
        dscr = float(finance.get("dscr") or 0.0)
    except Exception:
        dscr = 0.0

    coc_value = finance.get("cash_on_cash_return")
    if coc_value is None:
        coc_value = finance.get("coc")
    try:
        coc = float(coc_value or 0.0)
    except Exception:
        coc = 0.0

    # Convert analyzer rank_score into [0..100] lead score.
    base = _sigmoid(rank_score / 2.0)  # softness factor

    # Label nudges
    if label == "buy":
        base += 0.10
    elif label == "pass":
        base -= 0.10

    # DOM penalty (very mild)
    dom_pen = min(dom / 180.0, 1.0) * 0.08
    base -= dom_pen

    # DSCR stability nudge
    if dscr >= 1.2:
        base += 0.05
    elif dscr < 1.0:
        base -= 0.05

    lead_score = max(min(base, 1.0), 0.0) * 100.0

    reason = (
        f"rank={rank_score:.2f} label={label} dscr={dscr:.2f} coc={coc:.2f} "
        f"dom={dom:.0f}d strat={strategy}"
    )

    return {
        "lead_score": float(lead_score),
        "dscr": float(dscr),
        "cash_on_cash_return": float(coc),
        "rank_score": float(rank_score),
        "label": str(label),
        "reason": reason,
    }


# -----------------------------
# LEADS: endpoints
# -----------------------------
@app.post("/leads/from-properties")
def leads_from_properties(
    zip: str = Query(..., alias="zip"),
    max_price: float | None = Query(None),
    limit: int = Query(300, ge=1, le=2000),
    strategy: str = Query("rental", description="rental|flip (used for preview analysis)"),
    workers: int = Query(16, ge=1, le=64, description="Parallel preview workers (thread pool)"),
) -> dict[str, Any]:
    """
    Convert properties in core.properties (already ingested) into leads.

    SPEED + STABILITY CHANGES:
    - We compute previews in parallel (thread pool).
    - Preview analysis does NOT write to deals DB (repo=None, save=False).
      This prevents sqlite lock storms and makes /leads/from-properties much faster.

    Flow:
    1) Load properties
    2) Compute preview for each property (parallel)
    3) Upsert leads using precomputed preview fields (single DB writer)
    """
    t0 = datetime.utcnow()

    try:
        props = _property_repo.search(zipcode=zip, max_price=max_price, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"property search failed: {e}") from e

    # Normalize/guard workers (avoid runaway on tiny machines)
    cpu = os.cpu_count() or 4
    workers = max(1, min(int(workers), 64, cpu * 4))

    # Precompute preview results in parallel (NO DB writes inside this stage)
    preview_by_key: dict[str, dict[str, Any]] = {}
    excluded = 0
    failed = 0

    def _lead_key(p: dict[str, Any]) -> str:
        # match repo identity: prefer source+external_id, else source+address+zip
        source = str(p.get("source") or "unknown")
        ext = (p.get("external_id") or "").strip()
        if ext:
            return f"{source}::{ext}"
        return f"{source}::{str(p.get('address') or '').strip()}::{str(p.get('zipcode') or '').strip()}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for p in props:
            key = _lead_key(p)

            # early exclusion shortcut (so we don't even submit to executor)
            excluded_pt = _detect_excluded_property_type(p)
            if excluded_pt:
                excluded += 1
                preview_by_key[key] = {"lead_score": 0.0, "reason": f"excluded property_type: {excluded_pt}"}
                continue

            futures[ex.submit(_compute_lead_preview, p, strategy=strategy)] = key

        for fut in as_completed(futures):
            key = futures[fut]
            try:
                preview = fut.result()
            except Exception as e:
                failed += 1
                preview = {"lead_score": 0.0, "reason": f"preview worker failed: {e}"}
            preview_by_key[key] = preview

    # Now do a single writer pass into sqlite
    def _preview_fn(p: dict[str, Any]) -> dict[str, Any]:
        return preview_by_key.get(_lead_key(p), {"lead_score": 0.0, "reason": "missing preview"})

    try:
        stats = _lead_repo.upsert_from_properties(
            properties=props,
            compute_preview_fn=_preview_fn,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"lead upsert failed: {e}") from e

    dt = (datetime.utcnow() - t0).total_seconds()

    # stats currently contains created/updated from repo; add preview telemetry
    return {
        "zip": zip,
        "count_properties": len(props),
        "workers": workers,
        "strategy": strategy,
        "excluded": excluded,
        "failed_preview": failed,
        "seconds": dt,
        **stats,
    }


@app.get("/top-leads", response_model=list[LeadItem])
def top_leads(
    zip: str = Query(..., alias="zip"),
    limit: int = Query(200, ge=1, le=1000),
    stage: str | None = Query(None, description="Optional stage filter, e.g. new/contacted"),
) -> list[LeadItem]:
    rows: list[LeadRow] = _lead_repo.list_top_leads(zipcode=zip, limit=limit, stage=stage)

    out: list[LeadItem] = []
    for r in rows:
        out.append(
            LeadItem(
                lead_id=int(r.lead_id or 0),
                address=r.address,
                city=r.city,
                state=r.state,
                zipcode=r.zipcode,
                lat=r.lat,
                lon=r.lon,
                source=r.source,
                external_id=r.external_id,
                lead_score=float(r.lead_score or 0.0),
                stage=r.stage,  # validated by schema
                created_at=r.created_at,
                updated_at=r.updated_at,
                last_contacted_at=r.last_contacted_at,
                touches=int(r.touches or 0),
                owner=r.owner,
                list_price=r.list_price,
                dscr=r.dscr,
                cash_on_cash_return=r.cash_on_cash_return,
                rank_score=r.rank_score,
                label=r.label,
                reason=getattr(r, "reason", None),
            )
        )
    return out


@app.post("/leads/{lead_id}/event")
def add_lead_event(lead_id: int, body: LeadEventCreate) -> dict[str, Any]:
    try:
        ev = _lead_repo.add_event(
            lead_id=lead_id,
            event_type=body.event_type,
            note=body.note,
            meta=body.meta,
        )
        return {
            "event_id": ev.event_id,
            "lead_id": ev.lead_id,
            "event_type": ev.event_type,
            "ts": ev.ts.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
