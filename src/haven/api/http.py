from collections.abc import Sequence
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query

from haven.adapters.config import config
from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator
from haven.adapters.sql_repo import SqlDealRepository, SqlPropertyRepository, DealRow
from haven.analysis.scoring import score_property  # still used elsewhere if needed
from haven.domain.assumptions import UnderwritingAssumptions
from haven.domain.ports import DealRepository, PropertyRepository
from haven.domain.property import Property
from haven.services.deal_analyzer import (
    analyze_deal_with_defaults,
    analyze_deal,
)
from haven.adapters.rent_quantile_bundle import predict_rent_quantiles
from haven.adapters.arv_quantile_bundle import predict_arv_quantiles
from .schemas import AnalyzeRequest, AnalyzeResponse, TopDealItem

app = FastAPI()

DEFAULT_DOWN_PAYMENT_PCT = 0.25  # 25% down (fallback if validate doesn't override)
DEFAULT_INTEREST_RATE = 0.065  # 6.5% annual
DEFAULT_LOAN_TERM_YEARS = 30
DEFAULT_TAXES_ANNUAL = 3000.0  # fallback if missing
DEFAULT_INSURANCE_ANNUAL = 1200.0  # fallback if missing

_deal_repo: DealRepository = SqlDealRepository(config.DB_URI)
_property_repo: PropertyRepository = SqlPropertyRepository(config.DB_URI)

# Rent estimator is optional; if it blows up, we fall back to defaults inside analyze_deal
try:
    _rent_estimator: LightGBMRentEstimator | None = LightGBMRentEstimator()
except Exception:
    _rent_estimator = None

_default_assumptions = UnderwritingAssumptions(
    vacancy_rate=config.VACANCY_RATE,
    maintenance_rate=config.MAINTENANCE_RATE,
    property_mgmt_rate=config.PROPERTY_MGMT_RATE,
    capex_rate=config.CAPEX_RATE,
    closing_cost_pct=config.DEFAULT_CLOSING_COST_PCT,
    min_dscr_good=config.MIN_DSCR_GOOD,
)


@app.get("/deals", response_model=list[dict])
def list_deals(limit: int = 50) -> list[dict]:
    rows: Sequence[DealRow] = _deal_repo.list_recent(limit=limit)
    return [r.result | {"deal_id": r.id, "ts": r.ts.isoformat()} for r in rows]


@app.post("/analyze")
def analyze_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Low-friction analyze endpoint that just accepts a raw dict payload.

    This is what your debug scripts use.
    """
    try:
        return analyze_deal_with_defaults(raw_payload=payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/analyze2", response_model=AnalyzeResponse)
def analyze_endpoint2(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    Typed analyze endpoint using AnalyzeRequest/AnalyzeResponse schemas.
    """
    try:
        result = analyze_deal_with_defaults(raw_payload=payload.model_dump())
        return AnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/deals/{deal_id}", response_model=dict)
def get_deal(deal_id: int) -> dict:
    row = _deal_repo.get(deal_id)
    if not row:
        raise HTTPException(status_code=404, detail="deal not found")
    return row.result | {"deal_id": row.id, "ts": row.ts.isoformat()}


def _estimate_rent_quantiles(prop: Property, rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate rent quantiles using the LightGBM rent bundle if available.
    """

    bedrooms = float(rec.get("beds") or 0.0)
    bathrooms = float(rec.get("baths") or 0.0)
    sqft = float(rec.get("sqft") or 0.0)

    base = float(rec.get("list_price") or 0.0)

    features = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "zipcode_encoded": hash(prop.zipcode) % 10_000,
        "property_type_encoded": hash(prop.property_type) % 100,
        "base": base,
    }

    return predict_rent_quantiles(features)


def _estimate_arv_quantiles(prop: Property, rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate ARV quantiles using ARV bundle if present, else +/-10% around list price.
    Currently not wired into /top-deals, but kept for future calibration.
    """
    list_price = float(rec.get("list_price") or 0.0)
    sqft = float(rec.get("sqft") or 0.0)

    features = {
        "list_price": list_price,
        "sqft": sqft,
        "zipcode_encoded": hash(prop.zipcode) % 10_000,
        "property_type_encoded": hash(prop.property_type) % 100,
        "base": list_price,
    }

    return predict_arv_quantiles(features)


@app.get("/top-deals", response_model=list[TopDealItem])
def top_deals(
    zip: str = Query(..., alias="zip"),
    max_price: float | None = Query(None),
    strategy: str = Query("rental", regex="^(rental|flip)$"),
    limit: int = Query(50, ge=1, le=500),
    min_dscr: float | None = Query(
        None,
        description="If set, only return deals with DSCR >= this value.",
    ),
    min_coc: float | None = Query(
        None,
        description="If set, only return deals with Cash-on-Cash >= this value (as decimal, e.g. 0.08).",
    ),
    max_breakeven: float | None = Query(
        None,
        description="If set, only return deals with breakeven occupancy <= this value (as decimal, e.g. 0.85).",
    ),
    min_label: str | None = Query(
        None,
        description="If 'maybe', exclude PASS deals. If 'buy', include BUY only.",
    ),
) -> list[TopDealItem]:
    """
    Return ranked deals for a given zip and optional price ceiling.

    - strategy = "rental" → hold / cashflow ranking
    - strategy = "flip"   → spread / flip-probability ranking

    Investor-style filters:
    - min_dscr: require DSCR >= threshold
    - min_coc: require CoC >= threshold (decimal)
    - max_breakeven: require breakeven occupancy <= threshold (decimal)
    - min_label: require at least 'maybe' or 'buy'
    """
    # Normalize strategy to what the rest of the code expects.
    # UI sends "rental" or "flip".
    strategy = strategy.lower()
    if strategy not in ("rental", "flip"):
        strategy = "rental"

    try:
        records = _property_repo.search(
            zipcode=zip,
            max_price=max_price,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"search failed: {e}") from e

    items: list[TopDealItem] = []

    # Keywords we do NOT want in any deal (portfolio view)
    EXCLUDED_TYPE_KEYWORDS = (
        "manufactured",
        "mobile",
        "trailer",
        "condo",
        "condominium",
        "townhome",
        "townhouse",
    )

    for rec in records:
        if not rec.get("list_price"):
            continue

        raw = rec.get("raw") or {}

        # Filter out manufactured/mobile/trailer + condos/townhomes/apartments/etc.
        prop_type_raw = (rec.get("property_type") or "").lower()
        zillow_home_type = (raw.get("homeType") or "").lower()
        zillow_reso_type = (raw.get("resoPropertyType") or "").lower()

        combined = " ".join(
            [
                prop_type_raw,
                zillow_home_type,
                zillow_reso_type,
            ]
        )

        if any(tok in combined for tok in EXCLUDED_TYPE_KEYWORDS):
            continue

        # Days on market from raw payload if present
        dom = float(
            raw.get("daysOnZillow")
            or raw.get("dom")
            or rec.get("dom")
            or 0.0
        )

        # Build payload for the analyzer
        payload: dict[str, Any] = {
            "address": rec["address"],
            "city": rec["city"],
            "state": rec["state"],
            "zipcode": rec["zipcode"],
            "list_price": float(rec["list_price"]),
            "property_type": rec.get("property_type") or "single_family",
            "sqft": float(rec.get("sqft") or 0.0),
            "bedrooms": float(rec.get("beds") or raw.get("beds") or 0.0),
            "bathrooms": float(rec.get("baths") or raw.get("baths") or 0.0),
            "strategy": strategy,  # <-- KEY: pass strategy through
            "days_on_market": dom,
        }

        try:
            analysis = analyze_deal(
                raw_payload=payload,
                rent_estimator=_rent_estimator or LightGBMRentEstimator(),
                repo=_deal_repo,
            )
        except Exception:
            # If a single property blows up, skip it
            continue

        finance = analysis.get("finance", {}) or {}
        score = analysis.get("score", {}) or {}

        # Pull underwriting metrics; allow fallback from 'coc' to 'cash_on_cash_return'
        dscr = float(finance.get("dscr") or 0.0)
        coc_value = finance.get("cash_on_cash_return")
        if coc_value is None:
            coc_value = finance.get("coc") or 0.0
        coc = float(coc_value)
        breakeven = float(finance.get("breakeven_occupancy_pct") or 0.0)
        label = str(score.get("label") or "maybe")

        # -----------------------------
        # Apply investor-style filters
        # -----------------------------
        if min_label:
            ml = min_label.lower()
            lbl = label.lower()
            if ml == "maybe":
                # Drop only PASS deals
                if lbl == "pass":
                    continue
            elif ml == "buy":
                if lbl != "buy":
                    continue

        if min_dscr is not None and dscr < min_dscr:
            continue

        if min_coc is not None and coc < min_coc:
            continue

        if max_breakeven is not None and breakeven > max_breakeven:
            continue

        items.append(
            TopDealItem(
                external_id=str(rec.get("external_id") or ""),
                source=str(rec.get("source") or "unknown"),
                address=rec["address"],
                city=rec["city"],
                state=rec["state"],
                zipcode=rec["zipcode"],
                lat=rec.get("lat"),
                lon=rec.get("lon"),
                list_price=float(rec["list_price"]),
                dscr=dscr,
                cash_on_cash_return=coc,
                breakeven_occupancy_pct=breakeven,
                rank_score=float(score.get("rank_score", 0.0)),
                label=label,  # Pydantic will validate into the enum/union
                reason=str(score.get("reason", "")),
                dom=dom or None,
            )
        )

    # Sort best → worst using rank_score (strategy-aware)
    items.sort(key=lambda x: x.rank_score, reverse=True)
    return items
