# src/haven/api/http.py

from collections.abc import Sequence
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query

from haven.adapters.config import config
from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator
from haven.adapters.sql_repo import SqlDealRepository, SqlPropertyRepository, DealRow
from haven.analysis.finance import analyze_property_financials
from haven.analysis.scoring import score_property
from haven.domain.assumptions import UnderwritingAssumptions
from haven.domain.ports import DealRepository, PropertyRepository
from haven.domain.property import Property
from haven.services.deal_analyzer import analyze_deal_with_defaults
from haven.adapters.rent_quantile_bundle import predict_rent_quantiles
from haven.adapters.arv_quantile_bundle import predict_arv_quantiles
from .schemas import AnalyzeRequest, AnalyzeResponse, TopDealItem

app = FastAPI()

# --------------------------------------------------------------------------
# Screening defaults for /top-deals
# --------------------------------------------------------------------------

DEFAULT_DOWN_PAYMENT_PCT = 0.25        # 25% down
DEFAULT_INTEREST_RATE = 0.065          # 6.5% annual
DEFAULT_LOAN_TERM_YEARS = 30
DEFAULT_TAXES_ANNUAL = 3000.0          # fallback if missing
DEFAULT_INSURANCE_ANNUAL = 1200.0      # fallback if missing

# --------------------------------------------------------------------------
# Core dependencies
# --------------------------------------------------------------------------

_deal_repo: DealRepository = SqlDealRepository(config.DB_URI)
_property_repo: PropertyRepository = SqlPropertyRepository(config.DB_URI)

# Rent estimator is optional; if it blows up, we fall back in quantile helper
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

# --------------------------------------------------------------------------
# Existing endpoints
# --------------------------------------------------------------------------


@app.get("/deals", response_model=list[dict])
def list_deals(limit: int = 50) -> list[dict]:
    rows: Sequence[DealRow] = _deal_repo.list_recent(limit=limit)
    return [r.result | {"deal_id": r.id, "ts": r.ts.isoformat()} for r in rows]


@app.post("/analyze")
def analyze_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return analyze_deal_with_defaults(raw_payload=payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/analyze2", response_model=AnalyzeResponse)
def analyze_endpoint2(payload: AnalyzeRequest) -> AnalyzeResponse:
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


# --------------------------------------------------------------------------
# Helpers for /top-deals
# --------------------------------------------------------------------------


def _build_property_from_record(rec: Dict[str, Any]) -> Property:
    """
    Convert a property row/dict from SqlPropertyRepository.search into a Property
    with standardized screening assumptions.
    """
    return Property(
        property_type=rec.get("property_type", "single_family"),
        address=rec["address"],
        city=rec["city"],
        state=rec["state"],
        zipcode=rec["zipcode"],
        list_price=float(rec["list_price"]),
        down_payment_pct=DEFAULT_DOWN_PAYMENT_PCT,
        interest_rate_annual=DEFAULT_INTEREST_RATE,
        loan_term_years=DEFAULT_LOAN_TERM_YEARS,
        taxes_annual=float(rec.get("taxes_annual") or DEFAULT_TAXES_ANNUAL),
        insurance_annual=float(
            rec.get("insurance_annual") or DEFAULT_INSURANCE_ANNUAL
        ),
        hoa_monthly=float(rec.get("hoa_monthly") or 0.0),
        est_market_rent=None,
        units=None,
    )


def _estimate_rent_quantiles(prop: Property, rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate rent quantiles using:
      - LightGBMRentEstimator (if available) to get base rent
      - rent_quantile_bundle if configured
      - otherwise a +/-10% band around base
    """
    bedrooms = float(rec.get("beds") or 0.0)
    bathrooms = float(rec.get("baths") or 0.0)
    sqft = float(rec.get("sqft") or 0.0)

    base = 0.0
    if _rent_estimator is not None:
        try:
            base = float(
                _rent_estimator.predict_unit_rent(
                    bedrooms=bedrooms,
                    bathrooms=bathrooms,
                    sqft=sqft,
                    zipcode=prop.zipcode,
                    property_type=prop.property_type,
                )
                or 0.0
            )
        except Exception:
            base = 0.0

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


# --------------------------------------------------------------------------
# /top-deals: zip-level ranked opportunities
# --------------------------------------------------------------------------


@app.get("/top-deals", response_model=list[TopDealItem])
def top_deals(
    zip: str = Query(..., alias="zip"),
    max_price: float | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[TopDealItem]:
    """
    Return ranked deals for a given zip and optional price ceiling.
    Used directly by the frontend TopDealsTable.
    """
    try:
        records = _property_repo.search(
            zipcode=zip,
            max_price=max_price,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"search failed: {e}") from e

    items: list[TopDealItem] = []

    for rec in records:
        if not rec.get("list_price"):
            continue

        # 1) Build Property snapshot with default assumptions
        prop = _build_property_from_record(rec)

        # 2) Underwriting metrics
        finance = analyze_property_financials(prop, _default_assumptions)

        # 3) Quantiles (uncertainty)
        rent_q = _estimate_rent_quantiles(prop, rec)
        arv_q = _estimate_arv_quantiles(prop, rec)

        # 4) DOM
        raw = rec.get("raw") or {}
        dom = float(
            raw.get("days_on_zillow")
            or raw.get("dom")
            or rec.get("dom")
            or 0.0
        )

        # 5) Risk-adjusted score
        score = score_property(
            finance=finance,
            arv_q=arv_q,
            rent_q=rent_q,
            dom=dom,
            strategy="hold",
        )

        # 6) Best-effort persistence: history for future calibration
        try:
            _deal_repo.save_analysis(
                analysis={
                    "property": {
                        "external_id": rec.get("external_id"),
                        "source": rec.get("source"),
                        "address": rec.get("address"),
                        "city": rec.get("city"),
                        "state": rec.get("state"),
                        "zipcode": rec.get("zipcode"),
                    },
                    "finance": finance,
                    "score": score,
                    "arv_q": arv_q,
                    "rent_q": rent_q,
                    "dom": dom,
                },
                request_payload={
                    "zip": zip,
                    "max_price": max_price,
                    "screening_terms": {
                        "down_payment_pct": prop.down_payment_pct,
                        "interest_rate_annual": prop.interest_rate_annual,
                        "loan_term_years": prop.loan_term_years,
                    },
                },
            )
        except Exception:
            # don't break the endpoint if logging/history fails
            pass

        # 7) Build response object
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
                dscr=float(finance.get("dscr", 0.0)),
                cash_on_cash_return=float(
                    finance.get("cash_on_cash_return", 0.0)
                ),
                breakeven_occupancy_pct=float(
                    finance.get("breakeven_occupancy_pct", 0.0)
                ),
                rank_score=float(score.get("rank_score", 0.0)),
                label=str(score.get("label", "maybe")),
                reason=str(score.get("reason", "")),
                dom=dom or None,
            )
        )

    # 8) Sort best → worst
    items.sort(key=lambda x: x.rank_score, reverse=True)
    return items
