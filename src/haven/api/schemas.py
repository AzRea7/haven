# src/haven/api/schemas.py
from typing import Optional
from pydantic import BaseModel, Field


class UnitIn(BaseModel):
    bedrooms: float | None = None
    bathrooms: float | None = None
    sqft: float | None = None
    market_rent: float | None = None


class AnalyzeRequest(BaseModel):
    property_type: str
    address: str
    city: str
    state: str
    zipcode: str
    list_price: float

    down_payment_pct: float
    interest_rate_annual: float
    loan_term_years: int

    taxes_annual: float
    insurance_annual: float
    hoa_monthly: float = 0.0

    sqft: float | None = None
    est_market_rent: float | None = None
    units: list[UnitIn] | None = Field(default=None)


class AnalyzeResponse(BaseModel):
    """
    Thin Pydantic wrapper around the dict that analyze_deal_with_defaults
    returns. We keep everything as dicts so we don't fight the existing
    service-layer shape.
    """

    deal_id: int | None = None
    address: dict
    property_type: str
    finance: dict
    pricing: dict
    score: dict


class TopDealItem(BaseModel):
    """
    Shape returned by /top-deals and consumed by the frontend TopDealsTable/Map.

    NOTE: The frontend adds its own `id` client-side (or uses external_id)
    so we don't enforce it here. What matters is that these fields exist.
    """

    external_id: str = Field(..., description="Source listing ID")
    source: str = Field(..., description="Listing source identifier")

    address: str
    city: str
    state: str
    zipcode: str

    lat: Optional[float] = None
    lon: Optional[float] = None

    list_price: float

    dscr: float
    cash_on_cash_return: float
    breakeven_occupancy_pct: float

    rank_score: float
    label: str
    reason: str

    dom: Optional[float] = None
