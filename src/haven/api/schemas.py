
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
    deal_id: int | None = None
    address: dict
    property_type: str
    finance: dict
    pricing: dict
    score: dict
