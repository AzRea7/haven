from typing import Optional, List
from pydantic import BaseModel, Field

class UnitIn(BaseModel):
    bedrooms: Optional[float] = None
    bathrooms: Optional[float] = None
    sqft: Optional[float] = None
    market_rent: Optional[float] = None

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
    sqft: Optional[float] = None
    est_market_rent: Optional[float] = None
    units: Optional[List[UnitIn]] = Field(default=None)

class AnalyzeResponse(BaseModel):
    deal_id: Optional[int] = None
    address: dict
    property_type: str
    finance: dict
    pricing: dict
    score: dict
