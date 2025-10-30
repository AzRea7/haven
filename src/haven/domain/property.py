from typing import Optional, List, Literal
from pydantic import BaseModel, Field, validator

# All asset classes we support
PropertyType = Literal[
    "single_family",
    "condo_townhome",
    "duplex_4plex",
    "apartment_unit",
    "apartment_complex"
]

class Unit(BaseModel):
    bedrooms: Optional[float] = None
    bathrooms: Optional[float] = None
    sqft: Optional[float] = None
    market_rent: Optional[float] = None  # expected achievable rent for this unit

class Property(BaseModel):
    property_type: PropertyType

    address: str
    city: str
    state: str
    zipcode: str

    list_price: float = Field(..., description="Asking or assumed purchase price")
    down_payment_pct: float = Field(..., description="0.25 means 25% down")
    interest_rate_annual: float = Field(..., description="e.g. 0.065 for 6.5% APR")
    loan_term_years: int = Field(..., description="Amortization period in years")

    taxes_annual: float
    insurance_annual: float
    hoa_monthly: float = 0.0

    # For multifamily / complexes
    units: Optional[List[Unit]] = None

    # For single-door deals (house, condo, single apartment)
    est_market_rent: Optional[float] = None

    @validator("down_payment_pct")
    def _pct_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("down_payment_pct must be between 0 and 1")
        return v
