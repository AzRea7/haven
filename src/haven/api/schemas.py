# src/haven/api/schemas.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel
from pydantic import ConfigDict


# --------------------------------------------
# Analyze (typed endpoint)
# --------------------------------------------

Strategy = Literal["rental", "flip"]


class AnalyzeRequest(BaseModel):
    """
    Typed request for /analyze2.

    Keep this permissive so your debug scripts can pass extra fields without breaking.
    """
    model_config = ConfigDict(extra="allow")

    address: str
    city: str
    state: str
    zipcode: str

    list_price: float = 0.0
    bedrooms: float = 0.0
    bathrooms: float = 0.0
    sqft: float = 0.0
    property_type: str = "single_family"

    strategy: Strategy = "rental"
    days_on_market: float = 0.0


class AnalyzeResponse(BaseModel):
    """
    Typed response for /analyze2.

    Your analyzer returns a rich dict with nested structures.
    Make this permissive so it won't break when you add fields.
    """
    model_config = ConfigDict(extra="allow")


# --------------------------------------------
# Top Deals (existing UI)
# --------------------------------------------

Label = Literal["buy", "maybe", "pass"]


class TopDealItem(BaseModel):
    """
    Response item for /top-deals.
    Mirrors your frontend TopDealItem type.
    """
    model_config = ConfigDict(extra="allow")

    external_id: str = ""
    source: str = "unknown"

    address: str
    city: str
    state: str
    zipcode: str

    lat: float | None = None
    lon: float | None = None

    list_price: float = 0.0

    dscr: float = 0.0
    cash_on_cash_return: float = 0.0
    breakeven_occupancy_pct: float = 0.0

    rank_score: float = 0.0
    label: Label = "maybe"
    reason: str = ""

    dom: float | None = None


# --------------------------------------------
# Leads (new)
# --------------------------------------------

LeadStage = Literal[
    "new",
    "contacted",
    "appointment",
    "contract",
    "closed_won",
    "closed_lost",
    "dead",
]

LeadEventType = Literal[
    "attempt",
    "contacted",
    "appointment",
    "contract",
    "closed_won",
    "closed_lost",
    "dead",
    "note",
]


class LeadItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    lead_id: int

    address: str
    city: str
    state: str
    zipcode: str

    lat: float | None = None
    lon: float | None = None

    source: str
    external_id: str | None = None

    lead_score: float
    stage: LeadStage

    created_at: datetime
    updated_at: datetime

    last_contacted_at: datetime | None = None
    touches: int = 0
    owner: str | None = None

    # Optional underwriting preview
    list_price: float | None = None
    dscr: float | None = None
    cash_on_cash_return: float | None = None
    rank_score: float | None = None
    label: str | None = None
    reason: str | None = None  


class LeadEventCreate(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_type: LeadEventType
    note: str | None = None
    meta: dict[str, Any] | None = None
