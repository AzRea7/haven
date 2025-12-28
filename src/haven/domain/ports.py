# src/haven/domain/ports.py
from __future__ import annotations

from typing import Any, Iterable, Protocol, TypedDict


# ----------------------------
# Property storage
# ----------------------------

class PropertyRecord(TypedDict, total=False):
    source: str
    external_id: str
    address: str
    city: str
    state: str
    zipcode: str
    lat: float | None
    lon: float | None
    beds: float | None
    baths: float | None
    sqft: float | None
    year_built: int | None
    list_price: float | None
    list_date: str
    property_type: str
    raw: dict[str, Any]


class PropertyRepository(Protocol):
    def upsert_many(self, items: Iterable[PropertyRecord]) -> int:
        ...

    def search(
        self,
        *,
        zipcode: str,
        max_price: float | None = None,
        limit: int = 200,
    ) -> list[PropertyRecord]:
        ...


# ----------------------------
# Deal persistence
# ----------------------------

class DealRepository(Protocol):
    def save_analysis(self, analysis: dict[str, Any], payload: dict[str, Any]) -> int:
        ...

    def get(self, deal_id: int) -> Any:
        ...

    def list_recent(self, limit: int = 50) -> list[Any]:
        ...


# ----------------------------
# Rent estimator interface (standardized)
# ----------------------------

class RentEstimator(Protocol):
    def predict_unit_rent(
        self,
        *,
        address: str,
        city: str,
        state: str,
        zipcode: str,
        bedrooms: float | None,
        bathrooms: float | None,
        sqft: float | None,
        property_type: str | None = None,
    ) -> float:
        ...


# ----------------------------
# Leads
# ----------------------------

class LeadSearchResult(TypedDict):
    lead_id: int
    property_id: int
    org: str
    status: str
    lead_score: float
    reasons: list[str]

    address: str
    city: str
    state: str
    zipcode: str
    lat: float | None
    lon: float | None
    list_price: float

    beds: float | None
    baths: float | None
    sqft: float | None

    features: dict[str, Any]
    source: str
    external_id: str | None


class LeadRepository(Protocol):
    def upsert_many(self, items: list[dict[str, Any]]) -> list[int]:
        ...

    def list_top(
        self,
        *,
        org: str,
        zipcode: str,
        limit: int = 100,
        min_score: float | None = None,
        statuses: list[str] | None = None,
    ) -> list[LeadSearchResult]:
        ...

    def add_event(
        self,
        *,
        lead_id: int,
        org: str,
        event_type: str,
        actor: str,
        note: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        ...

    def set_status(
        self,
        *,
        lead_id: int,
        status: str,
        actor: str,
        note: str | None = None,
    ) -> None:
        ...

    def metrics(
        self,
        *,
        org: str,
        days: int = 30,
        k: int = 25,
    ) -> dict[str, Any]:
        ...
