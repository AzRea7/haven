# src/haven/domain/ports.py
from __future__ import annotations

from typing import Any, Protocol, TypedDict, Iterable


class RentEstimator(Protocol):
    """
    Provides rent estimates for a given unit / property configuration.
    Implementations can wrap your ML models or external APIs.
    """

    def predict_unit_rent(
        self,
        bedrooms: float,
        bathrooms: float,
        sqft: float,
        zipcode: str,
        property_type: str,
    ) -> float:
        ...


class DealRepository(Protocol):
    """
    Persists completed analyses.

    Implementations:
      - SqlDealRepository (production)
      - InMemoryDealRepository (tests)
    """

    def save_analysis(
        self,
        analysis: dict[str, Any],
        request_payload: dict[str, Any],
    ) -> int | None:
        """
        Persist a completed deal analysis + original request.

        Returns:
          - deal_id (int) for durable stores
          - None for fire-and-forget implementations
        """
        ...


class PropertyRecord(TypedDict, total=False):
    """
    Normalized listing shape for ingestion & search.

    This is intentionally minimal and generic so it works across:
    - HasData Zillow Listing API
    - Zillow deep property API
    - MLS / other feeds

    Required keys for our pipeline:
      - external_id, source
      - address, city, state, zipcode
      - list_price, property_type
      - core physical details: beds, baths, sqft
      - geo: lat, lon (0.0 allowed if missing)
    """

    external_id: str
    source: str

    address: str
    city: str
    state: str
    zipcode: str

    lat: float
    lon: float

    beds: float
    baths: float
    sqft: float
    year_built: int

    list_price: float
    list_date: str  # ISO8601 string (optional / may be "")

    property_type: str

    # Raw upstream payload for debugging / future features
    raw: dict[str, Any]


class PropertySource(Protocol):
    """
    Abstraction over HasData Zillow Listing API / Property API / MLS, etc.

    Implementations must:
      - Call their upstream API(s)
      - Map responses into PropertyRecord objects
    """

    def search(
        self,
        zipcode: str,
        property_types: list[str] | None = None,
        max_price: float | None = None,
        limit: int = 200,
    ) -> list[PropertyRecord]:
        """
        Return up to `limit` PropertyRecord entries matching the filters.
        """
        ...


class PropertyRepository(Protocol):
    """
    Storage for ingested listings used by models & /top-deals.

    Implementations:
      - SqlPropertyRepository
      - InMemoryPropertyRepository (tests)
    """

    def upsert_many(self, items: Iterable[PropertyRecord]) -> int:
        """
        Insert or update properties by (source, external_id).
        Returns number of rows written.
        """
        ...

    def search(
        self,
        zipcode: str,
        max_price: float | None = None,
        limit: int = 200,
    ) -> list[PropertyRecord]:
        """
        Query curated properties for a given ZIP (and optional price cap).
        Returns normalized PropertyRecord objects.
        """
        ...
