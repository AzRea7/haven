from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DealRowLike(Protocol):
    id: int
    ts: datetime
    result: dict[str, Any]

class RentEstimator(Protocol):
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
    def save_analysis(
        self,
        analysis: dict[str, Any],
        request_payload: dict[str, Any] | None = None,
    ) -> int | None: ...

    # Return read-only sequence to avoid list invariance issues
    def list_recent(self, limit: int = 50) -> Sequence[DealRowLike]: ...
    def get(self, deal_id: int) -> DealRowLike | None: ...
