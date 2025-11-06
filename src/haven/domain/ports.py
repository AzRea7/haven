from typing import Any, Protocol


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
    ) -> int | None:
        ...
