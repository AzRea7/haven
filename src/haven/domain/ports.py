from typing import Protocol, Dict, Any

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
    def save_analysis(self, analysis: Dict[str, Any]) -> None:
        ...
