# src/haven/domain/ports.py
from typing import Protocol

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
