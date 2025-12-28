# src/haven/adapters/rent_estimator_null.py
from __future__ import annotations

from dataclasses import dataclass

from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class NullRentEstimator:
    """
    Always-available fallback estimator that never errors.
    Useful if both RentCast and LightGBM are unavailable.
    """

    def predict_unit_rent(
        self,
        *,
        bedrooms: float,
        bathrooms: float,
        sqft: float,
        zipcode: str,
        property_type: str,
        address: str | None = None,
        city: str | None = None,
        state: str | None = None,
    ) -> float:
        sqft_f = float(sqft or 0.0)
        beds_f = float(bedrooms or 0.0)
        baths_f = float(bathrooms or 0.0)
        # conservative heuristic
        return max(1.05 * sqft_f + 150.0 * beds_f + 40.0 * baths_f, 0.0)
