# src/haven/adapters/rent_estimator_lightgbm.py
from __future__ import annotations

from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


class LightGBMRentEstimator:
    """
    Deterministic rent heuristic used by the pipeline.

    - Keeps interface compatible with a future ML model.
    - Avoids sklearn transformers to eliminate 'feature names' warnings.
    - Monotonic in beds/baths/sqft so scoring behaves sensibly.
    """

    def __init__(self, *_, **__):
        # If you later load a real model, set is_ready=True and gate logic.
        self.is_ready = False

    def predict_unit_rent(
        self,
        bedrooms: float,
        bathrooms: float,
        sqft: float,
        zipcode: str,
        property_type: str,
    ) -> float:
        b = max(float(bedrooms or 0.0), 0.0)
        ba = max(float(bathrooms or 0.0), 0.0)
        s = max(float(sqft or 0.0), 0.0)

        # Local zip anchors (tune these as you gather data)
        base = 900.0
        if zipcode == "48009":
            base = 1600.0
        elif zipcode.startswith("48"):
            base = 1300.0

        # Property type nudges
        if property_type in ("apartment_complex", "multifamily_5plus"):
            base *= 0.92
        elif property_type in ("condo_townhome",):
            base *= 0.97

        # Feature contributions (simple, monotonic, interpretable)
        bed_bonus = 250.0 * b
        bath_bonus = 175.0 * max(ba - 1.0, 0.0)
        size_bonus = 0.40 * max(s - 650.0, 0.0)  # $/sqft for area over studio size

        est = base + bed_bonus + bath_bonus + size_bonus

        # Sanity clamp
        est = float(max(500.0, min(est, 12000.0)))

        logger.info(
            "predict_unit_rent",
            extra={
                "context": {
                    "bedrooms": b,
                    "bathrooms": ba,
                    "sqft": s,
                    "zipcode": zipcode,
                    "property_type": property_type,
                    "predicted_rent": est,
                }
            },
        )
        return est
