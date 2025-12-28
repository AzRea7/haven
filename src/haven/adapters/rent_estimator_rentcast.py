# src/haven/adapters/rent_estimator_rentcast.py
from __future__ import annotations

from dataclasses import dataclass

from haven.adapters.logging_utils import get_logger
from haven.adapters.rentcast_client import RentCastError, make_rentcast_client

logger = get_logger(__name__)


@dataclass
class RentCastRentEstimator:
    """
    RentCast-backed rent estimator.

    IMPORTANT:
    RentCast works best with a full address, but we keep compatibility with the
    existing interface by accepting address/city/state as optional inputs.

    predict_unit_rent(...) returns a float point estimate.
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

        # If we don't have enough address info, we cannot reliably query RentCast.
        # Fallback to a safe heuristic (similar to LightGBM fallback).
        if not address or not city or not state or not zipcode:
            logger.warning(
                "rentcast_missing_address_fallback",
                extra={"has_address": bool(address), "has_city": bool(city), "has_state": bool(state), "zipcode": str(zipcode)},
            )
            return max(1.15 * sqft_f + 175.0 * beds_f + 50.0 * baths_f, 0.0)

        client = make_rentcast_client()
        full_addr = f"{address}, {city}, {state} {str(zipcode).strip()}"

        params: dict[str, object] = {"address": full_addr}

        # RentCast supports these optional hints
        if beds_f > 0:
            params["bedrooms"] = beds_f
        if baths_f > 0:
            params["bathrooms"] = baths_f
        if sqft_f > 0:
            params["squareFootage"] = sqft_f

        try:
            payload = client.get("/avm/rent/long-term", params=params)
            if not isinstance(payload, dict):
                raise RentCastError(f"Unexpected response type: {type(payload)}")

            rent = payload.get("rent")
            if rent is None:
                low = payload.get("rentRangeLow")
                high = payload.get("rentRangeHigh")
                if low is not None and high is not None:
                    return max((float(low) + float(high)) / 2.0, 0.0)
                raise RentCastError(f"No rent fields returned for address={full_addr}")

            return max(float(rent), 0.0)

        except Exception as e:
            logger.warning("rentcast_failed_fallback", extra={"error": str(e), "address": full_addr})
            return max(1.15 * sqft_f + 175.0 * beds_f + 50.0 * baths_f, 0.0)
