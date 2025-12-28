# src/haven/adapters/rentcast_source.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from haven.adapters.rentcast_client import RentCastClient
from haven.domain.ports import PropertyRecord


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


@dataclass(frozen=True)
class RentCastSaleListingSource:
    """
    Pulls active sale listings from RentCast and normalizes into Haven PropertyRecord.
    """
    client: RentCastClient

    def fetch_by_zip(
        self,
        zipcode: str,
        *,
        limit: int = 200,
        offset: int = 0,
        max_price: float | None = None,
        min_price: float | None = None,
    ) -> list[PropertyRecord]:
        # RentCast supports searching listings by zip/city/state and pagination.
        # We'll use the Sale Listings endpoint.
        params: dict[str, Any] = {
            "zipCode": zipcode,
            "limit": min(int(limit), 500),
            "offset": max(int(offset), 0),
        }
        if max_price is not None:
            params["maxPrice"] = float(max_price)
        if min_price is not None:
            params["minPrice"] = float(min_price)

        data = self.client.get("/listings/sale", params=params)

        # docs show it's a "paginated endpoint" returning listing records
        if not isinstance(data, list):
            # some APIs return {"listings":[...]} - handle both defensively
            listings = data.get("listings", []) if isinstance(data, dict) else []
        else:
            listings = data

        out: list[PropertyRecord] = []
        for it in listings:
            if not isinstance(it, dict):
                continue

            addr = it.get("address") or {}
            if not isinstance(addr, dict):
                addr = {}

            address1 = str(addr.get("street") or addr.get("address") or it.get("formattedAddress") or "").strip()
            city = str(addr.get("city") or it.get("city") or "").strip()
            state = str(addr.get("state") or it.get("state") or "").strip()
            z = str(addr.get("zipCode") or addr.get("zip") or zipcode).strip() or zipcode

            lat = _to_float(addr.get("latitude") or it.get("latitude") or it.get("lat"))
            lon = _to_float(addr.get("longitude") or it.get("longitude") or it.get("lon"))

            beds = _to_float(it.get("bedrooms") or it.get("beds"))
            baths = _to_float(it.get("bathrooms") or it.get("baths"))
            sqft = _to_float(it.get("squareFootage") or it.get("sqft"))
            year_built = _to_int(it.get("yearBuilt") or it.get("year_built"))

            list_price = _to_float(it.get("price") or it.get("listPrice") or it.get("list_price")) or 0.0
            property_type = str(it.get("propertyType") or it.get("property_type") or "single_family")

            # Stable external id if present
            external_id = str(it.get("id") or it.get("listingId") or it.get("propertyId") or "").strip()

            out.append(
                PropertyRecord(
                    source="rentcast",
                    external_id=external_id,
                    address=address1,
                    city=city,
                    state=state,
                    zipcode=z,
                    lat=lat or 0.0,
                    lon=lon or 0.0,
                    beds=beds or 0.0,
                    baths=baths or 0.0,
                    sqft=sqft or 0.0,
                    year_built=year_built or 0,
                    list_price=float(list_price),
                    list_date=str(it.get("listedDate") or it.get("listDate") or ""),
                    property_type=property_type,
                    raw=it,  # store whole record for feature mining
                )
            )

        return out
