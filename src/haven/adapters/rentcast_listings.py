# src/haven/adapters/rentcast_listings.py
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional

from haven.adapters.rentcast_client import make_rentcast_client
from haven.domain.ports import PropertyRecord


def _stable_id(*parts: str) -> str:
    s = "|".join([p.strip().lower() for p in parts if p])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:20]


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _normalize_property_type(raw: Dict[str, Any]) -> str:
    # You can refine this later based on actual RentCast schema fields you see.
    t = str(raw.get("propertyType") or raw.get("type") or "").lower()
    if "single" in t:
        return "single_family"
    if "condo" in t or "town" in t:
        return "condo_townhome"
    if "duplex" in t or "triplex" in t or "fourplex" in t:
        return "duplex_4plex"
    return "single_family"


class RentCastSaleListingSource:
    source_name = "rentcast"

    def search(
        self,
        *,
        zipcode: str,
        property_types: List[str] | None = None,
        max_price: float | None = None,
        limit: int = 300,
    ) -> List[PropertyRecord]:
        client = make_rentcast_client()

        # RentCast listings endpoint params:
        # We'll keep it simple: zip + limit + optional maxPrice
        params: Dict[str, Any] = {
            "zip": zipcode,
            "limit": int(limit),
        }
        if max_price is not None:
            # many APIs use maxPrice; if RentCast uses another key your logs will show it.
            params["maxPrice"] = float(max_price)

        payload = client.get("/listings/sale", params=params)

        # RentCast may return {"listings":[...]} or a bare list depending on endpoint shape.
        listings = payload.get("listings") if isinstance(payload, dict) else payload
        if not isinstance(listings, list):
            return []

        out: List[PropertyRecord] = []

        for raw in listings:
            if not isinstance(raw, dict):
                continue

            address = str(raw.get("address") or raw.get("formattedAddress") or "")
            city = str(raw.get("city") or "")
            state = str(raw.get("state") or "")
            zipc = str(raw.get("zip") or raw.get("zipcode") or zipcode)

            lat = _to_float(raw.get("latitude") or raw.get("lat"))
            lon = _to_float(raw.get("longitude") or raw.get("lon"))

            beds = _to_float(raw.get("bedrooms") or raw.get("beds"))
            baths = _to_float(raw.get("bathrooms") or raw.get("baths"))
            sqft = _to_float(raw.get("squareFootage") or raw.get("sqft"))
            year = _to_int(raw.get("yearBuilt"))

            price = _to_float(raw.get("price") or raw.get("listPrice") or raw.get("listingPrice"))
            if price is None:
                continue

            ptype = _normalize_property_type(raw)

            # stable external id:
            ext = (
                str(raw.get("id") or raw.get("listingId") or "")
                .strip()
            )
            if not ext:
                ext = _stable_id(address, city, state, zipc, str(price), str(sqft or ""))

            rec: PropertyRecord = {
                "source": self.source_name,
                "external_id": ext,
                "address": address,
                "city": city,
                "state": state,
                "zipcode": zipc,
                "lat": lat,
                "lon": lon,
                "beds": beds,
                "baths": baths,
                "sqft": sqft,
                "year_built": year,
                "list_price": float(price),
                "list_date": str(raw.get("listedDate") or raw.get("listDate") or ""),
                "property_type": ptype,
                "raw": raw,  # keep full raw for later feature extraction
            }

            # optional property type filtering
            if property_types and ptype not in property_types:
                continue

            out.append(rec)

        return out
