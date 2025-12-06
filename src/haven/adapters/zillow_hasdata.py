from __future__ import annotations

import os
import time
from typing import Any, List

import requests

from haven.adapters.logging_utils import get_logger
from haven.domain.ports import PropertySource, PropertyRecord

logger = get_logger(__name__)

# --- API key ---

HASDATA_API_KEY = os.getenv("HASDATA_API_KEY") or os.getenv("HASDATA_ZILLOW_API_KEY")
if not HASDATA_API_KEY:
    raise RuntimeError("HASDATA_API_KEY environment variable is required for HasData Zillow APIs")

HEADERS = {
    "x-api-key": HASDATA_API_KEY,
    "Content-Type": "application/json",
}

# --- Endpoints ---
# Check your HasData docs dashboard; adjust if they show a slightly different path.
HASDATA_ZILLOW_LISTING_URL = os.getenv(
    "HASDATA_ZILLOW_LISTING_URL",
    "https://api.hasdata.com/scrape/zillow/listing",
)

HASDATA_ZILLOW_PROPERTY_URL = os.getenv(
    "HASDATA_ZILLOW_PROPERTY_URL",
    "https://api.hasdata.com/scrape/zillow/property",
)


class HasDataZillowPropertySource(PropertySource):
    """
    Wrapper around HasData's Zillow Listing API.

    Uses:
      - keyword: ZIP or "city, state"
      - type: forSale | forRent | sold (required)
      - price.min / price.max
      - (optional) homeTypes, etc.

    We start with a conservative param set to avoid 500s:
      keyword + type + price.max + pagination.
    """

    source_name = "zillow_hasdata"

    def __init__(
        self,
        session: requests.Session | None = None,
        page_size: int = 50,
        max_pages: int = 5,
        pause_seconds: float = 1.0,
        listing_type: str = "forSale",
    ):
        """
        listing_type:
          - forSale
          - forRent
          - sold
        """
        self.s = session or requests.Session()
        self.page_size = page_size
        self.max_pages = max_pages
        self.pause = pause_seconds
        self.listing_type = listing_type

    # --------- PropertySource API ---------

    def search(
        self,
        zipcode: str,
        property_types: List[str] | None = None,
        max_price: float | None = None,
        limit: int = 200,
    ) -> List[PropertyRecord]:
        results: List[PropertyRecord] = []
        page = 1

        while len(results) < limit and page <= self.max_pages:
            params: dict[str, Any] = {
                # From HasData docs: keyword + type are required.
                "keyword": zipcode,
                "type": self.listing_type,  # e.g. "forSale"
                "page": page,
                "pageSize": self.page_size,
            }

            # Price filter: use dotted keys like docs
            if max_price is not None:
                params["price.max"] = int(max_price)

            resp = self.s.get(
                HASDATA_ZILLOW_LISTING_URL,
                headers=HEADERS,
                params=params,
                timeout=40,
            )

            # 429: rate-limited, backoff and retry same page
            if resp.status_code == 429:
                logger.warning(
                    "hasdata_rate_limited",
                    extra={"context": {"page": page, "zip": zipcode}},
                )
                time.sleep(self.pause * 2)
                continue

            # 400: HasData often sends this when they can't fulfill the page.
            # Treat it as "no more usable pages" instead of crashing ingest.
            if resp.status_code == 400:
                logger.error(
                    "hasdata_listing_error",
                    extra={
                        "context": {
                            "status": resp.status_code,
                            "url": resp.url,
                            "body": resp.text[:500],
                        }
                    },
                )
                logger.warning(
                    "hasdata_stop_pagination_on_400",
                    extra={"context": {"page": page, "zip": zipcode}},
                )
                break  # stop paging for this ZIP, keep what we have

            # Other 4xx/5xx are real failures: raise
            if resp.status_code >= 400:
                logger.error(
                    "hasdata_listing_error",
                    extra={
                        "context": {
                            "status": resp.status_code,
                            "url": resp.url,
                            "body": resp.text[:500],
                        }
                    },
                )
                resp.raise_for_status()

            data = resp.json() or {}

            # Try the common containers they use
            raw_props = (
                data.get("results")
                or data.get("listings")
                or data.get("data")
                or data.get("properties")
                or data
            )

            if not raw_props:
                break

            if isinstance(raw_props, dict):
                raw_props = (
                    raw_props.get("results")
                    or raw_props.get("listings")
                    or raw_props.get("items")
                    or []
                )

            if not isinstance(raw_props, list):
                # Unexpected shape
                logger.error(
                    "hasdata_unexpected_shape",
                    extra={"context": {"snippet": str(raw_props)[:400]}},
                )
                break

            for item in raw_props:
                if len(results) >= limit:
                    break
                try:
                    rec = self._normalize_listing(item)
                    results.append(rec)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "hasdata_normalize_failed",
                        extra={
                            "context": {
                                "error": str(exc),
                                "snippet": str(item)[:400],
                            }
                        },
                    )

            page += 1
            time.sleep(self.pause)

        logger.info(
            "hasdata_zillow_search_complete",
            extra={"context": {"zip": zipcode, "count": len(results)}},
        )
        return results


    # --------- internal helpers ---------

    def _normalize_listing(self, raw: dict[str, Any]) -> PropertyRecord:
        """
        Map HasData listing payload -> PropertyRecord.

        Adjust keys here if your actual response differs
        (inspect resp.text once to confirm).
        """

        # ID
        zpid = (
            raw.get("zpid")
            or raw.get("id")
            or raw.get("propertyId")
            or raw.get("zillowId")
        )

        # Address
        address_obj = raw.get("address") or {}
        if isinstance(address_obj, str):
            address_str = address_obj
            city = str(raw.get("city", ""))
            state = str(raw.get("state", ""))
            zipcode = str(raw.get("zipcode") or raw.get("zip") or "")
        else:
            address_str = (
                address_obj.get("streetAddress")
                or address_obj.get("line")
                or address_obj.get("street")
                or ""
            )
            city = str(address_obj.get("city", raw.get("city", "")))
            state = str(address_obj.get("state", raw.get("state", "")))
            zipcode = str(
                address_obj.get("zipcode")
                or address_obj.get("postalCode")
                or raw.get("zipcode")
                or raw.get("zip")
                or ""
            )

        # Geo
        lat = (
            raw.get("latitude")
            or raw.get("lat")
            or raw.get("coordinates", {}).get("lat")
        )
        lon = (
            raw.get("longitude")
            or raw.get("lon")
            or raw.get("coordinates", {}).get("lng")
            or raw.get("coordinates", {}).get("lon")
        )

        # Physical
        beds = raw.get("bedrooms") or raw.get("beds")
        baths = raw.get("bathrooms") or raw.get("baths")
        sqft = raw.get("livingArea") or raw.get("area") or raw.get("sqft")
        year_built = raw.get("yearBuilt") or raw.get("year_built")

        # Price
        price = (
            raw.get("price")
            or raw.get("unformattedPrice")
            or raw.get("listPrice")
            or raw.get("list_price")
        )

        # Type
        # Property type
        ptype_raw = (
            raw.get("homeType")
            or raw.get("propertyType")
            or raw.get("type")
            or ""
        )

        # --- Normalize external Zillow/HasData types to internal literals ---
        ptype_raw = str(ptype_raw).lower()
        if "condo" in ptype_raw or "town" in ptype_raw:
            ptype = "condo_townhome"
        elif "duplex" in ptype_raw or "triplex" in ptype_raw or "fourplex" in ptype_raw:
            ptype = "duplex_4plex"
        elif "apart" in ptype_raw:
            ptype = "apartment_unit"
        elif "multi" in ptype_raw or "plex" in ptype_raw:
            ptype = "apartment_complex"
        else:
            ptype = "single_family"


        # List date (best-effort string)
        list_date = (
            raw.get("listedAt")
            or raw.get("listDate")
            or raw.get("listingDate")
            or ""
        )

        return PropertyRecord(
            external_id=str(zpid) if zpid is not None else "",
            source=self.source_name,
            address=str(address_str),
            city=city,
            state=state,
            zipcode=zipcode,
            lat=float(lat) if lat is not None else 0.0,
            lon=float(lon) if lon is not None else 0.0,
            beds=float(beds) if beds is not None else 0.0,
            baths=float(baths) if baths is not None else 0.0,
            sqft=float(sqft) if sqft is not None else 0.0,
            year_built=int(year_built) if year_built is not None else 0,
            list_price=float(price) if price is not None else 0.0,
            list_date=str(list_date) if list_date else "",
            property_type=str(ptype),
            raw=raw,
        )
