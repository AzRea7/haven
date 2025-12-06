# src/haven/adapters/comps_api.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import math
import pandas as pd
import requests

from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SoldCompsAPISettings:
    """
    Settings for ATTOM sold-comps ingestion.

    If HAVEN_COMPS_API_BASE_URL is not set, we default to ATTOM's sale/snapshot
    endpoint documented here:
      https://api.gateway.attomdata.com/propertyapi/v1.0.0/sale/snapshot
    """
    base_url: str
    api_key: str
    timeout: int = 20
    page_size: int = 200  # ATTOM allows higher pageSize; 200 is a good balance.


class SoldCompsAPIClient:
    """
    ATTOM Sold-Property Ingestion Client.

    Uses /sale/snapshot with:
      - postalcode=<zip>
      - startSaleSearchDate=YYYY/MM/DD
      - endSaleSearchDate=YYYY/MM/DD
      - pageSize=N
      - page=P

    Parses the response into a DataFrame with columns:
      sold_price, sold_date, bedrooms, bathrooms, sqft, year_built,
      zipcode, property_type
    """

    def __init__(self, settings: Optional[SoldCompsAPISettings] = None) -> None:
        if settings is None:
            base_url = os.getenv(
                "HAVEN_COMPS_API_BASE_URL",
                "https://api.gateway.attomdata.com/propertyapi/v1.0.0/sale/snapshot",
            ).strip()
            api_key = os.getenv("HAVEN_COMPS_API_KEY", "").strip()

            if not api_key:
                raise RuntimeError(
                    "HAVEN_COMPS_API_KEY is not set. "
                    "Set it to your ATTOM API key before running sold ingestion."
                )

            settings = SoldCompsAPISettings(base_url=base_url, api_key=api_key)

        self.settings = settings

    # -------------------------------------------------------------------------
    # Low-level HTTP bits
    # -------------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        # ATTOM uses "apikey" header
        return {
            "apikey": self.settings.api_key,
            "Accept": "application/json",
        }

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------
    def _normalize_attom_records(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Map ATTOM sale/snapshot JSON records into our standardized schema.

        From docs:
          - sale amount:     property.sale.amount.saleamt
          - sale date:       property.sale.salesearchdate or saleTransDate
          - beds:            property.building.rooms.beds
          - baths (total):   property.building.rooms.bathsTotal
          - size (sqft):     property.building.size.universalsize
          - year built:      property.summary.yearbuilt
          - zipcode:         property.address.postal1
          - property type:   property.summary.propertyType
        """
        rows: List[Dict[str, Any]] = []

        for rec in records:
            sale = rec.get("sale") or {}
            building = rec.get("building") or {}
            summary = rec.get("summary") or {}
            address = rec.get("address") or {}

            amount = sale.get("amount") or {}
            rooms = (building.get("rooms") or {}) if isinstance(building, dict) else {}
            size = (building.get("size") or {}) if isinstance(building, dict) else {}

            row = {
                "sold_price": amount.get("saleamt"),
                # Prefer salesearchdate; fall back to saleTransDate
                "sold_date": sale.get("salesearchdate") or sale.get("saleTransDate"),

                "bedrooms": rooms.get("beds"),
                "bathrooms": rooms.get("bathsTotal"),
                "sqft": size.get("universalsize"),
                "year_built": summary.get("yearbuilt"),

                "zipcode": address.get("postal1"),
                "property_type": summary.get("propertyType") or summary.get("propclass"),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Basic numeric cleanup
        numeric_cols = ["sold_price", "bedrooms", "bathrooms", "sqft", "year_built"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Date parse
        if "sold_date" in df.columns:
            # ATTOM usually returns YYYY-MM-DD or ISO-ish; let pandas infer
            df["sold_date"] = pd.to_datetime(df["sold_date"], errors="coerce")

        # String cleanup
        if "zipcode" in df.columns:
            df["zipcode"] = df["zipcode"].astype(str).str.strip()

        if "property_type" in df.columns:
            df["property_type"] = df["property_type"].astype(str).str.strip()

        return df

    # -------------------------------------------------------------------------
    # Public API: fetch sold comps by ZIP
    # -------------------------------------------------------------------------
    def fetch_sold_by_zip(
        self,
        zipcode: str,
        days_back: int = 365,
        max_records: int = 2000,
    ) -> pd.DataFrame:
        """
        Fetch up to `max_records` sold comps in the given ZIP over the last `days_back` days.

        Uses ATTOM /sale/snapshot with:
          postalcode=<zip>
          startSaleSearchDate=YYYY/MM/DD
          endSaleSearchDate=YYYY/MM/DD

        Pagination is controlled via:
          page, pageSize
        """
        today = datetime.utcnow().date()
        since = today - timedelta(days=days_back)

        start_date = since.strftime("%Y/%m/%d")
        end_date = today.strftime("%Y/%m/%d")

        logger.info(
            "fetch_sold_by_zip_start",
            extra={
                "zipcode": zipcode,
                "startSaleSearchDate": start_date,
                "endSaleSearchDate": end_date,
                "max_records": max_records,
            },
        )

        all_frames: List[pd.DataFrame] = []
        page = 1
        pulled = 0

        while pulled < max_records:
            remaining = max_records - pulled
            page_size = min(self.settings.page_size, remaining)

            params = {
                "postalcode": zipcode,
                "startSaleSearchDate": start_date,
                "endSaleSearchDate": end_date,
                "pageSize": page_size,
                "page": page,
            }

            logger.info(
                "fetch_sold_page_request",
                extra={"zipcode": zipcode, "page": page, "pageSize": page_size},
            )

            resp = requests.get(
                self.settings.base_url,
                headers=self._headers(),
                params=params,
                timeout=self.settings.timeout,
            )

            # Helpful error context if ATTOM is unhappy
            if not resp.ok:
                logger.error(
                    "attom_fetch_error",
                    extra={
                        "status_code": resp.status_code,
                        "text": resp.text[:500],
                        "params": params,
                    },
                )
                resp.raise_for_status()

            data = resp.json()

            records = data.get("property") or []
            if not records:
                logger.info(
                    "fetch_sold_no_more_records",
                    extra={"zipcode": zipcode, "page": page},
                )
                break

            df_page = self._normalize_attom_records(records)
            if df_page.empty:
                logger.info(
                    "fetch_sold_empty_norm_page",
                    extra={"zipcode": zipcode, "page": page},
                )
                break

            all_frames.append(df_page)
            pulled += len(df_page)
            logger.info(
                "fetch_sold_page_success",
                extra={
                    "zipcode": zipcode,
                    "page": page,
                    "rows": len(df_page),
                    "pulled_total": pulled,
                },
            )

            # Pagination logic: ATTOM status.total with page/pageSize
            status = data.get("status") or {}
            total = status.get("total")
            pagesize_resp = status.get("pagesize") or page_size

            if isinstance(total, int) and total > 0:
                max_pages = math.ceil(total / pagesize_resp)
                if page >= max_pages:
                    break

            page += 1

        if not all_frames:
            logger.info(
                "fetch_sold_by_zip_no_results",
                extra={"zipcode": zipcode},
            )
            return pd.DataFrame()

        out = pd.concat(all_frames, ignore_index=True)
        logger.info(
            "fetch_sold_by_zip_done",
            extra={"zipcode": zipcode, "rows": len(out)},
        )
        return out
