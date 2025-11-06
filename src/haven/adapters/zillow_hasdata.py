import csv
import datetime as dt
import json
import os
import re
import time
from typing import Any

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

API_KEY = os.getenv("HASDATA_API_KEY")
if not API_KEY:
    raise SystemExit("Missing HASDATA_API_KEY in .env")

HEADERS = {"x-api-key": API_KEY}
BASE = "https://api.hasdata.com/scrape/zillow/property"
RAW_DIR = "data/raw/hasdata"
os.makedirs(RAW_DIR, exist_ok=True)

# valid Zillow homedetails URL pattern
VALID_HOMEDTLS = re.compile(r"^https://www\.zillow\.com/homedetails/.+?_zpid/$", re.IGNORECASE)


def safe_slug(url: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", url)
    return slug[:80]


def is_valid_zillow_homedetails(url: str) -> bool:
    if not isinstance(url, str):
        return False
    u = url.strip()
    if not u or u.startswith("#"):
        return False
    return bool(VALID_HOMEDTLS.match(u))


def _request_with_retries(url: str, params: dict[str, Any], max_retries: int = 4, backoff: float = 1.6) -> dict[str, Any]:
    """
    GET with retry/backoff. Handles 429 with Retry-After when available.
    """
    for attempt in range(1, max_retries + 1):
        r = requests.get(url, headers=HEADERS, params=params, timeout=60)
        # Rate limit handling
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else (backoff ** attempt)
            print(f" 429 rate-limited; sleeping {wait:.1f}s...")
            time.sleep(wait)
            continue

        if r.status_code < 400:
            try:
                return r.json()
            except Exception as e:
                raise RuntimeError(f"Unexpected non-JSON response: {e}\nText: {r.text[:300]}") from e

        # other error
        if attempt == max_retries:
            raise SystemExit(f"HasData error {r.status_code}: {r.text[:300]}")
        time.sleep(backoff ** attempt)

    raise RuntimeError("Unreachable")


def fetch_property_by_url(zillow_url: str) -> dict[str, Any]:
    """
    Fetch a single property's JSON from HasData and persist raw JSON under data/raw/hasdata/.
    Returns the parsed dict.
    """
    params = {"url": zillow_url}
    data = _request_with_retries(BASE, params)
    out = os.path.join(RAW_DIR, f"property_{safe_slug(zillow_url)}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"saved {out}")
    return data


def fetch_properties_from_file(urls_csv: str, out_csv: str, url_col: str = "url", sleep_s: float = 0.5) -> pd.DataFrame:
    """
    Read a CSV of Zillow URLs (column `url_col`), fetch each property, and write a tidy CSV.
    Skips invalid/blank/commented URLs. Logs failures to data/raw/hasdata/errors_<ts>.csv.

    Output columns include a best-effort flat view (zpid, address, price, beds, baths, sqft, etc.) when present.
    Raw JSON for each row is also stored under data/raw/hasdata/.
    """
    df_urls = pd.read_csv(urls_csv)
    if url_col not in df_urls.columns:
        raise ValueError(f"Column `{url_col}` not found in {urls_csv}")

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    tstamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    err_path = os.path.join(RAW_DIR, f"errors_{tstamp}.csv")

    url_list = df_urls[url_col].tolist()
    for i, raw in enumerate(url_list, start=1):
        url = str(raw).strip() if isinstance(raw, str) else ""
        if not is_valid_zillow_homedetails(url):
            errors.append({"row": i, "url": url, "error": "invalid_or_blank_url"})
            print(f"Skipping row {i}: invalid/blank URL")
            continue

        try:
            payload = fetch_property_by_url(url)
        except SystemExit as e:
            msg = str(e)
            print(f"Row {i} failed ({url}): {msg[:120]}")
            errors.append({"row": i, "url": url, "error": msg})
            continue
        except Exception as e:
            msg = repr(e)
            print(f"Row {i} failed ({url}): {msg[:120]}")
            errors.append({"row": i, "url": url, "error": msg})
            continue

        # Flatten common fields if present
        p = payload.get("property", payload)
        rec = {
            "source_url": url,
            "zpid": p.get("zpid"),
            "address": p.get("address") or p.get("streetAddress"),
            "city": p.get("city"),
            "state": p.get("state"),
            "zip": p.get("zipcode") or p.get("zipCode") or p.get("postalCode"),
            "lat": p.get("latitude") or p.get("lat"),
            "lon": p.get("longitude") or p.get("lng"),
            "beds": p.get("bedrooms") or p.get("beds"),
            "baths": p.get("bathrooms") or p.get("baths"),
            "sqft": p.get("livingArea") or p.get("sqft"),
            "list_price": p.get("price") or p.get("listPrice"),
            "sold_price": p.get("soldPrice"),
            "sold_date": p.get("soldDate"),
            "dom": p.get("daysOnZillow") or p.get("daysOnMarket"),
            "property_type": p.get("homeType") or p.get("propertyType"),
        }
        records.append(rec)

        if sleep_s > 0:
            time.sleep(sleep_s)
        if i % 25 == 0:
            print(f"…fetched {i} properties")

    out_df = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"wrote tidy CSV -> {out_csv} (rows={len(out_df)})")

    if errors:
        with open(err_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["row","url","error"])
            w.writeheader()
            w.writerows(errors)
        print(f"wrote error log -> {err_path} (rows={len(errors)})")

    return out_df


def fetch_zillow_dump(kind: str, out_path: str, url_list_path: str | None = None, url_col: str = "url") -> pd.DataFrame:
    """
    Shim used by CLI:
      - kind in {"sold","forSale"} picks default url list at data/raw/zillow_urls_{kind}.csv
      - You may pass a custom `url_list_path` instead.
    """
    if kind not in {"sold", "forSale"}:
        raise ValueError("kind must be one of {'sold','forSale'}")

    if url_list_path is None:
        url_list_path = f"data/raw/zillow_urls_{kind}.csv"
        if not os.path.exists(url_list_path):
            raise FileNotFoundError(
                f"Default url_list_path not found: {url_list_path}\n"
                f"Create a CSV with a `{url_col}` column of Zillow homedetails URLs for {kind} listings "
                f"(must end with '_zpid/')."
            )

    return fetch_properties_from_file(url_list_path, out_path, url_col=url_col)
