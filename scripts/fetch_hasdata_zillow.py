import os, time, json, math
from urllib.parse import urlencode
import requests
from dotenv import load_dotenv, find_dotenv

# Load .env (HAVEN/.env) and read key
load_dotenv(find_dotenv())
API_KEY = os.getenv("HASDATA_API_KEY")
if not API_KEY:
    raise SystemExit("Missing HASDATA_API_KEY in .env")

BASE = "https://api.hasdata.com/scrape/zillow/listing"
HEADERS = {"x-api-key": API_KEY}

RAW_DIR = "data/raw/hasdata"
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_listings(keyword: str,
                   typ: str = "forSale",
                   price_min: int | None = None,
                   price_max: int | None = None,
                   beds_min: int | None = None,
                   beds_max: int | None = None,
                   max_pages: int = 3,
                   delay_sec: float = 0.6):
    """
    Fetch paginated Zillow search results via HasData LISTING API.
    Saves each page as JSON under data/raw/hasdata/.
    """
    for page in range(1, max_pages + 1):
        params = {"keyword": keyword, "type": typ, "page": page}
        if price_min is not None: params["price.min"] = price_min
        if price_max is not None: params["price.max"] = price_max
        if beds_min  is not None: params["beds.min"]  = beds_min
        if beds_max  is not None: params["beds.max"]  = beds_max

        url = f"{BASE}?{urlencode(params)}"
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code >= 400:
            raise SystemExit(f"HasData LISTING error {r.status_code}: {r.text[:300]}")
        payload = r.json()

        out = os.path.join(RAW_DIR, f"listings_{keyword.replace(' ','_')}_p{page}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"✅ saved {out}")
        time.sleep(delay_sec)

if __name__ == "__main__":
    # Example metro filter (adjust freely)
    fetch_listings(
        keyword="Detroit, MI",
        typ="forSale",
        price_min=150000,
        price_max=600000,
        beds_min=3,
        max_pages=3
    )
