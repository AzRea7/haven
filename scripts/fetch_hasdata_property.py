# scripts/fetch_hasdata_property.py
import os, json, re, requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
API_KEY = os.getenv("HASDATA_API_KEY")
if not API_KEY:
    raise SystemExit("Missing HASDATA_API_KEY in .env")

HEADERS = {"x-api-key": API_KEY}
BASE = "https://api.hasdata.com/scrape/zillow/property"
RAW_DIR = "data/raw/hasdata"
os.makedirs(RAW_DIR, exist_ok=True)

def safe_slug(url: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", url)
    return slug[:80]

def fetch_property_by_url(zillow_url: str):
    params = {"url": zillow_url}
    r = requests.get(BASE, headers=HEADERS, params=params, timeout=60)
    if r.status_code >= 400:
        raise SystemExit(f"HasData PROPERTY error {r.status_code}: {r.text[:300]}")
    data = r.json()
    out = os.path.join(RAW_DIR, f"property_{safe_slug(zillow_url)}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"✅ saved {out}")
    return data

if __name__ == "__main__":
    # Paste a Zillow property URL to test
    fetch_property_by_url("https://www.zillow.com/homedetails/301-E-79th-St-APT-23S-New-York-NY-10075/31543731_zpid/")
