import glob, json, pandas as pd, numpy as np, os

RAW_GLOB = "data/raw/hasdata/listings_*.json"
OUT_CSV = "data/raw/listings.csv"

def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # HasData commonly returns {"results": [...]} or {"data": [...]}
    rows = payload.get("results") or payload.get("data") or payload
    if isinstance(rows, dict):  # edge case: single object
        rows = [rows]
    return rows

frames = []
for p in glob.glob(RAW_GLOB):
    rows = load_rows(p)
    if not rows: 
        continue
    frames.append(pd.json_normalize(rows))

if not frames:
    raise SystemExit("No listing JSON found. Run fetch_hasdata_zillow.py first.")

df = pd.concat(frames, ignore_index=True).drop_duplicates()

# Map provider fields -> Haven columns (adjust after inspecting a sample file)
colmap = {
    "zpid": "id",
    "hdpData.homeInfo.zpid": "id",
    "address": "address",
    "unformattedAddress": "address",
    "variableData.text": "headline",
    "price": "list_price",
    "hdpData.homeInfo.price": "list_price",
    "livingArea": "sqft",
    "hdpData.homeInfo.livingArea": "sqft",
    "bedrooms": "beds",
    "bathrooms": "baths",
    "postalCode": "zip",
    "hdpData.homeInfo.homeStatus": "status",
    "hdpData.homeInfo.daysOnZillow": "days_on_zillow",
    "detailUrl": "zillow_url",
    "latitude": "lat",
    "longitude": "lon",
    "hdpData.homeInfo.yearBuilt": "year_built",
}
for src, dst in colmap.items():
    if src in df.columns and dst not in df.columns:
        df[dst] = df[src]

# keep only columns we need now; keep provider cols separately if you want
keep = ["id","address","list_price","sqft","beds","baths","zip","status",
        "days_on_zillow","zillow_url","lat","lon","year_built"]
present = [c for c in keep if c in df.columns]
df = df[present].copy()

# basic cleanup
df["list_price"] = pd.to_numeric(df.get("list_price"), errors="coerce")
df["sqft"] = pd.to_numeric(df.get("sqft"), errors="coerce")
df["beds"] = pd.to_numeric(df.get("beds"), errors="coerce")
df["baths"] = pd.to_numeric(df.get("baths"), errors="coerce")
df["zip"] = df.get("zip").astype(str).str.extract(r"(\d{5})")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"✅ wrote {OUT_CSV} with shape {df.shape}")
