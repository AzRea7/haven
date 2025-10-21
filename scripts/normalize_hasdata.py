import glob, json, os, pandas as pd, numpy as np

RAW_GLOB = os.path.join("data", "raw", "hasdata", "listings_*.json")
OUT_CSV = os.path.join("data", "raw", "listings.csv")

def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # HasData patterns: {"results": [...]}, {"data": [...]}, or direct list
    rows = payload.get("results") or payload.get("data") or payload
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        rows = [rows]
    return rows

def safe_normalize(rows):
    """Flatten one page and keep only scalar columns (drop lists/dicts)."""
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, max_level=2)

    # Keep only scalar dtypes (bool, int, float, str, datetime) to avoid unhashables
    keep_cols = []
    for c in df.columns:
        sample = df[c].dropna().head(5).tolist()
        if any(isinstance(x, (list, dict)) for x in sample):
            continue
        keep_cols.append(c)
    return df[keep_cols].copy()

frames = []
files = sorted(glob.glob(RAW_GLOB))
if not files:
    raise SystemExit("No listing JSON found. Run fetch_hasdata_zillow.py first.")

for p in files:
    rows = load_rows(p)
    flat = safe_normalize(rows)
    if not flat.empty:
        frames.append(flat)

if not frames:
    raise SystemExit("Parsed 0 rows from JSON; inspect a raw file to adjust field paths.")

df = pd.concat(frames, ignore_index=True)

# We'll fill the first available source column for each target.
candidates = {
    "id": [
        "zpid",
        "hdpData.homeInfo.zpid",
        "restimateInfo.zpid"
    ],
    "address": [
        "address",
        "unformattedAddress",
        "hdpData.homeInfo.streetAddress"
    ],
    "list_price": [
        "price",
        "hdpData.homeInfo.price",
        "unformattedPrice"
    ],
    "sqft": [
        "livingArea",
        "hdpData.homeInfo.livingArea"
    ],
    "beds": [
        "bedrooms",
        "hdpData.homeInfo.bedrooms"
    ],
    "baths": [
        "bathrooms",
        "hdpData.homeInfo.bathrooms"
    ],
    "zip": [
        "postalCode",
        "hdpData.homeInfo.zipcode"
    ],
    "status": [
        "homeStatus",
        "hdpData.homeInfo.homeStatus"
    ],
    "days_on_zillow": [
        "daysOnZillow",
        "hdpData.homeInfo.daysOnZillow"
    ],
    "zillow_url": [
        "detailUrl",
        "hdpData.homeInfo.detailUrl"
    ],
    "lat": [
        "latitude",
        "hdpData.homeInfo.latitude"
    ],
    "lon": [
        "longitude",
        "hdpData.homeInfo.longitude"
    ],
    "year_built": [
        "yearBuilt",
        "hdpData.homeInfo.yearBuilt"
    ],
}

out = pd.DataFrame(index=df.index)
for target, sources in candidates.items():
    for src in sources:
        if src in df.columns:
            out[target] = df[src]
            break
    if target not in out.columns:
        out[target] = np.nan 

# ---- Basic cleanup & typing ----
to_num = ["list_price", "sqft", "beds", "baths", "lat", "lon", "year_built", "days_on_zillow"]
for c in to_num:
    out[c] = pd.to_numeric(out[c], errors="coerce")

# ZIP as 5-digit string when possible
if "zip" in out.columns:
    out["zip"] = out["zip"].astype(str).str.extract(r"(\d{5})")

# ---- Deduplicate on stable keys ----
if out["id"].notna().any():
    out = out.drop_duplicates(subset=["id"])
elif out["zillow_url"].notna().any():
    out = out.drop_duplicates(subset=["zillow_url"])
else:
    subset = [c for c in ["address", "list_price", "sqft"] if c in out.columns]
    out = out.drop_duplicates(subset=subset) if subset else out.drop_duplicates()

# Keep a sensible column order
cols = ["id","address","list_price","sqft","beds","baths","zip","status",
        "days_on_zillow","zillow_url","lat","lon","year_built"]
cols = [c for c in cols if c in out.columns]
out = out[cols]

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
out.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} with shape {out.shape}")

# Helpful debug: show a few columns we didn't map yet
unmapped = sorted(set(df.columns) - set(sum(candidates.values(), [])))
print(f"Unmapped scalar columns available ({len(unmapped)}):")
print(", ".join(unmapped[:40]), "..." if len(unmapped) > 40 else "")
