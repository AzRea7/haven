import argparse, pandas as pd
from pathlib import Path

SELL_COLS = ["property_id","lat","lon","zip","beds","baths","sqft","year_built","list_price",
             "sold_price","sold_date","dom","property_type"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    # keep & clean
    df = df[SELL_COLS].copy()
    df["sold_date"] = pd.to_datetime(df["sold_date"], errors="coerce")
    df = df.dropna(subset=["sold_price","sqft","sold_date","zip"])
    df = df[df["sold_price"] > 0]
    df["psf"] = df["sold_price"] / df["sqft"].clip(lower=300)  # floor sqft to reduce outliers
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

if __name__ == "__main__":
    main()
