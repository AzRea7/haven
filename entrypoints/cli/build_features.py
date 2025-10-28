import argparse, pandas as pd
from haven.adapters.storage import read_df, write_df
from haven.adapters.indices import compute_zip_momentum
from haven.adapters.geo import compute_ring_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Normalized base parquet (sold/listings)")
    ap.add_argument("--zhvi", required=True, help="ZIP index long-form parquet: zip,date,value")
    ap.add_argument("--zori", required=True, help="ZIP rent index long-form parquet: zip,date,value")
    ap.add_argument("--comps", required=True, help="For rings: forSale + sold comps parquet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = read_df(args.base)     
    zhvi = read_df(args.zhvi)
    zori = read_df(args.zori)
    comps = read_df(args.comps)

    zip_mom = compute_zip_momentum(zhvi, zori)
    base = base.merge(zip_mom.drop(columns=["date"]), on="zip", how="left")

    # add ring features for each subject in base using comps universe
    base = compute_ring_features(base, comps)

    write_df(base, args.out)

if __name__ == "__main__":
    main()
