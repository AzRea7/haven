import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import argparse
from haven.adapters.storage import read_df, write_df
from haven.adapters.indices import compute_zip_momentum
from haven.services.features import attach_momentum, attach_ring_features, finalize_feature_frame
from haven.adapters.logging_utils import get_logger

log = get_logger(__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Normalized parquet/csv (sold or listings)")
    ap.add_argument("--zhvi", required=True, help="Long-form ZHVI: zip,date,value")
    ap.add_argument("--zori", required=True, help="Long-form ZORI: zip,date,value")
    ap.add_argument("--comps", required=True, help="For rings: forSale+sold universe parquet/csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log.info("reading inputs")
    base = read_df(args.base)
    zhvi = read_df(args.zhvi)
    zori = read_df(args.zori)
    comps = read_df(args.comps)

    log.info("computing zip momentum")
    zip_mom = compute_zip_momentum(zhvi, zori)

    log.info("attaching momentum + ring features")
    base = attach_momentum(base, zip_mom)
    base = attach_ring_features(base, comps)
    base = finalize_feature_frame(base)

    write_df(base, args.out)
    log.info("wrote feature frame -> %s rows=%d", args.out, len(base))
    print(f"wrote feature frame -> {args.out} rows={len(base)}")

if __name__ == "__main__":
    main()
