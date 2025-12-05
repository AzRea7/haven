# entrypoints/cli/build_features.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import argparse

from haven.adapters.indices import compute_zip_momentum
from haven.adapters.logging_utils import get_logger
from haven.adapters.storage import read_df, write_df
from haven.services.features import (
    attach_momentum,
    attach_ring_features,
    attach_neighborhood_quality,
    finalize_feature_frame,
)

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base property frame (normalized listings)")
    ap.add_argument("--zhvi", required=True, help="ZHVI time series by zip")
    ap.add_argument("--zori", required=True, help="ZORI time series by zip")
    ap.add_argument("--comps", required=True, help="Sold comps for ring features")
    ap.add_argument(
        "--neighborhood",
        required=False,
        help="Optional CSV/Parquet with columns [zip, walk_score, school_score, crime_index, rent_demand_index]",
    )
    ap.add_argument("--out", required=True, help="Output feature frame")
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

    if args.neighborhood:
        log.info("attaching neighborhood quality from %s", args.neighborhood)
        nb = read_df(args.neighborhood)
        base = attach_neighborhood_quality(base, nb)
    else:
        log.info("no neighborhood quality file provided; using neutral defaults")

    base = finalize_feature_frame(base)

    write_df(base, args.out)
    log.info("wrote feature frame -> %s rows=%d", args.out, len(base))
    print(f"wrote feature frame -> {args.out} rows={len(base)}")


if __name__ == "__main__":
    main()
