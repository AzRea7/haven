# entrypoints/cli/add_neighborhood_to_rent_training.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import argparse

from haven.adapters.logging_utils import get_logger
from haven.adapters.storage import read_df, write_df
from haven.services.features import attach_neighborhood_quality

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        required=True,
        help="Existing rent_training parquet file (or any feature frame)",
    )
    ap.add_argument(
        "--neighborhood",
        required=True,
        help="Neighborhood quality file (CSV or Parquet) with columns [zip or zipcode, walk_score, school_score, crime_index, rent_demand_index]",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output file path (Parquet) for enriched feature frame",
    )
    ap.add_argument(
        "--on",
        default="zip",
        help="Join key column name (default: zip). Use 'zipcode' if your base has that instead.",
    )
    args = ap.parse_args()

    log.info("Reading base feature frame from %s", args.base)
    base = read_df(args.base)

    log.info("Reading neighborhood data from %s", args.neighborhood)
    nb = read_df(args.neighborhood)

    join_key = args.on

    # If user asked for 'zip' but base only has 'zipcode', auto-map.
    if join_key not in base.columns:
        if join_key == "zip" and "zipcode" in base.columns:
            log.info("Base has 'zipcode' instead of 'zip'; renaming for join.")
            base = base.rename(columns={"zipcode": "zip"})
        else:
            raise SystemExit(
                f"Join key '{join_key}' not found in base columns: {list(base.columns)}"
            )

    if join_key not in nb.columns:
        raise SystemExit(
            f"Join key '{join_key}' not found in neighborhood columns: {list(nb.columns)}"
        )

    log.info("Attaching neighborhood quality on key '%s'", join_key)
    enriched = attach_neighborhood_quality(base, nb, on=join_key)

    log.info("Writing enriched frame to %s (rows=%d)", args.out, len(enriched))
    write_df(enriched, args.out)
    print(f"wrote enriched rent training -> {args.out} rows={len(enriched)}")


if __name__ == "__main__":
    main()
