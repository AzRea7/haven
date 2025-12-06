# entrypoints/cli/fetch_sold_from_api.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from haven.adapters.comps_api import SoldCompsAPIClient
from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


def parse_zip_list(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.replace(",", " ").split()]
    return [p for p in parts if p]


def main():
    ap = argparse.ArgumentParser(description="Fetch sold comps from ATTOM API.")
    ap.add_argument("--zips", required=True, help='e.g. "48009 48363"')
    ap.add_argument("--days-back", type=int, default=365)
    ap.add_argument("--max-records-per-zip", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("data/raw/sold_properties.parquet"))
    args = ap.parse_args()

    zips = parse_zip_list(args.zips)

    client = SoldCompsAPIClient()

    frames = []
    for z in zips:
        df = client.fetch_sold_by_zip(
            zipcode=z,
            days_back=args.days_back,
            max_records=args.max_records_per_zip,
        )
        if not df.empty:
            frames.append(df)
        else:
            logger.info("no_sold_found", extra={"zip": z})

    if not frames:
        raise SystemExit("No sold comps found for any ZIP.")

    out_df = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    print(f"Wrote sold comps → {args.out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
