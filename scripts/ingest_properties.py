# scripts/ingest_properties.py
from __future__ import annotations

import argparse
import sys

from haven.adapters.config import config
from haven.adapters.rentcast_client import RentCastClient
from haven.adapters.rentcast_source import RentCastSaleListingSource
from haven.adapters.sql_repo import SqlPropertyRepository


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest sale listings from RentCast API into core.properties"
    )
    p.add_argument("--zip", dest="zips", nargs="+", required=True)
    p.add_argument("--max-price", dest="max_price", type=float, default=None)
    p.add_argument("--limit", dest="limit", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not config.RENTCAST_API_KEY:
        raise SystemExit(
            "Missing HAVEN_RENTCAST_API_KEY. "
            "Set it in your environment before ingesting."
        )

    repo = SqlPropertyRepository(config.DB_URI)

    client = RentCastClient(
        base_url=config.RENTCAST_BASE_URL,
        api_key=config.RENTCAST_API_KEY,
    )
    source = RentCastSaleListingSource(client=client)

    total = 0
    for z in args.zips:
        props = source.fetch_by_zip(
            zipcode=z,
            limit=args.limit,
            max_price=args.max_price,
            offset=0,
        )
        written = repo.upsert_many(props)
        print(f"[{z}] upserted {written} properties from RentCast")
        total += written

    print(f"Done. Total properties upserted: {total}")


if __name__ == "__main__":
    main()
