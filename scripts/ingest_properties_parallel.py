# scripts/ingest_properties_parallel.py
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from haven.adapters.config import config
from haven.adapters.rentcast_client import RentCastClient
from haven.adapters.rentcast_source import RentCastSaleListingSource
from haven.adapters.sql_repo import SqlPropertyRepository


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel ingest sale listings from RentCast API into core.properties"
    )
    p.add_argument("--zip", dest="zips", nargs="+", required=True)
    p.add_argument("--max-price", dest="max_price", type=float, default=None)
    p.add_argument("--limit", dest="limit", type=int, default=200)
    p.add_argument("--workers", dest="workers", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not config.RENTCAST_API_KEY:
        raise SystemExit("Missing HAVEN_RENTCAST_API_KEY")

    repo = SqlPropertyRepository(config.DB_URI)

    client = RentCastClient(
        base_url=config.RENTCAST_BASE_URL,
        api_key=config.RENTCAST_API_KEY,
    )
    source = RentCastSaleListingSource(client=client)

    def ingest_zip(z: str) -> tuple[str, int]:
        props = source.fetch_by_zip(
            zipcode=z,
            limit=args.limit,
            max_price=args.max_price,
            offset=0,
        )
        n = repo.upsert_many(props)
        return z, n

    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(ingest_zip, z) for z in args.zips]
        for f in as_completed(futs):
            z, n = f.result()
            print(f"[{z}] upserted {n} properties")
            total += n

    print(f"Done. Total properties upserted: {total}")


if __name__ == "__main__":
    main()
