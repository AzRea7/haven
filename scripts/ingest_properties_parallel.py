import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from haven.adapters.config import config
from haven.adapters.sql_repo import SqlPropertyRepository
from haven.adapters.zillow_hasdata import HasDataZillowPropertySource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel ingest of listings from HasData Zillow Listing API"
    )
    parser.add_argument(
        "--zip",
        dest="zips",
        nargs="+",
        required=True,
        help="One or more ZIP codes to ingest",
    )
    parser.add_argument(
        "--types",
        dest="types",
        nargs="*",
        default=[
            "single_family",
            "condo_townhome",
            "duplex_4plex",
            "apartment_unit",
            "apartment_complex",
        ],
    )
    parser.add_argument(
        "--max-price",
        dest="max_price",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=4,
        help="Number of parallel workers (ZIP-level concurrency)",
    )
    parser.add_argument(
        "--metrics-out",
        dest="metrics_out",
        type=str,
        default=None,
        help="Optional JSONL output for per-ZIP metrics",
    )
    return parser.parse_args()


def ingest_one_zip(zipcode: str, types, max_price, limit) -> dict:
    src = HasDataZillowPropertySource()
    repo = SqlPropertyRepository(uri=config.DB_URI)

    t0 = time.perf_counter()
    listings = src.search(
        zipcode=zipcode,
        property_types=types,
        max_price=max_price,
        limit=limit,
    )
    n = repo.upsert_many(listings)
    dt = time.perf_counter() - t0

    return {
        "zip": zipcode,
        "properties": int(n),
        "seconds": float(dt),
        "throughput_props_per_s": float(n / dt) if dt > 0 else 0.0,
    }


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()
    metrics = []

    print(
        f"Starting parallel ingest for {len(args.zips)} ZIPs "
        f"with {args.workers} workers..."
    )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                ingest_one_zip,
                z,
                args.types,
                args.max_price,
                args.limit,
            ): z
            for z in args.zips
        }

        for fut in as_completed(futures):
            m = fut.result()
            metrics.append(m)
            print(
                f"[{m['zip']}] {m['properties']} props in "
                f"{m['seconds']:.2f}s "
                f"({m['throughput_props_per_s']:.1f} props/s)"
            )

    total_time = time.perf_counter() - t_start
    total_props = sum(m["properties"] for m in metrics)

    print("\n=== Parallel Ingest Summary ===")
    print(f"Workers: {args.workers}")
    print(f"ZIPs: {len(args.zips)}")
    print(f"Total properties: {total_props}")
    print(f"Wall-clock time: {total_time:.2f}s")
    if total_time > 0:
        print(f"Overall throughput: {total_props / total_time:.1f} props/s")

    if args.metrics_out:
        import json

        with open(args.metrics_out, "w", encoding="utf-8") as f:
            for m in metrics:
                f.write(json.dumps(m) + "\n")
        print(f"Metrics written to {args.metrics_out}")


if __name__ == "__main__":
    main()
