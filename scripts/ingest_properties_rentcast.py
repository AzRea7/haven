# entrypoints/cli/ingest_properties_rentcast.py
import argparse

from haven.adapters.config import config
from haven.adapters.sql_repo import SqlPropertyRepository
from haven.adapters.rentcast_listings import RentCastSaleListingSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest sale listings from RentCast into core.properties"
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
        help="Internal property type labels to include",
    )
    parser.add_argument(
        "--max-price",
        dest="max_price",
        type=float,
        default=None,
        help="Optional max list price filter",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=300,
        help="Max listings per ZIP to store",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    src = RentCastSaleListingSource()
    repo = SqlPropertyRepository(uri=config.DB_URI)

    total = 0
    for z in args.zips:
        props = src.search(
            zipcode=z,
            property_types=args.types,
            max_price=args.max_price,
            limit=args.limit,
        )
        written = repo.upsert_many(props)
        print(f"[{z}] upserted {written} properties (RentCast)")
        total += written

    print(f"Done. Total properties upserted: {total}")


if __name__ == "__main__":
    main()
