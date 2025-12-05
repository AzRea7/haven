# scripts/rebuild_deals.py
from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import text
from sqlmodel import Session

from haven.adapters.sql_repo import SqlPropertyRepository, SqlDealRepository
from haven.services.deal_analyzer import analyze_deal_with_defaults

DEFAULT_DB_URI = "sqlite:///haven.db"
MIN_LIST_PRICE: float = 50_000.0
LIMIT_PROPERTIES_PER_ZIP: int = 200


def _truncate_deals_table(deal_repo: SqlDealRepository) -> None:
    """
    Hard-reset the 'deals' table so we can rebuild analyses from scratch.

    This is safe because deals are derived artifacts:
      properties + assumptions + models -> analysis.
    """
    with Session(deal_repo.engine) as session:
        session.exec(text("DELETE FROM deals"))
        session.commit()


def _get_all_zipcodes(prop_repo: SqlPropertyRepository) -> List[str]:
    """
    Enumerate all distinct ZIP codes present in the properties table.

    We use a raw SQL DISTINCT query so we don't have to depend on the exact
    SQLModel row class shape.
    """
    zips: List[str] = []

    with Session(prop_repo.engine) as session:
        result = session.exec(
            text("SELECT DISTINCT zipcode FROM properties WHERE zipcode IS NOT NULL")
        )
        for row in result:
            # SQLAlchemy Row: row[0] is the zipcode
            z = row[0]
            if z:
                zips.append(str(z))

    return sorted(set(zips))


def rebuild_deals(
    db_uri: str = DEFAULT_DB_URI,
    min_list_price: float = MIN_LIST_PRICE,
    limit_properties_per_zip: int = LIMIT_PROPERTIES_PER_ZIP,
) -> None:
    """
    Rebuild the deals table using the current rent estimator and scoring logic.

    Steps:
      1) Truncate the 'deals' table.
      2) For each ZIP in the properties table:
         - Load up to `limit_properties_per_zip` properties.
         - Skip ultra-cheap properties (< min_list_price).
         - Build a clean payload and call analyze_deal_with_defaults(),
           which persists the analysis via SqlDealRepository.
    """
    deal_repo = SqlDealRepository(uri=db_uri)
    prop_repo = SqlPropertyRepository(uri=db_uri)

    print(f"Using DB: {db_uri}")
    print("Truncating deals table...")
    _truncate_deals_table(deal_repo)
    print("Done.\n")

    zipcodes = _get_all_zipcodes(prop_repo)
    if not zipcodes:
        print("No ZIP codes found in properties table. Nothing to rebuild.")
        return

    print(f"Found {len(zipcodes)} ZIPs with properties.")
    print(f"Rebuilding deals with min_list_price=${min_list_price:,.0f} ...\n")

    total_props = 0
    total_analyzed = 0

    for zipcode in zipcodes:
        props = prop_repo.search(zipcode=zipcode, limit=limit_properties_per_zip)
        if not props:
            continue

        print(f"ZIP {zipcode}: {len(props)} properties loaded.")

        for p in props:
            raw_price = p.get("list_price") or 0.0
            list_price = float(raw_price)

            # Skip ultra-cheap properties that are usually distressed or outside
            # the target buy box.
            if list_price < min_list_price:
                continue

            bedrooms = p.get("bedrooms")
            if bedrooms is None:
                bedrooms = p.get("beds")

            bathrooms = p.get("bathrooms")
            if bathrooms is None:
                bathrooms = p.get("baths")

            sqft = p.get("sqft")
            year_built = p.get("year_built")

            payload: Dict[str, Any] = {
                "address": p.get("address"),
                "city": p.get("city"),
                "state": p.get("state"),
                "zipcode": p.get("zipcode") or zipcode,
                "list_price": list_price,
                "sqft": float(sqft or 0.0),
                "bedrooms": float(bedrooms or 0.0),
                "bathrooms": float(bathrooms or 0.0),
                "property_type": p.get("property_type") or "single_family",
                "strategy": "hold",
            }

            if year_built:
                try:
                    payload["year_built"] = int(year_built)
                except (TypeError, ValueError):
                    # Ignore bad year_built values
                    pass

            analyze_deal_with_defaults(payload)
            total_analyzed += 1

        total_props += len(props)

    print("\nRebuild complete.")
    print(f"Total properties scanned : {total_props}")
    print(f"Total deals analyzed     : {total_analyzed}")


if __name__ == "__main__":
    rebuild_deals()
