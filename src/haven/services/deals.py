# src/haven/services/deals.py
from __future__ import annotations

from typing import Any, Dict, List

from haven.adapters.sql_repo import SqlPropertyRepository
from haven.services.deal_analyzer import analyze_deal_with_defaults

# Minimum list price to consider as a serious investment candidate.
# This is a business rule: very cheap properties (< $50k) are often
# distressed, oddball, or otherwise outside the typical buy box.
MIN_LIST_PRICE: float = 50_000.0


def get_top_deals_for_zip(
    zipcode: str,
    *,
    limit_properties: int = 200,
    limit_results: int = 50,
    db_uri: str = "sqlite:///haven.db",
) -> List[Dict[str, Any]]:
    """
    Pull properties from DB for a ZIP, score them, and return the best ones.

    - limit_properties: how many raw listings to pull from DB.
    - limit_results: how many top deals to keep after scoring.

    Result items are exactly the dicts returned by analyze_deal_with_defaults,
    sorted by score.rank_score (descending).

    This function now enforces a couple of investor-style filters:
      - Ignores properties below MIN_LIST_PRICE (very cheap oddball deals).
      - Feeds through sqft / beds / baths / year_built where available
        so the scoring engine can apply size and age penalties.
    """
    repo = SqlPropertyRepository(uri=db_uri)

    # Pull a batch of properties for this ZIP.
    props = repo.search(zipcode=zipcode, limit=limit_properties)

    deals: List[Dict[str, Any]] = []

    for p in props:
        # Normalize / coalesce core fields out of the PropertyRecord.
        raw_price = p.get("list_price") or 0.0
        list_price = float(raw_price)

        # Skip ultra-cheap properties that are usually distressed or outside
        # the target buy box. This matches what a human investor would do.
        if list_price < MIN_LIST_PRICE:
            continue

        # Beds / baths can show up as "bedrooms"/"bathrooms" or "beds"/"baths"
        # depending on ingestion. We coalesce them defensively.
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
            # Strategy hints the scoring engine how to interpret risk/return.
            "strategy": "hold",
        }

        # Pass through year_built if we have it so the scoring logic can
        # apply age penalties (old homes => more capex risk).
        if year_built:
            try:
                payload["year_built"] = int(year_built)
            except (TypeError, ValueError):
                # Silently ignore bad year_built; the scoring fallback
                # will just skip the age penalty.
                pass

        # Run the full analysis + scoring using default repo & rent estimator.
        res = analyze_deal_with_defaults(payload)
        deals.append(res)

    # Sort by rank_score descending (higher = better)
    deals.sort(
        key=lambda d: float(d["score"]["rank_score"]),
        reverse=True,
    )

    if limit_results is not None and limit_results > 0:
        deals = deals[:limit_results]

    return deals
