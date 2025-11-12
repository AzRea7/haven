from __future__ import annotations

from typing import Any, Dict, List

from sqlmodel import Session, select

from haven.adapters.sql_repo import SqlPropertyRepository, PropertyRow
from haven.services.deal_analyzer import analyze_deal_with_defaults


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
    """
    repo = SqlPropertyRepository(uri=db_uri)

    # Use repo.search if you already have it wired;
    # alternatively we can do a direct SQLModel query.
    props = repo.search(zipcode=zipcode, limit=limit_properties)

    deals: List[Dict[str, Any]] = []

    for p in props:
        payload: Dict[str, Any] = {
            "address": p.get("address"),
            "city": p.get("city"),
            "state": p.get("state"),
            "zipcode": p.get("zipcode") or zipcode,
            "list_price": float(p.get("list_price") or 0.0),
            "sqft": float(p.get("sqft") or 0.0),
            "bedrooms": float(p.get("bedrooms") or 0.0),
            "bathrooms": float(p.get("bathrooms") or 0.0),
            "property_type": p.get("property_type") or "single_family",
            "strategy": "hold",
        }

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
