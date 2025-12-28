# src/haven/services/lead_engine.py
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

from haven.domain.property import Property


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class LeadCandidate:
    org: str
    property_id: int
    zipcode: str
    source: str
    external_id: str | None

    address: str
    city: str
    state: str
    lat: float | None
    lon: float | None

    list_price: float
    beds: float | None
    baths: float | None
    sqft: float | None

    lead_score: float
    reasons: list[str]
    features: dict[str, Any]
    status: str = "new"


def score_lead_from_analysis(
    *,
    prop: Property,
    property_id: int,
    org: str,
    analysis: dict[str, Any],
    source: str,
    external_id: str | None,
) -> LeadCandidate:
    pricing = analysis.get("pricing", {}) or {}
    finance = analysis.get("finance", {}) or {}

    # core signals already in your engine
    price_delta_pct = float(pricing.get("price_delta_pct") or 0.0)  # ask - fair / fair
    discount = max(0.0, -price_delta_pct)  # only care when underpriced
    dom = float(finance.get("days_on_market") or analysis.get("days_on_market") or 0.0)
    dscr = float(finance.get("dscr") or 0.0)
    coc = float(finance.get("cash_on_cash_return") or 0.0)

    raw = getattr(prop, "raw", {}) or {}
    raw_source = str(raw.get("source") or source or "").lower()
    is_auction = "auction" in raw_source
    is_tax = "tax" in raw_source or "treasur" in raw_source
    is_foreclosure = "foreclos" in raw_source
    distress = 1.0 if (is_auction or is_tax or is_foreclosure) else 0.0

    # scoring: interpret as "probability this lead is worth working"
    x = 0.0
    x += 3.5 * discount
    x += 0.018 * dom
    x += 1.3 * distress
    x += 0.7 * max(0.0, dscr - 1.0)
    x += 4.5 * max(0.0, coc)

    lead_score = float(_sigmoid(x))

    reasons: list[str] = []
    if discount >= 0.10:
        reasons.append(f"Underpriced vs model (~{discount*100:.0f}% discount)")
    elif discount >= 0.05:
        reasons.append(f"Discount vs model (~{discount*100:.0f}%)")

    if dom >= 45:
        reasons.append(f"Stale listing ({dom:.0f} DOM)")
    elif dom >= 21:
        reasons.append(f"Longer DOM ({dom:.0f})")

    if distress > 0:
        if is_tax:
            reasons.append("Distress signal: tax-related")
        if is_foreclosure:
            reasons.append("Distress signal: foreclosure")
        if is_auction:
            reasons.append("Distress signal: auction")

    if not reasons:
        reasons.append("Workable lead; monitor and outreach")

    features = {
        "price_delta_pct": price_delta_pct,
        "discount_pct": discount,
        "dom": dom,
        "dscr": dscr,
        "cash_on_cash_return": coc,
        "distress": distress,
        "is_auction": is_auction,
        "is_tax": is_tax,
        "is_foreclosure": is_foreclosure,
        "strategy": str(analysis.get("strategy") or ""),
    }

    addr = analysis.get("address", {}) or {}
    return LeadCandidate(
        org=org,
        property_id=property_id,
        zipcode=str(addr.get("zipcode") or prop.zipcode),
        source=source,
        external_id=external_id,
        address=str(addr.get("address") or prop.address),
        city=str(addr.get("city") or prop.city),
        state=str(addr.get("state") or prop.state),
        lat=getattr(prop, "lat", None),
        lon=getattr(prop, "lon", None),
        list_price=float(analysis.get("list_price") or finance.get("purchase_price") or getattr(prop, "list_price", 0.0) or 0.0),
        beds=float(getattr(prop, "bedrooms", None) or getattr(prop, "beds", None) or 0.0) if (getattr(prop, "beds", None) is not None) else None,
        baths=float(getattr(prop, "bathrooms", None) or getattr(prop, "baths", None) or 0.0) if (getattr(prop, "baths", None) is not None) else None,
        sqft=float(getattr(prop, "sqft", None) or 0.0) if (getattr(prop, "sqft", None) is not None) else None,
        lead_score=lead_score,
        reasons=reasons,
        features=features,
        status="new",
    )
