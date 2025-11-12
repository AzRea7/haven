# scripts/eval_sanity.py
"""
Quick sanity checks for Haven scoring and deal analysis.

Run:
    python scripts/eval_sanity.py

This does two things:
  1) Monotonicity check: for a fixed property, higher price -> worse rank_score.
  2) DB sample check: use real ingested listings from haven.db to see score/label distribution.
"""
from __future__ import annotations

from collections import Counter

from sqlmodel import Session, select

from haven.adapters.sql_repo import SqlPropertyRepository, PropertyRow
from haven.services.deal_analyzer import analyze_deal_with_defaults

DB_URI = "sqlite:///haven.db"


def eval_monotonicity() -> None:
    """
    As list_price increases (everything else fixed), rank_score should decrease (worse).
    """
    base = {
        "address": "Monotonic Test",
        "city": "Testville",
        "state": "MI",
        "zipcode": "48009",
        "property_type": "single_family",
        "sqft": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "strategy": "hold",
    }

    prices = [150_000, 200_000, 250_000, 300_000, 400_000]
    scores: list[tuple[float, float]] = []

    for price in prices:
        payload = dict(base, list_price=float(price))
        res = analyze_deal_with_defaults(payload)
        rank = float(res["score"]["rank_score"])
        scores.append((price, rank))

    print("=== Monotonicity: price vs rank_score ===")
    for price, score in scores:
        print(f"  price={price:8,.0f}  rank_score={score:7.2f}")

    monotone = all(scores[i][1] >= scores[i + 1][1] for i in range(len(scores) - 1))
    print(f"=> Monotone decreasing wrt price? {monotone}\n")


def _get_any_zip(repo: SqlPropertyRepository) -> str | None:
    with Session(repo.engine) as session:
        row = session.exec(
            select(PropertyRow.zipcode).where(PropertyRow.zipcode != "").limit(1)
        ).first()
        return row if row else None


def eval_sample_from_db(limit_per_zip: int = 50) -> None:
    """
    Pull real properties from haven.db and inspect label/score distribution.
    """
    repo = SqlPropertyRepository(uri=DB_URI)
    zipcode = _get_any_zip(repo)

    if not zipcode:
        print("No properties found in DB. Run an ingest script first (e.g. ingest_properties_parallel).")
        return

    props = repo.search(zipcode=zipcode, limit=limit_per_zip)
    if not props:
        print(f"No properties returned for zipcode={zipcode}.")
        return

    labels = Counter()
    scores: list[float] = []

    print(f"=== Evaluating up to {len(props)} properties from DB (zip={zipcode}) ===")
    for p in props:
        payload = {
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

        label = res["score"]["label"]
        rank = float(res["score"]["rank_score"])

        labels[label] += 1
        scores.append(rank)

    total = sum(labels.values())
    print("Label distribution:")
    for label, cnt in labels.items():
        pct = (cnt / total) * 100 if total else 0.0
        print(f"  {label:5}: {cnt:3} ({pct:5.1f}%)")

    if scores:
        print(f"Rank_score range: {min(scores):.1f} .. {max(scores):.1f}")
        print(f"Rank_score mean:  {sum(scores) / len(scores):.2f}")
    print()


def main() -> None:
    eval_monotonicity()
    eval_sample_from_db(limit_per_zip=100)


if __name__ == "__main__":
    main()
