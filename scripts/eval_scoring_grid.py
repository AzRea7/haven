# scripts/eval_scoring_grid.py
from __future__ import annotations

from haven.services.deal_analyzer import analyze_deal
from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator


def run() -> None:
    est = LightGBMRentEstimator()

    # Scenarios sweep price and rent. We pass rent explicitly via a Unit to lock the income.
    scenarios = [
        # Clearly good (relative to price)
        dict(price=150_000, rent=1_800, bedrooms=3, bathrooms=2, sqft=1400),
        dict(price=200_000, rent=2_200, bedrooms=3, bathrooms=2, sqft=1400),
        # Marginal
        dict(price=250_000, rent=2_000, bedrooms=3, bathrooms=2, sqft=1400),
        # Bad
        dict(price=400_000, rent=1_800, bedrooms=3, bathrooms=2, sqft=1400),
    ]

    for s in scenarios:
        payload = {
            "address": "Test",
            "city": "Birmingham",
            "state": "MI",
            "zipcode": "48009",
            "list_price": float(s["price"]),
            "property_type": "single_family",

            # Provide a single unit and pin market_rent to the scenario's rent so we
            # evaluate the scoring at *exactly* this income level.
            "units": [
                {
                    "bedrooms": float(s["bedrooms"]),
                    "bathrooms": float(s["bathrooms"]),
                    "sqft": float(s["sqft"]),
                    "market_rent": float(s["rent"]),
                }
            ],

            # Top-level features (kept for completeness; units drive rent and sizing)
            "bedrooms": float(s["bedrooms"]),
            "bathrooms": float(s["bathrooms"]),
            "sqft": float(s["sqft"]),

            "strategy": "hold",
        }

        res = analyze_deal(payload, rent_estimator=est, repo=None)
        fin = res["finance"]
        sc = res["score"]

        print(
            f"price={s['price']:,}, "
            f"est_rent={fin['gross_rent_monthly']:.0f}, "
            f"dscr={fin['dscr']:.2f}, "
            f"coc={fin['cash_on_cash_return']:.2%}, "
            f"label={sc['label']}, "
            f"rank_score={sc.get('rank_score'):.2f}"
        )


if __name__ == "__main__":
    run()
