# scripts/eval_rent_heuristic.py
"""
Quick check of the current rent estimator.

Run from project root:
    python scripts/eval_rent_heuristic.py
"""

from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator


def main() -> None:
    est = LightGBMRentEstimator()

    samples = [
        dict(bedrooms=3, bathrooms=2, sqft=1500, zipcode="48009", property_type="single_family"),
        dict(bedrooms=2, bathrooms=1, sqft=900, zipcode="48201", property_type="single_family"),
        dict(bedrooms=1, bathrooms=1, sqft=650, zipcode="48202", property_type="apartment_complex"),
        dict(bedrooms=4, bathrooms=3, sqft=2200, zipcode="48363", property_type="single_family"),
    ]

    print("=== Rent estimator outputs ===")
    for s in samples:
        r = est.predict_unit_rent(**s)
        print(f"{s} -> est_rent = ${r:,.0f}/mo")


if __name__ == "__main__":
    main()
