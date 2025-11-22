# scripts/debug_rent_estimator.py

from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator
from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


def run() -> None:
    est = LightGBMRentEstimator()

    scenarios = [
        # suburban 3/2
        dict(bedrooms=3, bathrooms=2, sqft=1500, zipcode="48009", property_type="single_family"),
        # small 2/1
        dict(bedrooms=2, bathrooms=1, sqft=900, zipcode="48201", property_type="single_family"),
        # 1/1 apartment
        dict(bedrooms=1, bathrooms=1, sqft=650, zipcode="48202", property_type="apartment_complex"),
        # larger 4/3
        dict(bedrooms=4, bathrooms=3, sqft=2200, zipcode="48363", property_type="single_family"),
    ]

    for s in scenarios:
        rent = est.predict_unit_rent(**s)
        print("is_ready:", getattr(est, "is_ready", None))
        print(f"predicted rent: {rent:.1f}")
        print("=== Rent estimator outputs ===")
        print(
            {
                "bedrooms": s["bedrooms"],
                "bathrooms": s["bathrooms"],
                "sqft": s["sqft"],
                "zipcode": s["zipcode"],
                "property_type": s["property_type"],
            },
            f"-> est_rent = ${rent:,.0f}/mo",
        )
        print()


if __name__ == "__main__":
    run()
