# debug_rent_estimator.py

from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator

def main() -> None:
    re = LightGBMRentEstimator()

    print("is_ready:", getattr(re, "is_ready", None))

    rent = re.predict_unit_rent(
        bedrooms=3,
        bathrooms=2,
        sqft=1500,
        zipcode="48009",
        property_type="single_family",
    )
    print("predicted rent:", rent)

if __name__ == "__main__":
    main()
