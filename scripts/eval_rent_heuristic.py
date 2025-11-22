# scripts/eval_rent_heuristic.py

from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator


def run() -> None:
    est = LightGBMRentEstimator()

    zips = ["48009", "48201", "48202", "48363"]
    bed_bath_pairs = [(1, 1), (2, 1), (3, 2), (4, 3)]
    sqfts = [650, 900, 1200, 1500, 2200]

    print("is_ready:", getattr(est, "is_ready", None))
    print("Evaluating rent heuristic across a small grid...\n")

    for z in zips:
        print(f"ZIP {z}")
        for beds, baths in bed_bath_pairs:
            for s in sqfts:
                rent = est.predict_unit_rent(
                    bedrooms=beds,
                    bathrooms=baths,
                    sqft=s,
                    zipcode=z,
                    property_type="single_family",
                )
                print(
                    f"  {beds}bd/{baths}ba, {s} sqft -> est rent ${rent:,.0f}/mo"
                )
        print()


if __name__ == "__main__":
    run()
