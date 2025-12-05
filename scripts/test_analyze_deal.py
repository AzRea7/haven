# scripts/test_analyze_deal.py
from haven.services.deal_analyzer import analyze_deal
from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator


def main() -> None:
    est = LightGBMRentEstimator()

    payload = {
        "address": "Test Property",
        "city": "Birmingham",
        "state": "MI",
        "zipcode": "48009",
        "list_price": 300000,
        "sqft": 1400,
        "bedrooms": 3,
        "bathrooms": 2,
        "property_type": "single_family",
        "strategy": "hold",
    }

    res = analyze_deal(payload, rent_estimator=est, repo=None)

    print("=== DEAL ANALYSIS ===")
    print("Label:", res.get("label"))
    print("Reason:", res.get("reason"))

    fin = res.get("finance", {})
    print("\n--- Finance ---")
    for k in sorted(fin.keys()):
        print(f"{k:25s}: {fin[k]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
