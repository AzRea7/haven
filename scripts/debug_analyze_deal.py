from pprint import pprint

from haven.services.deal_analyzer import analyze_deal_with_defaults


def main() -> None:
    payload = {
        "address": "123 Test St",
        "city": "Birmingham",
        "state": "MI",
        "zipcode": "48009",
        "list_price": 250000,
        "sqft": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "property_type": "single_family",
        # optional:
        # "days_on_market": 30,
    }

    result = analyze_deal_with_defaults(payload)

    print("\n=== ADDRESS ===")
    pprint(result["address"])

    print("\n=== FINANCE ===")
    pprint(result["finance"])

    print("\n=== PRICING ===")
    pprint(result["pricing"])

    print("\n=== SCORE (new) ===")
    pprint(result["score"])

    print("\n=== SCORE LEGACY ===")
    pprint(result["score_legacy"])

    print("\nflip_p_good:", result.get("flip_p_good"))


if __name__ == "__main__":
    main()
