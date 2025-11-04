from haven.services.deal_analyzer import analyze_deal_with_defaults

def test_multifamily_end_to_end():
    payload = {
        "property_type": "apartment_complex",
        "address": "456 Elm St",
        "city": "Detroit",
        "state": "MI",
        "zipcode": "48202",
        "list_price": 1200000,
        "down_payment_pct": 0.25,
        "interest_rate_annual": 0.07,
        "loan_term_years": 30,
        "taxes_annual": 18000,
        "insurance_annual": 9000,
        "hoa_monthly": 0,
        "sqft": 15000,
        "units": [
            {"bedrooms": 1, "bathrooms": 1, "sqft": 700,  "market_rent": 950},
            {"bedrooms": 2, "bathrooms": 1, "sqft": 900,  "market_rent": 1200},
            {"bedrooms": 2, "bathrooms": 1, "sqft": 900,  "market_rent": 1200}
        ]
    }

    result = analyze_deal_with_defaults(payload)

    assert "finance" in result
    assert "pricing" in result
    assert "score" in result

    assert result["finance"]["noi_annual"] > 0
    assert result["finance"]["dscr"] > 0
    assert result["score"]["suggestion"] in ["buy", "maybe negotiate", "maybe (low DSCR)", "pass"]
