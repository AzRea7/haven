# tests/test_api_percent_and_string_inputs.py
def test_percent_string_inputs_are_normalized(client):
    # Note the strings and percentages:
    payload = {
        "property_type": "duplex_4plex",
        "address": "456 Elm St",
        "city": "Detroit",
        "state": "MI",
        "zipcode": "48202",
        "list_price": "250000",
        "down_payment_pct": "25%",          # should normalize to 0.25
        "interest_rate_annual": "6.5%",     # should normalize to 0.065
        "loan_term_years": 30,
        "taxes_annual": "3500",
        "insurance_annual": "1500",
        "hoa_monthly": "0",
        "sqft": "2200",
        "units": [
            {"bedrooms": 2, "bathrooms": 1, "sqft": 1100, "market_rent": None},
            {"bedrooms": 2, "bathrooms": 1, "sqft": 1100, "market_rent": None},
        ],
    }
    r = client.post("/analyze", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    fin = data["finance"]
    assert fin["gross_rent_monthly"] > 0  # rent estimator filled missing rents
    assert fin["noi_annual"] > 0
    assert data["score"]["suggestion"] in {"buy", "maybe negotiate", "maybe (low DSCR)", "pass"}
