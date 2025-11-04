# tests/test_api_analyze_success.py
def test_analyze_single_family_success(client):
    payload = {
        "property_type": "single_family",
        "address": "123 Main St",
        "city": "Detroit",
        "state": "MI",
        "zipcode": "48201",
        "list_price": 200000,
        "down_payment_pct": 0.20,
        "interest_rate_annual": 0.065,
        "loan_term_years": 30,
        "taxes_annual": 3000,
        "insurance_annual": 1200,
        "hoa_monthly": 0,
        "sqft": 1500,
        "est_market_rent": 1800.0,
    }
    r = client.post("/analyze", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    # basic shape checks
    assert "finance" in data and "pricing" in data and "score" in data
    fin = data["finance"]
    assert fin["noi_annual"] > 0
    assert fin["dscr"] > 0
    assert "cash_on_cash_return" in fin
    assert data["score"]["suggestion"] in {"buy", "maybe negotiate", "maybe (low DSCR)", "pass"}
