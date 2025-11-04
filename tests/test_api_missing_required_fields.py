# tests/test_api_missing_required_fields.py
def test_missing_required_field_returns_400(client):
    bad_payload = {
        # "property_type" is missing on purpose
        "address": "789 Oak",
        "city": "Detroit",
        "state": "MI",
        "zipcode": "48203",
        "list_price": 180000,
        "down_payment_pct": 0.2,
        "interest_rate_annual": 0.065,
        "loan_term_years": 30,
        "taxes_annual": 2800,
        "insurance_annual": 1100,
    }
    r = client.post("/analyze", json=bad_payload)
    # Your endpoint wraps exceptions as 400; validator raises ValueError → 400
    assert r.status_code == 400
    assert "Missing required field" in r.text
