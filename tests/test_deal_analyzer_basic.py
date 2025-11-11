# tests/test_deal_analyzer_basic.py
from haven.services.deal_analyzer import analyze_deal_with_defaults


def test_analyze_deal_with_defaults_minimal_payload():
    payload = {
        "address": "123 Test St",
        "city": "Birmingham",
        "state": "MI",
        "zipcode": "48009",
        "list_price": 250000,
        "sqft": 1200,
        "property_type": "single_family",
    }

    result = analyze_deal_with_defaults(payload)

    # Structure checks
    assert "address" in result
    assert "finance" in result
    assert "pricing" in result
    assert "score" in result
    assert "score_legacy" in result

    score = result["score"]
    assert "rank_score" in score
    assert "label" in score
    assert "reason" in score

    # Legacy remains present
    legacy = result["score_legacy"]
    assert "label" in legacy
    assert "cash_on_cash_return" in legacy
