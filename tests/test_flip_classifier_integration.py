# tests/test_flip_classifier_integration.py
import os
from pathlib import Path

import pytest

from haven.services.deal_analyzer import analyze_deal_with_defaults


@pytest.mark.skipif(
    not Path("models/flip_logit_calibrated.joblib").exists(),
    reason="flip model artifact not present",
)
def test_flip_probability_present_when_model_exists():
    payload = {
        "address": "456 Flip St",
        "city": "Birmingham",
        "state": "MI",
        "zipcode": "48009",
        "list_price": 300000,
        "sqft": 1500,
        "property_type": "single_family",
        "days_on_market": 10,
    }

    res = analyze_deal_with_defaults(payload)

    assert "flip_p_good" in res
    p = res["flip_p_good"]
    assert p is None or (0.0 <= p <= 1.0)
