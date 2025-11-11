# tests/test_deal_analyzer_scoring_order.py
from haven.services.deal_analyzer import analyze_deal


class DummyEstimator:
    # Returns high rent to simulate good cashflow.
    def predict_unit_rent(self, **kwargs) -> float:
        return 2500.0


def test_good_deal_scores_higher_than_bad_deal(tmp_path):
    rent_estimator = DummyEstimator()

    base_payload = {
        "address": "123 Base St",
        "city": "Birmingham",
        "state": "MI",
        "zipcode": "48009",
        "property_type": "single_family",
        "sqft": 1200,
    }

    # Bad deal: expensive, low rent implied (but we override via estimator).
    bad = dict(base_payload)
    bad["list_price"] = 600000

    # Good deal: cheaper
    good = dict(base_payload)
    good["list_price"] = 200000

    bad_res = analyze_deal(bad, rent_estimator=rent_estimator, repo=None)
    good_res = analyze_deal(good, rent_estimator=rent_estimator, repo=None)

    bad_score = bad_res["score"]["rank_score"]
    good_score = good_res["score"]["rank_score"]

    assert good_score > bad_score
