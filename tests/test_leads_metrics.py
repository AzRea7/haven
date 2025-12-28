# tests/test_leads_metrics.py
from haven.adapters.sql_repo import SqlLeadRepository


def test_lead_metrics_precision_and_funnel(tmp_path):
    db = f"sqlite:///{tmp_path}/test.db"
    repo = SqlLeadRepository(db)

    # Insert leads with different scores + statuses
    ids = repo.upsert_many(
        [
            {"org": "default", "property_id": 1, "zipcode": "48009", "address": "1 A St", "city": "B", "state": "MI", "list_price": 200000, "lead_score": 0.95, "status": "appointment", "reasons": ["x"], "features": {}, "source": "rentcast", "external_id": "a"},
            {"org": "default", "property_id": 2, "zipcode": "48009", "address": "2 A St", "city": "B", "state": "MI", "list_price": 210000, "lead_score": 0.70, "status": "new", "reasons": ["x"], "features": {}, "source": "rentcast", "external_id": "b"},
            {"org": "default", "property_id": 3, "zipcode": "48009", "address": "3 A St", "city": "B", "state": "MI", "list_price": 220000, "lead_score": 0.10, "status": "closed_lost", "reasons": ["x"], "features": {}, "source": "rentcast", "external_id": "c"},
        ]
    )
    assert len(ids) == 3

    # Add outreach attempts with cost
    repo.add_event(lead_id=ids[0], org="default", event_type="outreach_attempt", actor="test", data={"cost": 2.0})
    repo.add_event(lead_id=ids[1], org="default", event_type="outreach_attempt", actor="test", data={"cost": 2.0})

    m = repo.metrics(org="default", days=30, k=2)
    assert m["funnel"]["total_leads"] == 3
    # positives are appointment/under_contract/closed_won -> lead 1 is positive
    assert m["rank"]["precision_at_k"] >= 0.5
    assert m["ops"]["total_cost"] == 4.0
