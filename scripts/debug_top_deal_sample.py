# scripts/debug_top_deal_sample.py

from haven.adapters.config import config
from haven.adapters.rent_estimator_lightgbm import LightGBMRentEstimator
from haven.adapters.sql_repo import SqlPropertyRepository
from haven.services.deal_analyzer import analyze_deal
from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


def run() -> None:
    repo = SqlPropertyRepository(config.DB_URI)
    est = LightGBMRentEstimator()

    # Pull a few properties from a couple of zips
    zips = ["48363", "48009"]
    limit = 5

    for z in zips:
        props = repo.search(zipcode=z, max_price=None, limit=limit)
        if not props:
            print(f"No properties found in ZIP {z}; run ingest_properties.py first.")
            continue

        for i, prop in enumerate(props, start=1):
            print("=" * 80)
            print(f"RAW PROPERTY #{i}")
            print(prop)

            raw = prop.get("raw") or {}

            payload = {
                "address": prop["address"],
                "city": prop["city"],
                "state": prop["state"],
                "zipcode": prop["zipcode"],
                "list_price": float(prop["list_price"]),
                "property_type": prop.get("property_type") or "single_family",
                "sqft": float(prop.get("sqft") or 0.0),
                "bedrooms": float(prop.get("beds") or raw.get("beds") or 0.0),
                "bathrooms": float(prop.get("baths") or raw.get("baths") or 0.0),
                "strategy": "hold",
                "days_on_market": float(
                    raw.get("daysOnZillow")
                    or raw.get("dom")
                    or prop.get("dom")
                    or 0.0
                ),
            }

            print("-" * 80)
            print("PAYLOAD USED FOR ANALYSIS:")
            print(payload)
            print("-" * 80)

            result = analyze_deal(payload, rent_estimator=est, repo=None)

            finance = result["finance"]
            score = result["score"]

            print("FINANCE:")
            print(f"  purchase_price        = {finance.get('purchase_price'):.0f}")
            print(f"  gross_rent            = {finance.get('gross_rent_monthly', 0):.0f}")
            print(f"  eff_rent              = {finance.get('effective_rent_monthly', 0):.0f}")
            print(f"  opx_monthly           = {finance.get('operating_expenses_monthly', 0):.0f}")
            print(f"  mortgage              = {finance.get('mortgage_monthly', 0):.0f}")
            print(f"  NOI_monthly           = {finance.get('noi_monthly', 0):.0f}")
            print(f"  cashflow_after_debt   = {finance.get('cashflow_monthly_after_debt', 0):.0f}")
            print(f"  dscr                  = {finance.get('dscr', 0):.2f}")
            print(f"  coc                   = {finance.get('cash_on_cash_return', 0) * 100:.1f}%")
            print(
                f"  breakeven_occ         = {finance.get('breakeven_occupancy_pct', 0) * 100:.1f}%"
            )

            print("SCORE:")
            print(f"  label                 = {score.get('label')}")
            print(f"  rank_score            = {score.get('rank_score'):.1f}")
            print(f"  reason                = {score.get('reason')}")
            print()
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    run()
