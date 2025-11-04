import pandas as pd
from haven.services.features import attach_ring_features

def test_ring_feature_columns_exist_tz_safe():
    subjects = pd.DataFrame([{"lat": 42.3314, "lon": -83.0458}])
    comps = pd.DataFrame([
        {"lat": 42.33, "lon": -83.04, "sqft": 1200, "list_price": 180000, "sold_price": 175000,
         "dom": 20, "price_cut": 0, "sold_date": "2025-08-01T00:00:00Z"},  # tz-aware string
        {"lat": 42.36, "lon": -83.01, "sqft": 950,  "list_price": 150000, "sold_price": None,
         "dom": 35, "price_cut": 1, "sold_date": None},
    ])
    df = attach_ring_features(subjects, comps, rings=("050","100","150"))
    assert any(c.startswith("ring050_") for c in df.columns)
