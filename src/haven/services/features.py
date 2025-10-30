from __future__ import annotations
import pandas as pd
import numpy as np

# single source of truth
FEATURES = [
    "beds","baths","sqft","year_built","zip","psf",
    "zhvi_chg_3m","zhvi_chg_6m","zhvi_chg_12m",
    "zori_chg_3m","zori_chg_6m","zori_chg_12m",
    "ring050_psf_med","ring100_psf_med","ring150_psf_med",
    "ring050_dom_med","ring100_dom_med","ring150_dom_med",
    "ring050_sale_to_list_med","ring100_sale_to_list_med","ring150_sale_to_list_med",
    "ring050_price_cuts_p","ring100_price_cuts_p","ring150_price_cuts_p",
    "ring050_mos","ring100_mos","ring150_mos",
]

REQUIRED_SOLD = [
    "property_id","lat","lon","zip","beds","baths","sqft","year_built",
    "list_price","sold_price","sold_date","dom","property_type"
]

def normalize_sold(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in REQUIRED_SOLD if c in df.columns]
    out = df[cols].copy()
    if "sold_date" in out.columns:
        out["sold_date"] = pd.to_datetime(out["sold_date"], errors="coerce")
    out = out.dropna(subset=["sold_price","sqft","sold_date","zip"])
    out = out[out["sold_price"] > 0]
    out["sqft"] = out["sqft"].clip(lower=300)
    out["psf"]  = out["sold_price"] / out["sqft"]
    return out

def attach_momentum(base: pd.DataFrame, zip_momentum: pd.DataFrame) -> pd.DataFrame:
    # zip_momentum columns: ['zip','date','zhvi_chg_3m','...','zori_chg_12m'] (one per zip, most recent)
    cols = [c for c in zip_momentum.columns if c != "date"]
    return base.merge(zip_momentum[cols], on="zip", how="left")

def attach_ring_features(subjects: pd.DataFrame, comps: pd.DataFrame,
                         rings=("050","100","150")) -> pd.DataFrame:
    """
    subjects: rows to train/score (must have lat/lon)
    comps: universe with columns used by adapters.geo.compute_ring_features
    """
    from haven.adapters.geo import compute_ring_features
    return compute_ring_features(subjects, comps, rings=rings)

def finalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # ensure all FEATURES exist 
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    return df
