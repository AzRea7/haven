# src/haven/services/features.py
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

FEATURES = [
    "beds","baths","sqft","year_built","zip","psf",
    "zhvi_chg_3m","zhvi_chg_6m","zhvi_chg_12m",
    "zori_chg_3m","zori_chg_6m","zori_chg_12m",
    "ring050_psf_med","ring100_psf_med","ring150_psf_med",
    "ring050_dom_med","ring100_dom_med","ring150_dom_med",
    "ring050_sale_to_list_med","ring100_sale_to_list_med","ring150_sale_to_list_med",
    "ring050_price_cuts_p","ring100_price_cuts_p","ring150_price_cuts_p",
    "ring050_mos","ring100_mos","ring150_mos", "walk_score",          # 0–100
    "school_score",      
    "crime_index",        
    "rent_demand_index",
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
    cols = [c for c in zip_momentum.columns if c != "date"]
    return base.merge(zip_momentum[cols], on="zip", how="left")

def _normalize_rings(rings: Iterable[float | str] | None) -> tuple[float, ...]:
    if rings is None:
        return (0.5, 1.0, 1.5)
    out = []
    for r in rings:
        if isinstance(r, str):
            # accept "050" / "100" / "150" → 0.5 / 1.0 / 1.5
            out.append(float(r) / 100.0)
        else:
            out.append(float(r))
    # ensure sorted unique
    return tuple(sorted(set(out)))

def attach_ring_features(
    subjects: pd.DataFrame,
    comps: pd.DataFrame,
    rings: Iterable[float | str] = ("050", "100", "150"),
) -> pd.DataFrame:
    """
    subjects: rows to train/score (must have lat/lon)
    comps: universe with columns used by adapters.geo.compute_ring_features
    rings: iterable of rings (either strings like '050' or floats in miles)
    """
    from haven.adapters.geo import compute_ring_features
    ring_distances = _normalize_rings(rings)
    return compute_ring_features(subjects, comps, rings=ring_distances)

def finalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    return df

def attach_neighborhood_quality(
    base: pd.DataFrame,
    neighborhoods: pd.DataFrame,
    on: str = "zip",
) -> pd.DataFrame:
    """
    Join in neighborhood-quality metrics (walkability, schools, crime, rent demand)
    by ZIP (or other key).

    neighborhoods must have at least:
      - a join key column (default 'zip')
      - some subset of:
          'walk_score', 'school_score', 'crime_index', 'rent_demand_index'
    """
    cols = ["walk_score", "school_score", "crime_index", "rent_demand_index"]

    # If base is missing the join key entirely, just add empty columns and bail.
    if on not in base.columns:
        for c in cols:
            if c not in base.columns:
                base[c] = 0.0
        return base

    # Limit to columns that actually exist in the provided neighborhoods frame
    present = [c for c in cols if c in neighborhoods.columns]
    if not present:
        # Nothing to join; ensure columns exist on base and return
        for c in cols:
            if c not in base.columns:
                base[c] = 0.0
        return base

    # Work on copies so we don't mutate callers' DataFrames unexpectedly
    base = base.copy()
    nb = neighborhoods.copy()

    # Ensure join key exists in neighborhood frame
    if on not in nb.columns:
        # If the neighborhoods file used 'zipcode' while base used 'zip', try that
        if on == "zip" and "zipcode" in nb.columns:
            nb = nb.rename(columns={"zipcode": "zip"})
        else:
            # Can't join; just add default columns and return
            for c in cols:
                if c not in base.columns:
                    base[c] = 0.0
            return base

    # Normalize join key dtype: convert both to string (and zero-pad 5 digits for ZIPs)
    base[on] = base[on].astype(str).str.zfill(5)
    nb[on] = nb[on].astype(str).str.zfill(5)

    join_cols = [on] + present
    nb = nb[join_cols].drop_duplicates(on)

    merged = base.merge(nb, how="left", on=on, suffixes=("", "_nbhd"))

    # Fill any NAs with neutral-ish defaults
    for c in cols:
        if c not in merged.columns:
            merged[c] = 0.0
        else:
            merged[c] = merged[c].fillna(0.0)

    return merged

