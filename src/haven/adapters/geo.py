from __future__ import annotations
import numpy as np
import pandas as pd

EARTH_R_MI = 3958.8  

def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Vectorized haversine distance (miles).
    lat1/lon1 can be scalars; lat2/lon2 can be arrays (or vice versa).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2.0 * EARTH_R_MI * np.arcsin(np.sqrt(a))


def _ring_bucket(distances: np.ndarray) -> np.ndarray:
    """
    Map each distance to a ring: 0.5, 1.0, 1.5 miles, NaN if >1.5.
    """
    r = np.full_like(distances, np.nan, dtype=float)
    r[distances <= 0.5] = 0.5
    r[(distances > 0.5) & (distances <= 1.0)] = 1.0
    r[(distances > 1.0) & (distances <= 1.5)] = 1.5
    return r


def compute_ring_features(subjects: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
    """
    For each subject (row in `subjects` with columns lat/lon), compute ring-based medians
    from `comps` (forSale + sold universe). Returns subjects with columns appended:

        ring050_psf_med, ring100_psf_med, ring150_psf_med
        ring050_dom_med, ring100_dom_med, ring150_dom_med
        ring050_sale_to_list_med, ring100_sale_to_list_med, ring150_sale_to_list_med
        ring050_price_cuts_p, ring100_price_cuts_p, ring150_price_cuts_p
        ring050_mos, ring100_mos, ring150_mos

    Required comp columns (best-effort; fill NaNs if missing):
        lat, lon, sqft, list_price, sold_price, dom, price_cut, close_date
    """
    req_subj_cols = {"lat","lon"}
    if not req_subj_cols.issubset(subjects.columns):
        missing = req_subj_cols - set(subjects.columns)
        raise ValueError(f"subjects missing required columns: {missing}")

    for col in ["lat","lon"]:
        if col not in comps.columns:
            raise ValueError(f"comps missing '{col}'")

    # Prepare arrays for speed
    subj_xy = subjects[["lat","lon"]].to_numpy()
    comp_xy = comps[["lat","lon"]].to_numpy()

    # Robust psf: prefer sold_price; fallback to list_price
    price = comps["sold_price"].where(comps["sold_price"].notna(), comps.get("list_price"))
    sqft = comps.get("sqft")
    psf = None
    if sqft is not None:
        psf = price / sqft.clip(lower=300)  # clip to cut heavy outliers
    sale_to_list = None
    if "sold_price" in comps.columns and "list_price" in comps.columns:
        sale_to_list = comps["sold_price"] / comps["list_price"]

    price_cuts = comps.get("price_cut")
    dom = comps.get("dom")

    # recent sold window for MOS (~90 days)
    now = pd.Timestamp.utcnow().normalize()
    close_date = pd.to_datetime(comps.get("close_date"), errors="coerce") if "close_date" in comps.columns else None
    sold_recent = (now - close_date).dt.days <= 90 if close_date is not None else pd.Series(False, index=comps.index)
    is_active = comps["sold_price"].isna() if "sold_price" in comps.columns else pd.Series(False, index=comps.index)

    rows = []
    for i, (slat, slon) in enumerate(subj_xy):
        dists = haversine(slat, slon, comp_xy[:,0], comp_xy[:,1])
        rings = _ring_bucket(dists)

        feats = {}
        for key, label in [(0.5, "050"), (1.0, "100"), (1.5, "150")]:
            idx = np.where(rings == key)[0]
            if idx.size == 0:
                feats.update({
                    f"ring{label}_psf_med": np.nan,
                    f"ring{label}_dom_med": np.nan,
                    f"ring{label}_sale_to_list_med": np.nan,
                    f"ring{label}_price_cuts_p": np.nan,
                    f"ring{label}_mos": np.nan
                })
                continue

            if psf is not None:
                feats[f"ring{label}_psf_med"] = float(np.nanmedian(psf.iloc[idx].to_numpy()))
            else:
                feats[f"ring{label}_psf_med"] = np.nan

            if dom is not None:
                feats[f"ring{label}_dom_med"] = float(np.nanmedian(dom.iloc[idx].to_numpy()))
            else:
                feats[f"ring{label}_dom_med"] = np.nan

            if sale_to_list is not None:
                feats[f"ring{label}_sale_to_list_med"] = float(np.nanmedian(sale_to_list.iloc[idx].to_numpy()))
            else:
                feats[f"ring{label}_sale_to_list_med"] = np.nan

            if price_cuts is not None:
                feats[f"ring{label}_price_cuts_p"] = float(np.nanmean(price_cuts.iloc[idx].to_numpy()))
            else:
                feats[f"ring{label}_price_cuts_p"] = np.nan

            # MOS proxy: active listings divided by (monthly sales)
            if is_active is not None and sold_recent is not None:
                ring_active = int(np.nansum(is_active.iloc[idx].to_numpy()))
                ring_sold90 = int(np.nansum(sold_recent.iloc[idx].to_numpy()))
                monthly_sales = max(ring_sold90 / 3.0, 0.001)
                feats[f"ring{label}_mos"] = float(ring_active / monthly_sales)
            else:
                feats[f"ring{label}_mos"] = np.nan

        rows.append(feats)

    ring_df = pd.DataFrame(rows, index=subjects.index)
    return pd.concat([subjects.reset_index(drop=True), ring_df.reset_index(drop=True)], axis=1)
