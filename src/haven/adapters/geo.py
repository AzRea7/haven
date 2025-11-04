# src/haven/adapters/geo.py
from __future__ import annotations
import numpy as np
import pandas as pd

EARTH_R_MI = 3958.8

def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2.0 * EARTH_R_MI * np.arcsin(np.sqrt(a))

def _label_for_ring(r: float) -> str:
    # 0.5 -> "050", 1.0 -> "100", 1.5 -> "150"
    return f"{int(round(r * 100)):03d}"

def compute_ring_features(
    subjects: pd.DataFrame,
    comps: pd.DataFrame,
    rings: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> pd.DataFrame:
    """
    For each subject (lat/lon), compute ring-based medians within successive
    distance intervals (prev, r], for r in `rings` miles.

    Appends columns (per ring):
      ringXXX_psf_med, ringXXX_dom_med, ringXXX_sale_to_list_med,
      ringXXX_price_cuts_p, ringXXX_mos

    Expected comp columns (best-effort; NaN if missing):
      lat, lon, sqft, list_price, sold_price, dom, price_cut, sold_date|close_date
    """
    req_subj_cols = {"lat", "lon"}
    if not req_subj_cols.issubset(subjects.columns):
        missing = req_subj_cols - set(subjects.columns)
        raise ValueError(f"subjects missing required columns: {missing}")

    for col in ["lat", "lon"]:
        if col not in comps.columns:
            raise ValueError(f"comps missing '{col}'")

    # Ensure rings are sorted and unique
    rings = tuple(sorted(set(float(r) for r in rings)))
    if len(rings) == 0:
        # nothing to compute; return subjects unchanged
        return subjects.copy()

    subj_xy = subjects[["lat", "lon"]].to_numpy()
    comp_xy = comps[["lat", "lon"]].to_numpy()

    # Robust price-per-sqft: prefer sold_price, fallback to list_price
    price = comps["sold_price"].where(comps.get("sold_price").notna(), comps.get("list_price"))
    sqft = comps.get("sqft")
    psf = price / sqft.clip(lower=300) if sqft is not None else None

    sale_to_list = None
    if "sold_price" in comps.columns and "list_price" in comps.columns:
        sale_to_list = comps["sold_price"] / comps["list_price"]

    price_cuts = comps.get("price_cut")
    dom = comps.get("dom")

    # sold recency 90 days, for months-of-supply calculation
    sold_dt_col = "sold_date" if "sold_date" in comps.columns else ("close_date" if "close_date" in comps.columns else None)
    if sold_dt_col is not None:
        # parse to datetime
        close_date = pd.to_datetime(comps[sold_dt_col], errors="coerce")

        # make BOTH sides tz-naive
        # (alternatively: close_date = pd.to_datetime(..., utc=True).dt.tz_convert(None))
        now = pd.Timestamp.utcnow().tz_localize(None).normalize()
        if getattr(close_date, "dt", None) is not None:
            close_date = close_date.dt.tz_localize(None)

        sold_recent = (now - close_date).dt.days <= 90
    else:
        sold_recent = pd.Series(False, index=comps.index)


    is_active = comps["sold_price"].isna() if "sold_price" in comps.columns else pd.Series(False, index=comps.index)

    rows = []
    for (slat, slon) in subj_xy:
        dists = haversine(slat, slon, comp_xy[:, 0], comp_xy[:, 1])

        feats = {}
        prev = 0.0
        for r in rings:
            lab = _label_for_ring(r)
            idx = np.where((dists > prev) & (dists <= r))[0]

            if idx.size == 0:
                feats.update({
                    f"ring{lab}_psf_med": np.nan,
                    f"ring{lab}_dom_med": np.nan,
                    f"ring{lab}_sale_to_list_med": np.nan,
                    f"ring{lab}_price_cuts_p": np.nan,
                    f"ring{lab}_mos": np.nan,
                })
                prev = r
                continue

            if psf is not None:
                feats[f"ring{lab}_psf_med"] = float(np.nanmedian(psf.iloc[idx].to_numpy()))
            else:
                feats[f"ring{lab}_psf_med"] = np.nan

            if dom is not None:
                feats[f"ring{lab}_dom_med"] = float(np.nanmedian(dom.iloc[idx].to_numpy()))
            else:
                feats[f"ring{lab}_dom_med"] = np.nan

            if sale_to_list is not None:
                feats[f"ring{lab}_sale_to_list_med"] = float(np.nanmedian(sale_to_list.iloc[idx].to_numpy()))
            else:
                feats[f"ring{lab}_sale_to_list_med"] = np.nan

            if price_cuts is not None:
                feats[f"ring{lab}_price_cuts_p"] = float(np.nanmean(price_cuts.iloc[idx].to_numpy()))
            else:
                feats[f"ring{lab}_price_cuts_p"] = np.nan

            # MOS proxy: active listings divided by monthly sales (~90d / 3)
            ring_active = int(np.nansum(is_active.iloc[idx].to_numpy()))
            ring_sold90 = int(np.nansum(sold_recent.iloc[idx].to_numpy()))
            monthly_sales = max(ring_sold90 / 3.0, 0.001)
            feats[f"ring{lab}_mos"] = float(ring_active / monthly_sales)

            prev = r

        rows.append(feats)

    ring_df = pd.DataFrame(rows, index=subjects.index)
    return pd.concat([subjects.reset_index(drop=True), ring_df.reset_index(drop=True)], axis=1)
