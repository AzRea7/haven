# src/haven/adapters/geo.py
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

EARTH_R_MI = 3958.8


def haversine(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> np.ndarray:
    lat1_arr, lon1_arr, lat2_arr, lon2_arr = map(lambda a: np.radians(np.asarray(a, dtype=float)),
                                                 [lat1, lon1, lat2, lon2])
    dlat = lat2_arr - lat1_arr
    dlon = lon2_arr - lon1_arr
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_arr) * np.cos(lat2_arr) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_MI * np.arcsin(np.sqrt(a))


def _label_for_ring(r: float) -> str:
    return f"{round(r * 100):03d}"


def compute_ring_features(
    subjects: pd.DataFrame,
    comps: pd.DataFrame,
    rings: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> pd.DataFrame:
    req_subj_cols = {"lat", "lon"}
    if not req_subj_cols.issubset(subjects.columns):
        missing = req_subj_cols - set(subjects.columns)
        raise ValueError(f"subjects missing required columns: {missing}")

    for col in ["lat", "lon"]:
        if col not in comps.columns:
            raise ValueError(f"comps missing '{col}'")

    rings = tuple(sorted(set(float(r) for r in rings)))
    if len(rings) == 0:
        return subjects.copy()

    subj_xy = subjects[["lat", "lon"]].to_numpy(dtype=float)
    comp_xy = comps[["lat", "lon"]].to_numpy(dtype=float)

    # price = sold where available, else list; if neither present, NaN series
    sold = comps["sold_price"] if "sold_price" in comps.columns else None
    listp = comps["list_price"] if "list_price" in comps.columns else None

    price = sold.where(sold.notna(), listp) if (sold is not None and listp is not None) else sold
    if price is None:
        price = pd.Series(np.nan, index=comps.index)

    sqft = comps["sqft"] if "sqft" in comps.columns else None
    if sqft is not None:
        sqft_clip = pd.to_numeric(sqft, errors="coerce").clip(lower=300)
        psf = pd.to_numeric(price, errors="coerce") / sqft_clip
    else:
        psf = None

    sale_to_list = None
    if "sold_price" in comps.columns and "list_price" in comps.columns:
        sale_to_list = pd.to_numeric(comps["sold_price"], errors="coerce") / pd.to_numeric(comps["list_price"], errors="coerce")

    price_cuts = pd.to_numeric(comps["price_cut"], errors="coerce") if "price_cut" in comps.columns else None
    dom = pd.to_numeric(comps["dom"], errors="coerce") if "dom" in comps.columns else None

    sold_dt_col = "sold_date" if "sold_date" in comps.columns else ("close_date" if "close_date" in comps.columns else None)
    if sold_dt_col is not None:
        # Make tz-naive and compute "recent" (<= 90 days) as a boolean Series
        close_date = pd.to_datetime(comps[sold_dt_col], errors="coerce", utc=True).dt.tz_convert(None)
        now = pd.Timestamp.now(tz=None).normalize()
        sold_recent = (now - close_date).dt.days <= 90
        sold_recent = sold_recent.fillna(False)
    else:
        sold_recent = pd.Series(False, index=comps.index)

    is_active = comps["sold_price"].isna() if "sold_price" in comps.columns else pd.Series(False, index=comps.index)

    rows: list[dict[str, float]] = []
    for (slat, slon) in subj_xy:
        dists = haversine(slat, slon, comp_xy[:, 0], comp_xy[:, 1])

        feats: dict[str, float] = {}
        prev = 0.0
        for r in rings:
            lab = _label_for_ring(r)
            idx = np.where((dists > prev) & (dists <= r))[0]

            if idx.size == 0:
                feats.update({
                    f"ring{lab}_psf_med": float("nan"),
                    f"ring{lab}_dom_med": float("nan"),
                    f"ring{lab}_sale_to_list_med": float("nan"),
                    f"ring{lab}_price_cuts_p": float("nan"),
                    f"ring{lab}_mos": float("nan"),
                })
                prev = r
                continue

            if psf is not None:
                feats[f"ring{lab}_psf_med"] = float(np.nanmedian(psf.to_numpy()[idx]))
            else:
                feats[f"ring{lab}_psf_med"] = float("nan")

            if dom is not None:
                feats[f"ring{lab}_dom_med"] = float(np.nanmedian(dom.to_numpy()[idx]))
            else:
                feats[f"ring{lab}_dom_med"] = float("nan")

            if sale_to_list is not None:
                feats[f"ring{lab}_sale_to_list_med"] = float(np.nanmedian(sale_to_list.to_numpy()[idx]))
            else:
                feats[f"ring{lab}_sale_to_list_med"] = float("nan")

            if price_cuts is not None:
                feats[f"ring{lab}_price_cuts_p"] = float(np.nanmean(price_cuts.to_numpy()[idx]))
            else:
                feats[f"ring{lab}_price_cuts_p"] = float("nan")

            # Pure-Pandas boolean sums (no NumPy on datetime)
            ring_active = int(is_active.iloc[idx].sum())
            ring_sold90 = int(sold_recent.iloc[idx].sum())
            monthly_sales = max(ring_sold90 / 3.0, 0.001)
            feats[f"ring{lab}_mos"] = float(ring_active / monthly_sales)

            prev = r

        rows.append(feats)

    ring_df = pd.DataFrame(rows, index=subjects.index)
    return pd.concat([subjects.reset_index(drop=True), ring_df.reset_index(drop=True)], axis=1)
