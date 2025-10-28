import numpy as np, pandas as pd

EARTH_R = 3958.8  # miles

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def ring_buckets(dists):
    # returns ring key per neighbor: 0.5, 1.0, 1.5 (None if > 1.5)
    r = np.full_like(dists, 999.0, dtype=float)
    r[dists <= 0.5] = 0.5
    r[(dists > 0.5) & (dists <= 1.0)] = 1.0
    r[(dists > 1.0) & (dists <= 1.5)] = 1.5
    r[r==999.0] = np.nan
    return r

def compute_ring_features(subjects: pd.DataFrame, comps: pd.DataFrame):
    # subjects: listings to score (lat,lon, zip, beds,baths,sqft, list_price, etc.)
    # comps:   mix of forSale + sold with needed columns: ['lat','lon','sqft','list_price','sold_price','dom','price_cut','close_date','list_date']
    out_rows = []
    subj_arr = subjects[["lat","lon"]].to_numpy()
    comp_arr = comps[["lat","lon"]].to_numpy()

    # Precompute scalar arrays for comp stats
    comp_psf = (comps["sold_price"].fillna(comps["list_price"])) / comps["sqft"].clip(lower=300)
    sale_to_list = comps["sold_price"] / comps["list_price"]
    price_cuts_p = comps["price_cut"].fillna(0.0)  # assume 0/1 or a %

    # months of supply approximate:
    # MOS ~= Active Listings / Monthly Sales (use 3m average)
    # You can precompute per ZIP; here, rough local proxy from comps windows.
    # For simplicity we’ll compute MOS per ring as: actives / (sold in last 90d / 3)
    recent_days = 90
    now = pd.Timestamp.utcnow().normalize()
    sold_recent = (now - pd.to_datetime(comps["close_date"], errors="coerce")).dt.days <= recent_days
    active_mask = comps["sold_price"].isna()

    for i, (slat, slon) in enumerate(subj_arr):
        dists = haversine(slat, slon, comp_arr[:,0], comp_arr[:,1])
        rings = ring_buckets(dists)

        feats = {}
        for key, label in [(0.5,"050"),(1.0,"100"),(1.5,"150")]:
            idx = np.where(rings == key)[0]
            if idx.size == 0:
                # fill NaNs if no comps in ring
                feats.update({
                    f"ring{label}_psf_med": np.nan,
                    f"ring{label}_dom_med": np.nan,
                    f"ring{label}_sale_to_list_med": np.nan,
                    f"ring{label}_price_cuts_p": np.nan,
                    f"ring{label}_mos": np.nan
                })
                continue

            feats[f"ring{label}_psf_med"] = np.nanmedian(comp_psf.iloc[idx])
            feats[f"ring{label}_dom_med"] = np.nanmedian(comps["dom"].iloc[idx])
            feats[f"ring{label}_sale_to_list_med"] = np.nanmedian(sale_to_list.iloc[idx])
            feats[f"ring{label}_price_cuts_p"] = float(np.nanmean(price_cuts_p.iloc[idx]))

            # MOS proxy
            ring_active = np.sum(active_mask.iloc[idx])
            ring_sold90 = np.sum(sold_recent.iloc[idx])
            monthly_sales = max(ring_sold90 / 3.0, 0.001)
            feats[f"ring{label}_mos"] = ring_active / monthly_sales

        out_rows.append(feats)

    ring_df = pd.DataFrame(out_rows, index=subjects.index)
    return pd.concat([subjects.reset_index(drop=True), ring_df.reset_index(drop=True)], axis=1)
