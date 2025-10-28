from __future__ import annotations
import pandas as pd

def compute_zip_momentum(zhvi_df: pd.DataFrame, zori_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect both inputs in LONG format:
        ['zip', 'date', 'value']
    Returns one row per ZIP (most recent date) with:
        zhvi_chg_3m, zhvi_chg_6m, zhvi_chg_12m,
        zori_chg_3m, zori_chg_6m, zori_chg_12m
    """
    def add_chgs(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["zip","date","value"]).sort_values(["zip","date"])
        d["v_3m"]  = d.groupby("zip")["value"].shift(3)
        d["v_6m"]  = d.groupby("zip")["value"].shift(6)
        d["v_12m"] = d.groupby("zip")["value"].shift(12)
        d[f"{prefix}_chg_3m"]  = (d["value"] - d["v_3m"])  / d["v_3m"]
        d[f"{prefix}_chg_6m"]  = (d["value"] - d["v_6m"])  / d["v_6m"]
        d[f"{prefix}_chg_12m"] = (d["value"] - d["v_12m"]) / d["v_12m"]
        return d[["zip","date",f"{prefix}_chg_3m",f"{prefix}_chg_6m",f"{prefix}_chg_12m"]]

    zhvi = add_chgs(zhvi_df, "zhvi")
    zori = add_chgs(zori_df, "zori")
    m = zhvi.merge(zori, on=["zip","date"], how="outer")

    # Keep most recent per ZIP
    idx = m.sort_values("date").groupby("zip").tail(1)
    return idx.reset_index(drop=True)
