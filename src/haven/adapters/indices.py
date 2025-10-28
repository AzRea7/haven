import pandas as pd

def compute_zip_momentum(zhvi_df: pd.DataFrame, zori_df: pd.DataFrame):
    # both long: ['zip','date','value']
    def add_chgs(df, prefix):
        df = df.sort_values(["zip","date"])
        df["value_3m"]  = df.groupby("zip")["value"].shift(3)
        df["value_6m"]  = df.groupby("zip")["value"].shift(6)
        df["value_12m"] = df.groupby("zip")["value"].shift(12)
        df[f"{prefix}_chg_3m"]  = (df["value"] - df["value_3m"]) / df["value_3m"]
        df[f"{prefix}_chg_6m"]  = (df["value"] - df["value_6m"]) / df["value_6m"]
        df[f"{prefix}_chg_12m"] = (df["value"] - df["value_12m"]) / df["value_12m"]
        return df[["zip","date",f"{prefix}_chg_3m",f"{prefix}_chg_6m",f"{prefix}_chg_12m"]]

    zhvi = add_chgs(zhvi_df, "zhvi")
    zori = add_chgs(zori_df, "zori")
    m = zhvi.merge(zori, on=["zip","date"], how="outer")
    # keep most recent per ZIP for listing join
    idx = m.sort_values("date").groupby("zip").tail(1)
    return idx  # ['zip','date','zhvi_chg_*','zori_chg_*']
