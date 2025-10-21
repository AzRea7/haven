import os, pandas as pd, numpy as np

RAW = "data/raw/listings.csv"          # <- now from normalize_hasdata.py
OUT = "data/curated/properties.parquet"

os.makedirs("data/curated", exist_ok=True)
df = pd.read_csv(RAW)

# ---- defaults / placeholders; replace as you get better data ----
def col(name, default):
    return df.get(name, pd.Series([default]*len(df))).fillna(default)

df["sqft"] = pd.to_numeric(df.get("sqft"), errors="coerce").fillna(1200)
df["list_price"] = pd.to_numeric(df.get("list_price"), errors="coerce")

df["est_rent"] = col("est_rent", df["list_price"] * 0.0065)   # rough rent/price
df["tax_annual"] = col("tax_annual", df["list_price"] * 0.012)
df["insurance_annual"] = col("insurance_annual", 600)
df["hoa_monthly"] = col("hoa_monthly", 0)
df["vacancy_rate"] = col("vacancy_rate", 0.07)
df["rate_annual"] = col("rate_annual", 0.065)
df["hold_months"] = col("hold_months", 6)
df["buy_closing_rate"] = col("buy_closing_rate", 0.02)
df["sell_closing_rate"] = col("sell_closing_rate", 0.07)
df["rehab_est"] = col("rehab_est", 20000)

# ---- finance features ----
df["price_per_sqft"] = df["list_price"] / df["sqft"].clip(lower=1)

gross_rent_annual = df["est_rent"] * 12
opex_annual = (df["tax_annual"] + df["insurance_annual"] +
               df["hoa_monthly"] * 12 + gross_rent_annual * 0.08)
noi = gross_rent_annual * (1 - df["vacancy_rate"]) - opex_annual
df["NOI"] = noi
df["CapRate"] = df["NOI"] / df["list_price"]

monthly_rate = df["rate_annual"] / 12
carry_interest = df["list_price"] * monthly_rate * df["hold_months"]
non_mort_hold = (df["tax_annual"] + df["insurance_annual"]) * (df["hold_months"]/12) + df["hoa_monthly"] * df["hold_months"]
df["HoldCost"] = carry_interest + non_mort_hold

df["BuyFees"] = df["list_price"] * df["buy_closing_rate"]
df["SellFees_rate"] = df["sell_closing_rate"]

df.to_parquet(OUT, index=False)
print(f"✅ wrote {OUT} with {df.shape[0]} rows and {df.shape[1]} cols")
