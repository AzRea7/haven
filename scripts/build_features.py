# scripts/build_features.py
import os, pandas as pd, numpy as np

RAW = "data/raw/candidates.csv"         # put your listings here
OUT = "data/curated/properties.parquet"

os.makedirs("data/curated", exist_ok=True)
df = pd.read_csv(RAW)

# ---- sanity defaults (override in your CSV) ----
df["sqft"] = df.get("sqft", pd.Series([np.nan]*len(df))).fillna(1200)
df["list_price"] = df["list_price"].astype(float)
df["est_rent"] = df.get("est_rent", pd.Series([np.nan]*len(df))).fillna(df["list_price"] * 0.0065)  # ≈ rough rent/price
df["tax_annual"] = df.get("tax_annual", pd.Series([np.nan]*len(df))).fillna(df["list_price"] * 0.012)
df["insurance_annual"] = df.get("insurance_annual", pd.Series([600]*len(df)))
df["hoa_monthly"] = df.get("hoa_monthly", pd.Series([0]*len(df)))
df["vacancy_rate"] = df.get("vacancy_rate", pd.Series([0.07]*len(df)))
df["rate_annual"] = df.get("rate_annual", pd.Series([0.065]*len(df)))
df["hold_months"] = df.get("hold_months", pd.Series([6]*len(df)))
df["buy_closing_rate"] = df.get("buy_closing_rate", pd.Series([0.02]*len(df)))
df["sell_closing_rate"] = df.get("sell_closing_rate", pd.Series([0.07]*len(df)))
df["rehab_est"] = df.get("rehab_est", pd.Series([20000]*len(df)))  # placeholder; will be replaced by model later

# ---- finance features ----
df["price_per_sqft"] = df["list_price"] / df["sqft"].clip(lower=1)

# NOI for rentals (useful even for flips to gauge demand)
gross_rent_annual = df["est_rent"] * 12
opex_annual = (df["tax_annual"] + df["insurance_annual"] +
               df["hoa_monthly"] * 12 + gross_rent_annual * 0.08)  # 8% mgmt placeholder
noi = gross_rent_annual * (1 - df["vacancy_rate"]) - opex_annual
df["NOI"] = noi
df["CapRate"] = df["NOI"] / df["list_price"]

# debt service approx (interest-only carry for hold period; OK for flip)
monthly_rate = df["rate_annual"] / 12
carry_interest = df["list_price"] * monthly_rate * df["hold_months"]
non_mort_hold = (df["tax_annual"] + df["insurance_annual"]) * (df["hold_months"] / 12) + df["hoa_monthly"] * df["hold_months"]
df["HoldCost"] = carry_interest + non_mort_hold

# transaction costs
df["BuyFees"] = df["list_price"] * df["buy_closing_rate"]
df["SellFees_rate"] = df["sell_closing_rate"]

# labels/targets will be added after we collect outcomes; for now features only
df.to_parquet(OUT, index=False)
print(f"✅ wrote {OUT} with {df.shape[0]} rows and {df.shape[1]} cols")
