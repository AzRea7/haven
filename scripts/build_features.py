import os, numpy as np, pandas as pd

RAW = "data/raw/listings.csv"
OUT = "data/curated/properties.parquet"

os.makedirs("data/curated", exist_ok=True)
df = pd.read_csv(RAW)

# ---------- helpers ----------
def to_num(s, default=None):
    out = pd.to_numeric(s, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out

def ensure_col(name, default):
    """
    Return a numeric/string Series for column `name`.
    - If present: coerce and return.
    - If missing:
        - if default is callable -> call to get a vector (length len(df)).
        - else -> fill scalar default across rows.
    """
    if name in df.columns:
        try:
            return to_num(df[name])
        except Exception:
            return df[name]
    if callable(default):
        vec = default()
        if not isinstance(vec, pd.Series):
            vec = pd.Series(vec, index=df.index)
        return vec
    else:
        return pd.Series([default] * len(df), index=df.index)

# ---------- required base fields ----------
# list_price & sqft must exist or be derivable
if "list_price" not in df.columns:
    raise SystemExit("list_price is required in data/raw/listings.csv (after normalize).")

df["list_price"] = to_num(df["list_price"])
if "sqft" not in df.columns:
    df["sqft"] = np.nan
df["sqft"] = to_num(df["sqft"]).replace(0, np.nan).fillna(1200)

# ---------- finance inputs ----------
est_rent = ensure_col("est_rent", lambda: df["list_price"] * 0.0065)
tax_annual = ensure_col("tax_annual", lambda: df["list_price"] * 0.012)
insurance_annual = ensure_col("insurance_annual", 600.0)
hoa_monthly = ensure_col("hoa_monthly", 0.0)
vacancy_rate = ensure_col("vacancy_rate", 0.07)
rate_annual = ensure_col("rate_annual", 0.065)
hold_months = ensure_col("hold_months", 6.0)
buy_closing_rate = ensure_col("buy_closing_rate", 0.02)
sell_closing_rate = ensure_col("sell_closing_rate", 0.07)
rehab_est = ensure_col("rehab_est", 20000.0)

# bind them
df["est_rent"] = to_num(est_rent, 0.0)
df["tax_annual"] = to_num(tax_annual, 0.0)
df["insurance_annual"] = to_num(insurance_annual, 0.0)
df["hoa_monthly"] = to_num(hoa_monthly, 0.0)
df["vacancy_rate"] = to_num(vacancy_rate, 0.07)
df["rate_annual"] = to_num(rate_annual, 0.065)
df["hold_months"] = to_num(hold_months, 6.0)
df["buy_closing_rate"] = to_num(buy_closing_rate, 0.02)
df["sell_closing_rate"] = to_num(sell_closing_rate, 0.07)
df["rehab_est"] = to_num(rehab_est, 20000.0)

# ---------- engineered finance features ----------
df["price_per_sqft"] = df["list_price"] / df["sqft"].clip(lower=1)

gross_rent_annual = df["est_rent"] * 12
opex_annual = (
    df["tax_annual"]
    + df["insurance_annual"]
    + df["hoa_monthly"] * 12
    + gross_rent_annual * 0.08 
)
df["NOI"] = gross_rent_annual * (1 - df["vacancy_rate"]) - opex_annual
df["CapRate"] = df["NOI"] / df["list_price"]

monthly_rate = df["rate_annual"] / 12.0
carry_interest = df["list_price"] * monthly_rate * df["hold_months"]
non_mort_hold = (
    (df["tax_annual"] + df["insurance_annual"]) * (df["hold_months"] / 12.0)
    + df["hoa_monthly"] * df["hold_months"]
)
df["HoldCost"] = carry_interest + non_mort_hold

df["BuyFees"] = df["list_price"] * df["buy_closing_rate"]
df["SellFees_rate"] = df["sell_closing_rate"]

# ---------- write parquet ----------
if "zip" in df.columns:
    df["zip"] = df["zip"].astype(str)

df.to_parquet(OUT, index=False)
print(f"wrote {OUT} with shape {df.shape}")
