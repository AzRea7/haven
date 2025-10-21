# scripts/score_candidates.py
import joblib, pandas as pd, numpy as np

FIN = {
  "target_margin": 0.12,
  "buy_closing_rate": 0.02,
  "sell_closing_rate": 0.07
}

df = pd.read_parquet("data/curated/properties.parquet")
X = df[[c for c in df.columns if c not in ["id","address","sale_price_after_rehab","flip_success"]]]

arv_p10 = joblib.load("models/arv_q10.joblib").predict(X)
arv_p50 = joblib.load("models/arv_q50.joblib").predict(X)
arv_p90 = joblib.load("models/arv_q90.joblib").predict(X)
p_succ = joblib.load("models/flip_logit_calibrated.joblib").predict_proba(X)[:,1]

buy = df["list_price"].values
rehab = df.get("rehab_est", pd.Series([20000]*len(df))).values
hold = df["HoldCost"].values
buy_fee = buy * df.get("buy_closing_rate", FIN["buy_closing_rate"]).values
sell_rate = df.get("sell_closing_rate", FIN["sell_closing_rate"]).values

def profit(arv):
    sell_fee = arv * sell_rate
    total = buy + rehab + hold + buy_fee + sell_fee
    return arv - total, total

p10, cost10 = profit(arv_p10)
p50, cost50 = profit(arv_p50)
p90, cost90 = profit(arv_p90)

# MAO from p50 ARV:
mao = arv_p50 * (1 - FIN["target_margin"] - sell_rate) - rehab - hold - buy_fee

out = pd.DataFrame({
    "id": df.get("id", pd.Series(range(len(df)))),
    "flip_prob": p_succ,
    "ARV_P10": arv_p10, "ARV_P50": arv_p50, "ARV_P90": arv_p90,
    "Profit_P10": p10,   "Profit_P50": p50,   "Profit_P90": p90,
    "MAO": mao
}).sort_values("flip_prob", ascending=False)

out.to_csv("out/scored.csv", index=False)
print("✅ wrote out/scored.csv")
