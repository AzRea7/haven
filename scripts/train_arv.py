import os, joblib, pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

INP = "data/curated/properties.parquet"
TARGET_COL = "sale_price_after_rehab"
TIME_COL = "listed_date_quarter_index"  

df = pd.read_parquet(INP)
if TARGET_COL not in df.columns:
    raise SystemExit("Add a historical label column 'sale_price_after_rehab' to train ARV.")

y = df[TARGET_COL].astype(float)
feature_cols = [c for c in df.columns if c not in [TARGET_COL, "id", "address"]]
X = df[feature_cols]

if TIME_COL in df.columns:
    groups = df[TIME_COL].values
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X, groups=groups))
else:
    splits = [(range(0, int(0.8*len(X))), range(int(0.8*len(X)), len(X)))]

def train_quantile(alpha):
    model = LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=1500,
                          learning_rate=0.03, num_leaves=64, subsample=0.8, colsample_bytree=0.8)
    mape_scores = []
    for tr, te in splits:
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        mape_scores.append(mean_absolute_percentage_error(y.iloc[te], pred))
    print(f"Quantile {alpha:.2f} CV MAPE: {sum(mape_scores)/len(mape_scores):.4f}")
    model.fit(X, y)
    return model

os.makedirs("models", exist_ok=True)
m_p10 = train_quantile(0.10); joblib.dump(m_p10, "models/arv_q10.joblib")
m_p50 = train_quantile(0.50); joblib.dump(m_p50, "models/arv_q50.joblib")
m_p90 = train_quantile(0.90); joblib.dump(m_p90, "models/arv_q90.joblib")
print("saved models/arv_q{10,50,90}.joblib")
