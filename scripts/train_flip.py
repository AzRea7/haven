import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INP = "data/curated/properties.parquet"
LABEL = "flip_success"

df = pd.read_parquet(INP)
if LABEL not in df.columns:
    raise SystemExit("Add a binary 'flip_success' label to train flip classifier.")

y = df[LABEL].astype(int)
X = df[[c for c in df.columns if c not in ["id","address",LABEL,"sale_price_after_rehab"]]]

tscv = TimeSeriesSplit(n_splits=5)
aps, briers = [], []
for tr, te in tscv.split(X):
    base = Pipeline([
        ("sc", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=3000, C=1.0, penalty="l2"))
    ])
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X.iloc[tr], y.iloc[tr])
    p = clf.predict_proba(X.iloc[te])[:,1]
    aps.append(average_precision_score(y.iloc[te], p))
    briers.append(brier_score_loss(y.iloc[te], p))

print(f"PR-AUC: {np.mean(aps):.3f} | Brier: {np.mean(briers):.3f}")
os.makedirs("models", exist_ok=True)
clf.fit(X, y)
joblib.dump(clf, "models/flip_logit_calibrated.joblib")
print("saved models/flip_logit_calibrated.joblib")
