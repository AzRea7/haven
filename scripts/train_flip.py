import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

INP = "data/curated/properties.parquet"
LABEL = "flip_success"

# === Load data ===
df = pd.read_parquet(INP)
if LABEL not in df.columns:
    raise SystemExit("Add a binary 'flip_success' label to train flip classifier.")

y = df[LABEL].astype(int)

# Sanity: ensure both classes exist
if y.nunique() < 2:
    raise SystemExit("flip_success has only one class; adjust label construction.")

# Drop obvious non-features; then keep only numeric columns
drop_cols = [
    "id",
    "address",
    LABEL,
    "sale_price_after_rehab",
    "property_type",
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
X = X.select_dtypes(include=[np.number])

if X.empty:
    raise SystemExit("No numeric feature columns available for training.")

tscv = TimeSeriesSplit(n_splits=5)


def fit_and_eval_fold(tr_idx, te_idx):
    base = Pipeline(
        [
            ("sc", StandardScaler(with_mean=False)),
            (
                "lr",
                LogisticRegression(
                    max_iter=3000,
                    C=1.0,
                    penalty="l2",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    clf = CalibratedClassifierCV(
        base,
        method="isotonic",
        cv=3,
    )

    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]

    ap = average_precision_score(y_te, p)
    brier = brier_score_loss(y_te, p)
    return ap, brier


# === Parallel cross-validation ===
cv_start = time.perf_counter()

results = Parallel(n_jobs=-1)(
    delayed(fit_and_eval_fold)(tr_idx, te_idx)
    for tr_idx, te_idx in tscv.split(X)
)

cv_time = time.perf_counter() - cv_start

aps = [r[0] for r in results]
briers = [r[1] for r in results]

print(f"PR-AUC: {np.mean(aps):.3f} | Brier: {np.mean(briers):.3f}")
print(f"CV training time (parallel): {cv_time:.2f}s")

# === Final fit on all data ===
base = Pipeline(
    [
        ("sc", StandardScaler(with_mean=False)),
        (
            "lr",
            LogisticRegression(
                max_iter=3000,
                C=1.0,
                penalty="l2",
                n_jobs=-1,
            ),
        ),
    ]
)

final_clf = CalibratedClassifierCV(
    base,
    method="isotonic",
    cv=3,
)

train_start = time.perf_counter()
final_clf.fit(X, y)
train_time = time.perf_counter() - train_start

os.makedirs("models", exist_ok=True)
joblib.dump(final_clf, "models/flip_logit_calibrated.joblib")

print("saved models/flip_logit_calibrated.joblib")
print(f"Full-data fit time: {train_time:.2f}s")
