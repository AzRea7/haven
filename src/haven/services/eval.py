import numpy as np, pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

def mape(y, yhat): y = np.maximum(y, 1.0); return np.mean(np.abs(yhat - y) / y)
def mae(y, yhat):  return float(np.mean(np.abs(yhat - y)))

def eval_arv_by_time_zip(df: pd.DataFrame, date_col="sold_date", zip_col="zip"):
    df = df.copy().dropna(subset=["sold_price","q50"])
    df["quarter"] = pd.to_datetime(df[date_col]).dt.to_period("Q").astype(str)
    out = []
    for (q, z), g in df.groupby(["quarter", zip_col]):
        out.append(dict(quarter=q, zip=z,
                        n=len(g), mape=mape(g["sold_price"], g["q50"]),
                        mae=mae(g["sold_price"], g["q50"])))
    return pd.DataFrame(out).sort_values(["quarter","zip"])

def eval_classifier(y_true, p_hat):
    prec, rec, thr = precision_recall_curve(y_true, p_hat)
    ap = average_precision_score(y_true, p_hat)
    frac_pos, mean_pred = calibration_curve(y_true, p_hat, n_bins=10)
    return {
        "ap": ap,
        "pr_curve": (prec, rec, thr),
        "calibration": (frac_pos, mean_pred)
    }
