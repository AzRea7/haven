from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve


def mape(y: ArrayLike, yhat: ArrayLike) -> float:
    y_arr = np.asarray(y, dtype=float)
    yhat_arr = np.asarray(yhat, dtype=float)
    y_arr = np.maximum(y_arr, 1.0)
    return float(np.mean(np.abs(yhat_arr - y_arr) / y_arr))


def mae(y: ArrayLike, yhat: ArrayLike) -> float:
    y_arr = np.asarray(y, dtype=float)
    yhat_arr = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(yhat_arr - y_arr)))


def eval_arv_by_time_zip(
    df: pd.DataFrame,
    date_col: str = "sold_date",
    zip_col: str = "zip",
) -> pd.DataFrame:
    # ensure numeric arrays before math and guard NA
    work = df.copy().dropna(subset=["sold_price", "q50"])
    work["quarter"] = pd.to_datetime(work[date_col], errors="coerce").dt.to_period("Q").astype(str)

    out: list[dict[str, Any]] = []
    for (q, z), g in work.groupby(["quarter", zip_col], dropna=False):
        sold = np.asarray(g["sold_price"].to_numpy(), dtype=float)
        pred = np.asarray(g["q50"].to_numpy(), dtype=float)
        out.append(
            dict(
                quarter=q,
                zip=z,
                n=len(g),
                mape=mape(sold, pred),
                mae=mae(sold, pred),
            )
        )
    return pd.DataFrame(out).sort_values(["quarter", "zip"])


def eval_classifier(y_true: ArrayLike, p_hat: ArrayLike) -> dict[str, Any]:
    # coerce to 1D float arrays to keep mypy and numpy happy
    y_arr = np.asarray(y_true, dtype=float).ravel()
    p_arr = np.asarray(p_hat, dtype=float).ravel()

    prec, rec, thr = precision_recall_curve(y_arr, p_arr)
    ap = float(average_precision_score(y_arr, p_arr))
    frac_pos, mean_pred = calibration_curve(y_arr, p_arr, n_bins=10)

    return {
        "ap": ap,
        "pr_curve": (prec, rec, thr),        # tuple[np.ndarray, np.ndarray, np.ndarray]
        "calibration": (frac_pos, mean_pred) # tuple[np.ndarray, np.ndarray]
    }
