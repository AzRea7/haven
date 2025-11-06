import lightgbm as lgb
import numpy as np
import pandas as pd
from collections.abc import Iterator
from numpy.typing import ArrayLike
from typing import Any

TARGET = "sold_price"
DATE = "sold_date"

QUANTILES: list[float] = [0.10, 0.50, 0.90]
GBM_PARAMS: dict[str, Any] = dict(
    objective="quantile",
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=64,
    min_data_in_leaf=50,
    subsample=0.9,
    colsample_bytree=0.9,
    n_estimators=2000,
)

FEATURES: list[str] = [
    "beds", "baths", "sqft", "year_built", "zip",
    "psf",
    "zhvi_chg_3m", "zhvi_chg_6m", "zhvi_chg_12m",
    "zori_chg_3m", "zori_chg_6m", "zori_chg_12m",
    "ring050_psf_med", "ring100_psf_med", "ring150_psf_med",
    "ring050_dom_med", "ring100_dom_med", "ring150_dom_med",
    "ring050_sale_to_list_med", "ring100_sale_to_list_med", "ring150_sale_to_list_med",
    "ring050_price_cuts_p", "ring100_price_cuts_p", "ring150_price_cuts_p",
    "ring050_mos", "ring100_mos", "ring150_mos",
]


def _add_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df[DATE].dt.year
    df["month"] = df[DATE].dt.month
    df["quarter"] = df[DATE].dt.to_period("Q").astype(str)
    return df


def time_splits(df: pd.DataFrame, freq: str = "Q") -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_idx, valid_idx) by chronological folds."""
    df = df.sort_values(DATE)
    keys = df[DATE].dt.to_period(freq).astype(str).unique()
    for i in range(3, len(keys)):  # start after 3 periods for stability
        train_periods = keys[:i]
        valid_period = keys[i]
        tr_idx = df[df[DATE].dt.to_period(freq).astype(str).isin(train_periods)].index
        va_idx = df[df[DATE].dt.to_period(freq).astype(str) == valid_period].index
        if len(va_idx) > 200:  # ensure meaningful validation
            yield tr_idx, va_idx


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_true_arr = np.maximum(y_true_arr, 1.0)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr) / y_true_arr))


def train_quantile_models(
    df: pd.DataFrame, mlflow_run: Any | None = None
) -> tuple[dict[float, Any], dict[float, float]]:
    df = df.copy()
    df = _add_time_keys(df)
    df = df.dropna(subset=[TARGET, DATE])
    df = df.sort_values(DATE)

    X: pd.DataFrame = df[FEATURES]
    y: np.ndarray = df[TARGET].to_numpy(dtype=float)

    models: dict[float, Any] = {}
    cv_scores: dict[float, float] = {q: 0.0 for q in QUANTILES}

    for q in QUANTILES:
        params = GBM_PARAMS.copy()
        params["alpha"] = q

        all_preds: list[np.ndarray] = []
        all_true: list[np.ndarray] = []

        for tr_idx, va_idx in time_splits(df, freq="Q"):
            dtr = lgb.Dataset(X.loc[tr_idx], label=y[tr_idx])
            dva = lgb.Dataset(X.loc[va_idx], label=y[va_idx])

            # Some LightGBM stub versions don't declare these kwargs; keep them but silence mypy.
            model = lgb.train(  # type: ignore[call-arg]
                params,
                dtr,
                valid_sets=[dva],
                valid_names=["val"],
                verbose_eval=False,
                num_boost_round=8000,
                early_stopping_rounds=200,
            )
            preds = model.predict(X.loc[va_idx], num_iteration=getattr(model, "best_iteration", None))
            all_preds.append(np.asarray(preds, dtype=float))
            all_true.append(y[va_idx])

        y_true_all = np.concatenate(all_true) if all_true else np.array([], dtype=float)
        y_pred_all = np.concatenate(all_preds) if all_preds else np.array([], dtype=float)
        m = mape(y_true_all, y_pred_all) if y_true_all.size else float("nan")
        cv_scores[q] = float(m)

        # retrain on all data (simple cap; or tune)
        dall = lgb.Dataset(X, label=y)
        model_full = lgb.train(params, dall, num_boost_round=2000)
        models[q] = model_full

    if mlflow_run:
        import mlflow
        for q, score in cv_scores.items():
            mlflow.log_metric(f"cv_mape_q{int(q*100)}", float(score))

    return models, cv_scores
