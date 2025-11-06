import lightgbm as lgb
import numpy as np
import pandas as pd

TARGET = "sold_price"
DATE   = "sold_date"

QUANTILES = [0.10, 0.50, 0.90]
GBM_PARAMS = dict(objective="quantile", boosting_type="gbdt",
                  learning_rate=0.05, num_leaves=64, min_data_in_leaf=50,
                  subsample=0.9, colsample_bytree=0.9, n_estimators=2000)

FEATURES = [
  "beds","baths","sqft","year_built","zip",
  "psf",                       # from normalization
  # momentum features (added in step 2)
  "zhvi_chg_3m","zhvi_chg_6m","zhvi_chg_12m",
  "zori_chg_3m","zori_chg_6m","zori_chg_12m",
  # ring comps (added in step 3)
  "ring050_psf_med","ring100_psf_med","ring150_psf_med",
  "ring050_dom_med","ring100_dom_med","ring150_dom_med",
  "ring050_sale_to_list_med","ring100_sale_to_list_med","ring150_sale_to_list_med",
  "ring050_price_cuts_p","ring100_price_cuts_p","ring150_price_cuts_p",
  "ring050_mos","ring100_mos","ring150_mos"
]

def _add_time_keys(df: pd.DataFrame):
    df["year"] = df[DATE].dt.year
    df["month"] = df[DATE].dt.month
    df["quarter"] = df[DATE].dt.to_period("Q").astype(str)
    return df

def time_splits(df: pd.DataFrame, freq="Q"):
    """Yield (train_idx, valid_idx) by chronological folds."""
    df = df.sort_values(DATE)
    # roll by quarter: use earlier quarters for train, next quarter as valid
    keys = df[DATE].dt.to_period(freq).astype(str).unique()
    for i in range(3, len(keys)):  # start after 3 periods for stability
        train_periods = keys[:i]
        valid_period  = keys[i]
        tr_idx = df[df[DATE].dt.to_period(freq).astype(str).isin(train_periods)].index
        va_idx = df[df[DATE].dt.to_period(freq).astype(str) == valid_period].index
        if len(va_idx) > 200:  # ensure meaningful validation
            yield tr_idx, va_idx

def mape(y_true, y_pred):
    y_true = np.maximum(y_true, 1.0)
    return np.mean(np.abs(y_true - y_pred) / y_true)

def train_quantile_models(df: pd.DataFrame, mlflow_run=None):
    df = df.copy()
    df = _add_time_keys(df)
    df = df.dropna(subset=[TARGET, DATE])
    df = df.sort_values(DATE)

    X = df[FEATURES]
    y = df[TARGET].values

    models = {}
    cv_scores = {q: [] for q in QUANTILES}

    for q in QUANTILES:
        params = GBM_PARAMS.copy()
        params["alpha"] = q
        # early stopping by valid
        all_preds, all_true = [], []

        for tr_idx, va_idx in time_splits(df, freq="Q"):
            dtr = lgb.Dataset(X.loc[tr_idx], label=y[tr_idx])
            dva = lgb.Dataset(X.loc[va_idx], label=y[va_idx])
            model = lgb.train(params, dtr, valid_sets=[dva], valid_names=["val"],
                              verbose_eval=False, num_boost_round=8000, early_stopping_rounds=200)
            preds = model.predict(X.loc[va_idx], num_iteration=model.best_iteration)
            all_preds.append(preds)
            all_true.append(y[va_idx])

        m = mape(np.concatenate(all_true), np.concatenate(all_preds))
        cv_scores[q] = m

        # retrain on all data with best iteration heuristic
        dall = lgb.Dataset(X, label=y)
        model_full = lgb.train(params, dall,
                               num_boost_round=int(np.mean([2000]*1)))  # simple cap; or tune
        models[q] = model_full

    # log to MLflow if provided
    if mlflow_run:
        import mlflow
        for q, score in cv_scores.items():
            mlflow.log_metric(f"cv_mape_q{int(q*100)}", float(score))

    return models, cv_scores
