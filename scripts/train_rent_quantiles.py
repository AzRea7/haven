import pickle
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from joblib import Parallel, delayed

DATA_PATH = Path("data/curated/rent_training.parquet")
OUT_PATH = Path("models/rent_quantiles.pkl")


def main() -> None:
    t_start = time.perf_counter()

    df = pd.read_parquet(DATA_PATH)

    # Columns we expect from build_features_parallel.py
    feature_cols = [
        "bedrooms",
        "bathrooms",
        "sqft",
        "zipcode_encoded",
        "property_type_encoded",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns in rent_training: {missing}")

    if "rent" not in df.columns:
        raise SystemExit("Expected 'rent' column in rent_training.parquet")

    # Ensure numeric dtypes for LightGBM
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df["rent"], errors="coerce").fillna(0.0)

    train_set = lgb.Dataset(X, label=y)

    params_base = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "max_depth": -1,
        "num_threads": -1,  # use all cores per model
    }

    def train_q(alpha: float):
        params = {
            **params_base,
            "objective": "quantile",
            "alpha": alpha,
        }
        return lgb.train(params, train_set, num_boost_round=400)

    quantiles = [0.10, 0.50, 0.90]

    t_train_start = time.perf_counter()
    models_list = Parallel(n_jobs=3)(
        delayed(train_q)(alpha) for alpha in quantiles
    )
    t_train = time.perf_counter() - t_train_start

    models = {
        "q10": models_list[0],
        "q50": models_list[1],
        "q90": models_list[2],
        "feature_cols": feature_cols,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as f:
        pickle.dump(models, f)

    total_time = time.perf_counter() - t_start

    print(f"Saved rent quantile bundle to {OUT_PATH}")
    print(f"Quantile models training time (parallel): {t_train:.2f}s")
    print(f"End-to-end script time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
