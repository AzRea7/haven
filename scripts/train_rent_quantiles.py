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

    feature_cols = [
        "bedrooms",
        "bathrooms",
        "sqft",
        "zipcode_encoded",
        "property_type_encoded",
        # include any other engineered features you use
    ]

    if "rent" not in df.columns:
        raise SystemExit("Expected 'rent' column in rent_training.parquet")

    X = df[feature_cols]
    y = df["rent"]

    train_set = lgb.Dataset(X, label=y)

    # Base params shared by all quantile models
    params_base = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "max_depth": -1,
        # Enable parallel tree building across CPU cores
        "num_threads": -1,
    }

    def train_q(alpha: float):
        params = {
            **params_base,
            "objective": "quantile",
            "alpha": alpha,
        }
        # Each call trains one quantile model
        return lgb.train(params, train_set, num_boost_round=400)

    # Train three quantile models in parallel.
    # n_jobs=3 here since we have 3 independent models; each still uses num_threads=-1 internally.
    # For stricter measurements, you can set num_threads to a fixed value instead of -1.
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
