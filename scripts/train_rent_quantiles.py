import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd

DATA_PATH = Path("data/curated/rent_training.parquet")
OUT_PATH = Path("models/rent_quantiles.pkl")

def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    feature_cols = [
        "bedrooms",
        "bathrooms",
        "sqft",
        "zipcode_encoded",
        "property_type_encoded",
        # etc: any engineered features you already use
    ]
    y = df["rent"]

    # You should be using proper encoding / CV; keeping it compact here.
    train = lgb.Dataset(df[feature_cols], label=y)

    params_base = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "max_depth": -1,
    }

    def train_q(alpha: float):
        params = {
            **params_base,
            "objective": "quantile",
            "alpha": alpha,
        }
        return lgb.train(params, train, num_boost_round=400)

    models = {
        "q10": train_q(0.10),
        "q50": train_q(0.50),
        "q90": train_q(0.90),
        "feature_cols": feature_cols,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as f:
        pickle.dump(models, f)

    print(f"Saved rent quantile bundle to {OUT_PATH}")

if __name__ == "__main__":
    main()
