from pathlib import Path

import pandas as pd


def read_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def write_df(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
