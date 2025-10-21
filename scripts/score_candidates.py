# minimal placeholder: load models later; for now, echo input columns
import pandas as pd, sys
inp = "data/curated/properties.parquet"
df = pd.read_parquet(inp)
print("Scorable columns:", list(df.columns))
