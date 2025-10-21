# minimal placeholder: read raw data, compute a single toy feature, write parquet
import os, pandas as pd
os.makedirs("data/curated", exist_ok=True)
df = pd.read_csv("data/raw/example_properties.csv")  # put a CSV later
df["price_per_sqft"] = df["list_price"] / df["sqft"].clip(lower=1)
df.to_parquet("data/curated/properties.parquet", index=False)
print("Wrote data/curated/properties.parquet")
