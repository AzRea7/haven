import argparse, pandas as pd
from services.eval import eval_arv_by_time_zip, eval_classifier

ap = argparse.ArgumentParser()
ap.add_argument("--sold_scored", required=True)  # file with sold_price and predicted q50
ap.add_argument("--flip_labels", default=None)   # optional for classifier
args = ap.parse_args()

df = pd.read_parquet(args.sold_scored)
print("ARV by quarter/ZIP:")
print(eval_arv_by_time_zip(df).head(20))

if args.flip_labels:
    lab = pd.read_parquet(args.flip_labels)  # columns: id, flip_success
    df2 = df.merge(lab, on="property_id")
    y = df2["flip_success"].astype(int).values
    p = df2["p_success"].values
    out = eval_classifier(y, p)
    print("AP:", out["ap"])
    # You can also save PR & calibration points to CSV and plot elsewhere
