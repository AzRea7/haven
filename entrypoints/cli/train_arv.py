import argparse, joblib, mlflow, pandas as pd
from haven.services.arv_trainer import train_quantile_models

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()

    mlflow.set_experiment("ARV")
    with mlflow.start_run():
        df = pd.read_parquet(args.inp)
        models, scores = train_quantile_models(df, mlflow_run=True)
        for q, model in models.items():
            joblib.dump(model, f"{args.outdir}/arv_q{int(q*100)}.joblib")
