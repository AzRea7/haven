from collections.abc import Mapping
from typing import Any

import joblib
import pandas as pd


def load_arv_bundle(model_dir: str = "models") -> dict[str, Any]:
    q10 = joblib.load(f"{model_dir}/arv_q10.joblib")
    q50 = joblib.load(f"{model_dir}/arv_q50.joblib")
    q90 = joblib.load(f"{model_dir}/arv_q90.joblib")
    return {"q10": q10, "q50": q50, "q90": q90}

def score_arv(models: Mapping[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    preds: dict[str, Any] = {}
    for k, m in models.items():
        preds[k] = m.predict(X)
    return pd.DataFrame(preds, index=X.index)

def compute_profit_and_mao(cands: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """
    Requires cands with rehab, hold_costs, closing_costs, selling_cost_rate, buy_cost_rate, etc.
    MAO = ARV * (1 - selling_cost_rate) - rehab - hold_costs - desired_profit - buy_costs
    """
    df = cands.join(preds)
    for k in ["q10","q50","q90"]:
        arv = df[k]
        selling_costs = arv * df["selling_cost_rate"]
        buy_costs     = arv * df["buy_cost_rate"]
        net = arv - selling_costs - df["rehab"] - df["hold_costs"] - buy_costs
        df[f"profit_{k}"] = net - df.get("offer_price", 0.0)
        df[f"mao_{k}"]    = arv*(1 - df["selling_cost_rate"]) - df["rehab"] - df["hold_costs"] - df["desired_profit"] - arv*df["buy_cost_rate"]
    return df
