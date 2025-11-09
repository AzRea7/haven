# src/haven/adapters/rent_quantile_bundle.py

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from haven.adapters.config import config

# Optional path; this must be set in config to use trained quantile models
_path_str = getattr(config, "RENT_QUANTILE_PATH", None)
_BUNDLE_PATH: Optional[Path] = Path(_path_str) if _path_str else None


@lru_cache
def _load_bundle() -> Optional[Dict[str, Any]]:
    """
    Load rent quantile model bundle if configured and present.

    Expected structure:
      {
        "feature_cols": [...],
        "q10": lightgbm.Booster,
        "q50": lightgbm.Booster,
        "q90": lightgbm.Booster,
      }
    """
    if _BUNDLE_PATH is None:
        return None
    if not _BUNDLE_PATH.exists():
        return None
    with _BUNDLE_PATH.open("rb") as f:
        return pickle.load(f)


def predict_rent_quantiles(features: Dict[str, float]) -> Dict[str, float]:
    """
    Predict rent quantiles.

    If bundle exists:
      - Use trained models.
    If not:
      - Use +/-10% band around features["base"] if provided.
      - If no base, return zeros.
    """
    bundle = _load_bundle()

    if bundle is None:
        base = float(features.get("base", 0.0))
        if base <= 0:
            return {"q10": 0.0, "q50": 0.0, "q90": 0.0}
        spread = base * 0.10
        return {
            "q10": max(base - spread, 0.0),
            "q50": base,
            "q90": base + spread,
        }

    feature_cols = bundle["feature_cols"]
    q10_model = bundle["q10"]
    q50_model = bundle["q50"]
    q90_model = bundle["q90"]

    X = [[features.get(col, 0.0) for col in feature_cols]]

    return {
        "q10": float(q10_model.predict(X)[0]),
        "q50": float(q50_model.predict(X)[0]),
        "q90": float(q90_model.predict(X)[0]),
    }
