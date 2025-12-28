# src/haven/adapters/rent_estimator_lightgbm.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RentModelBundle:
    alphas: List[float]
    feature_names: List[str]
    models: Dict[float, Any]


class LightGBMRentEstimator:
    """
    Rent estimator that uses a LightGBM quantile bundle.

    Looks for:
      - models/rent_quantiles_with_neighborhood.joblib (preferred)
      - models/rent_quantiles.joblib (fallback)
    """

    def __init__(self, model_path: str | None = None) -> None:
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            preferred = Path("models/rent_quantiles_with_neighborhood.joblib")
            fallback = Path("models/rent_quantiles.joblib")
            self.model_path = preferred if preferred.exists() else fallback

        if not self.model_path.exists():
            logger.warning("rent_model_not_found", extra={"path": str(self.model_path)})
            self.bundle: RentModelBundle | None = None
            self.is_ready = False
            return

        bundle_raw = joblib.load(self.model_path)
        self.bundle = RentModelBundle(
            alphas=bundle_raw.get("alphas", [0.5]),
            feature_names=bundle_raw["feature_names"],
            models=bundle_raw["models"],
        )
        self.is_ready = True
        logger.info("rent_model_loaded", extra={"path": str(self.model_path), "alphas": self.bundle.alphas})

    def _ensure_ready(self) -> None:
        if not getattr(self, "is_ready", False) or self.bundle is None:
            raise RuntimeError("Rent model not loaded. Train rent_quantiles_with_neighborhood first.")

    def _build_feature_row(
        self,
        *,
        bedrooms: float,
        bathrooms: float,
        sqft: float,
        zipcode: str,
        property_type: str,
    ) -> Dict[str, float]:
        """
        Basic numeric feature vector. Neighborhood variables are baked in at
        training time via zipcode merge; at inference we just need core fields.
        """
        self._ensure_ready()
        zipcode = str(zipcode).strip().zfill(5)
        property_type = str(property_type).strip() or "single_family"

        feat: Dict[str, float] = {}
        for name in self.bundle.feature_names:
            if name == "bedrooms":
                feat[name] = float(bedrooms)
            elif name == "bathrooms":
                feat[name] = float(bathrooms)
            elif name == "sqft":
                feat[name] = float(sqft)
            elif name == "zipcode":
                feat[name] = float(int(zipcode)) if zipcode.isdigit() else 0.0
            elif name == "property_type":
                feat[name] = 1.0 if property_type == "single_family" else 0.0
            else:
                feat[name] = 0.0

        return feat

    def predict_unit_rent(
        self,
        *,
        bedrooms: float,
        bathrooms: float,
        sqft: float,
        zipcode: str,
        property_type: str,
        # Backward-compatible extras (ignored for LightGBM)
        address: str | None = None,
        city: str | None = None,
        state: str | None = None,
    ) -> float:
        """
        Predict median rent (alpha=0.5). Falls back to mean of all alphas if needed.
        Accepts optional address fields for compatibility with RentCast, but does not use them.
        """
        if not getattr(self, "is_ready", False) or self.bundle is None:
            # last-resort: crude heuristic
            logger.warning("rent_predict_fallback", extra={"reason": "model_not_ready"})
            sqft_f = float(sqft or 0.0)
            beds_f = float(bedrooms or 0.0)
            # basic: $1.10/sqft + $150/bedroom floor
            return max(1.10 * sqft_f + 150.0 * beds_f, 0.0)

        feat_row = self._build_feature_row(
            bedrooms=float(bedrooms or 0.0),
            bathrooms=float(bathrooms or 0.0),
            sqft=float(sqft or 0.0),
            zipcode=str(zipcode),
            property_type=str(property_type or "single_family"),
        )

        X = np.array([[feat_row[name] for name in self.bundle.feature_names]])

        models = self.bundle.models
        try:
            if 0.5 in models:
                pred = float(models[0.5].predict(X)[0])
            else:
                preds = [float(m.predict(X)[0]) for m in models.values()]
                pred = float(sum(preds) / max(len(preds), 1))
        except Exception as e:
            logger.warning("rent_predict_exception", extra={"error": str(e)})
            sqft_f = float(sqft or 0.0)
            pred = 1.10 * sqft_f

        return max(pred, 0.0)
