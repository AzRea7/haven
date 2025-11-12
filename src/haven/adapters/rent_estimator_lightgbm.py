from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - for environments without LightGBM
    lgb = None  # type: ignore[assignment]


# Fixed feature order used by both training & prediction
RENT_FEATURE_COLUMNS: List[str] = [
    "bedrooms",
    "bathrooms",
    "sqft",
    "zipcode_encoded",
    "property_type_encoded",
]


@dataclass
class _Artifacts:
    model: Any
    scaler: Any


class LightGBMRentEstimator:
    """
    Rent estimator.

    - If model/scaler artifacts are available, uses them.
    - Otherwise falls back to a reasonably sane heuristic based on sqft, beds, baths.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ) -> None:
        # Default locations (override these or wire via config if needed)
        self._model_path = model_path or "artifacts/rent_model.lgb"
        self._scaler_path = scaler_path or "artifacts/rent_scaler.pkl"

        self._artifacts: Optional[_Artifacts] = None
        self._load_artifacts_if_available()

    # -------------------------------------------------------------------------
    # Artifact loading
    # -------------------------------------------------------------------------

    def _load_artifacts_if_available(self) -> None:
        if not (os.path.exists(self._model_path) and os.path.exists(self._scaler_path)):
            # No artifacts – we will use heuristics
            self._artifacts = None
            return

        if lgb is None:
            # LightGBM not installed – also fall back to heuristics
            self._artifacts = None
            return

        model = lgb.Booster(model_file=self._model_path)
        scaler = joblib.load(self._scaler_path)

        self._artifacts = _Artifacts(model=model, scaler=scaler)

    # -------------------------------------------------------------------------
    # Encoding helpers
    # -------------------------------------------------------------------------

    def _encode_zip(self, zipcode: str) -> float:
        """
        Simple numeric encoding of ZIP.
        You can replace this with a proper mapping if you already have one.
        """
        zipcode = (zipcode or "").strip()
        if not zipcode.isdigit():
            return 0.0
        return float(int(zipcode))

    def _encode_property_type(self, property_type: str) -> float:
        """
        Simple categorical encoding.
        Replace with your own mapping if you already have one in training.
        """
        pt = (property_type or "").lower().strip()
        mapping = {
            "single_family": 1.0,
            "condo": 2.0,
            "condo_townhome": 2.0,
            "townhome": 2.0,
            "apartment": 3.0,
            "apartment_complex": 3.0,
            "multi_family": 4.0,
        }
        return mapping.get(pt, 0.0)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def predict_unit_rent(
        self,
        *,
        bedrooms: float,
        bathrooms: float,
        sqft: float,
        zipcode: str,
        property_type: str,
    ) -> float:
        """
        Main entrypoint.

        If LightGBM artifacts are available:
          - Build a DataFrame with fixed columns.
          - Apply scaler.
          - Get model prediction.

        Otherwise:
          - Use a rule-of-thumb heuristic that scales with sqft, beds, baths, and zip.
        """
        beds = float(bedrooms or 0.0)
        baths = float(bathrooms or 0.0)
        size = float(sqft or 0.0)
        zip_str = str(zipcode or "")
        pt_str = str(property_type or "single_family")

        zip_enc = self._encode_zip(zip_str)
        pt_enc = self._encode_property_type(pt_str)

        if self._artifacts is None:
            # Heuristic fallback – deliberately simple but not totally dumb.
            base_per_sqft = 1.2  # ~ $1.20/sqft
            beds_bonus = 150.0 * beds
            baths_bonus = 100.0 * baths

            # Very crude metro adjustment based on first 3 digits of ZIP
            metro_factor = 1.0
            if zip_str.startswith("48"):
                metro_factor = 1.1  # SE Michigan-ish boost

            rent = (size * base_per_sqft + beds_bonus + baths_bonus) * metro_factor
            return float(max(rent, 0.0))

        # Proper model path
        arr = [[beds, baths, size, zip_enc, pt_enc]]
        df = pd.DataFrame(arr, columns=RENT_FEATURE_COLUMNS)

        X_scaled = self._artifacts.scaler.transform(df)
        # LightGBM Booster expects numpy array
        pred = self._artifacts.model.predict(X_scaled)[0]
        return float(max(pred, 0.0))
