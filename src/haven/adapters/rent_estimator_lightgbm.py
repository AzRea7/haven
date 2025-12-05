# src/haven/adapters/rent_estimator_lightgbm.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib
import numpy as np
import pandas as pd


RENT_MODEL_PATH = Path("models/rent_quantiles.pkl")
RENT_TRAINING_PATH = Path("data/curated/rent_training.parquet")


@dataclass
class _RentArtifacts:
    models: dict[float, Any]
    feature_names: list[str]


class LightGBMRentEstimator:
    """
    Rent estimator with two layers:

    1) If LightGBM quantile models exist on disk (models/rent_quantiles.pkl),
       use them to predict q10 / q50 / q90 for rent.
    2) Otherwise, use a heuristic rule of thumb.

    On top of that, we apply a ZIP-level realism layer:

        - Load rent_training.parquet (if available).
        - Compute median rent-per-sqft per zipcode.
        - Cap predicted rent at 1.4x the ZIP median for a same-sized unit.

      This prevents pathological combos like a $30k house with $1,800 rent.
    """

    def __init__(self) -> None:
        self._artifacts: Optional[_RentArtifacts] = None
        self._zip_rent_psf: dict[str, float] = {}
        self.is_ready: bool = False

        self._load_artifacts_if_available()
        self._load_zip_rent_stats()
        # "Ready" if either ML model or heuristic + zip stats exist
        self.is_ready = True

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def _load_artifacts_if_available(self) -> None:
        if not RENT_MODEL_PATH.exists():
            self._artifacts = None
            return

        obj = joblib.load(RENT_MODEL_PATH)
        # Expected structure from train_rent_quantiles.py:
        #   {"models": {0.10: model, 0.50: model, 0.90: model},
        #    "feature_names": ["bedrooms", "bathrooms", ...]}
        models = obj.get("models", {})
        feature_names = obj.get("feature_names", [])

        if not models or not feature_names:
            self._artifacts = None
            return

        self._artifacts = _RentArtifacts(models=models, feature_names=feature_names)

    def _load_zip_rent_stats(self) -> None:
        """
        Build a zipcode -> median_rent_per_sqft map from rent_training.parquet.

        This does NOT need to be perfect – it's a realism prior.
        Any future improvements to rent_training.parquet will automatically
        make this more accurate.
        """
        if not RENT_TRAINING_PATH.exists():
            self._zip_rent_psf = {}
            return

        try:
            df = pd.read_parquet(RENT_TRAINING_PATH)
        except Exception:
            self._zip_rent_psf = {}
            return

        # We expect at least: ["zipcode", "sqft", "rent"]
        cols = {"zipcode", "sqft", "rent"}
        if not cols.issubset(df.columns):
            self._zip_rent_psf = {}
            return

        work = df.copy()
        work = work.dropna(subset=["zipcode", "sqft", "rent"])
        work = work[work["sqft"] > 0]
        work = work[work["rent"] > 0]

        if work.empty:
            self._zip_rent_psf = {}
            return

        work["rent_psf"] = work["rent"] / work["sqft"]

        grouped = (
            work.groupby("zipcode", dropna=True)["rent_psf"]
            .median()
            .rename("median_rent_psf")
        )

        self._zip_rent_psf = {
            str(z): float(psf) for z, psf in grouped.dropna().items() if psf > 0
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_quantiles_ml(
        self,
        bedrooms: float,
        bathrooms: float,
        sqft: float,
        zipcode_encoded: float,
        property_type_encoded: float,
        is_small_unit: float,
    ) -> Optional[Mapping[float, float]]:
        if self._artifacts is None:
            return None

        X = pd.DataFrame(
            [
                dict(
                    bedrooms=float(bedrooms),
                    bathrooms=float(bathrooms),
                    sqft=float(sqft),
                    zipcode_encoded=float(zipcode_encoded),
                    property_type_encoded=float(property_type_encoded),
                    is_small_unit=float(is_small_unit),
                )
            ],
            columns=self._artifacts.feature_names,
        )

        out: dict[float, float] = {}
        for q, model in self._artifacts.models.items():
            yhat = model.predict(X)
            out[q] = float(np.clip(yhat[0], 0.0, np.inf))

        return out

    def _heuristic_rent(
        self,
        bedrooms: float,
        bathrooms: float,
        sqft: float,
        zipcode: str,
        property_type: str,
    ) -> float:
        """
        Simple rule-of-thumb rent estimator used when no ML model is present.

        Rough idea:
          - base rent per sqft anchored by bedroom/bath mix
          - moderate uplift for 'good' zips
          - mild penalty for very small units
        """
        beds = float(bedrooms or 0)
        baths = float(bathrooms or 0)
        size = max(float(sqft or 0), 350.0)  # guard against 0 / tiny

        # Base rent per sqft – intentionally conservative to avoid crazy rents.
        base_psf = 1.10  # $/sqft baseline

        # Bedrooms / baths bumps
        if beds >= 3:
            base_psf += 0.10
        if beds >= 4:
            base_psf += 0.05
        if baths >= 2:
            base_psf += 0.05

        # Small unit premium capped
        if size < 700:
            base_psf += 0.10

        # Very large units slightly cheaper per sqft
        if size > 2200:
            base_psf -= 0.05

        # Very rough ZIP-based uplift for strong inner-ring / downtown areas
        zip_str = str(zipcode or "")
        if zip_str.startswith("48"):
            base_psf += 0.05

        # Property-type nudge
        if property_type in {"apartment", "condo_townhome"}:
            base_psf += 0.05

        base_psf = max(base_psf, 0.80)  # don't go totally off a cliff

        rent = base_psf * size
        return float(max(rent, 300.0))

    def _apply_zip_cap(self, zipcode: str, sqft: float, raw_rent: float) -> float:
        """
        Apply ZIP-level realism cap:

            cap_rent = 1.4 * median_rent_psf(zip) * max(sqft, 400)

        If we don't have stats for this ZIP, we leave the rent as-is.
        """
        if raw_rent <= 0:
            return 0.0

        zip_str = str(zipcode or "")
        psf = self._zip_rent_psf.get(zip_str)
        if psf is None or psf <= 0:
            return float(raw_rent)

        effective_sqft = max(float(sqft or 0.0), 400.0)
        zip_median_rent = psf * effective_sqft
        cap_rent = 1.4 * zip_median_rent

        return float(min(raw_rent, cap_rent))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_unit_rent(
        self,
        bedrooms: float,
        bathrooms: float,
        sqft: float,
        zipcode: str,
        property_type: str,
        zipcode_encoded: float | None = None,
        property_type_encoded: float | None = None,
    ) -> float:
        """
        Predict monthly rent for a *single* unit.

        If ML quantile models are available, we take q50 as the base prediction.
        Otherwise we fall back to the heuristic.

        In both cases, we then apply the ZIP-level realism cap to avoid
        pathological overestimates.
        """
        size = float(sqft or 0.0)
        beds = float(bedrooms or 0.0)
        baths = float(bathrooms or 0.0)

        # Small-unit flag consistent with train_rent_quantiles.py
        is_small_unit = 1.0 if size < 700 else 0.0

        rent_raw: float

        if (
            self._artifacts is not None
            and zipcode_encoded is not None
            and property_type_encoded is not None
        ):
            q = self._predict_quantiles_ml(
                bedrooms=beds,
                bathrooms=baths,
                sqft=size,
                zipcode_encoded=float(zipcode_encoded),
                property_type_encoded=float(property_type_encoded),
                is_small_unit=is_small_unit,
            )
            if q:
                rent_raw = float(q.get(0.50, 0.0))
            else:
                rent_raw = self._heuristic_rent(beds, baths, size, zipcode, property_type)
        else:
            rent_raw = self._heuristic_rent(beds, baths, size, zipcode, property_type)

        rent_capped = self._apply_zip_cap(zipcode=zipcode, sqft=size, raw_rent=rent_raw)
        return float(max(rent_capped, 0.0))
