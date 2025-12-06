# src/haven/adapters/flip_classifier.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from haven.adapters.logging_utils import get_logger

logger = get_logger(__name__)


class FlipClassifier:
    """
    Thin wrapper around a LightGBM binary classifier trained by
    entrypoints/cli/audit_flip_classifier.py.

    Expects a joblib bundle with keys:
      - "feature_names": List[str]
      - "model": sklearn-compatible classifier with predict_proba
    """

    def __init__(
        self,
        model_path: str | Path = "models/flip_classifier_lgb.joblib",
    ) -> None:
        self.model_path = Path(model_path)
        self.is_ready: bool = False
        self.feature_names: List[str] = []
        self.model: Any | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            logger.info(
                "flip_classifier_model_missing",
                extra={"path": str(self.model_path)},
            )
            return

        try:
            bundle = joblib.load(self.model_path)
        except Exception as exc:
            logger.exception(
                "flip_classifier_load_failed",
                extra={"path": str(self.model_path), "error": str(exc)},
            )
            return

        self.model = bundle.get("model")
        feature_names = bundle.get("feature_names") or []

        if self.model is None or not feature_names:
            logger.warning(
                "flip_classifier_bundle_invalid",
                extra={"path": str(self.model_path)},
            )
            return

        self.feature_names = list(feature_names)
        self.is_ready = True

        logger.info(
            "flip_classifier_loaded",
            extra={
                "path": str(self.model_path),
                "n_features": len(self.feature_names),
            },
        )

    def predict_proba_one(self, features: Dict[str, float]) -> float | None:
        """
        Predict probability that a given deal is a "good flip".

        Parameters
        ----------
        features:
            Dict with keys matching the feature_names used during training.
            Missing features are filled with 0.0.

        Returns
        -------
        Probability in [0,1] or None on failure.
        """
        if not self.is_ready or self.model is None:
            return None

        try:
            row = np.array(
                [[float(features.get(name, 0.0)) for name in self.feature_names]],
                dtype=float,
            )

            proba = self.model.predict_proba(row)[0, 1]
            return float(proba)
        except Exception as exc:
            logger.exception(
                "flip_classifier_predict_failed",
                extra={"error": str(exc)},
            )
            return None
