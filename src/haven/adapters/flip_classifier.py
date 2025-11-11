# src/haven/adapters/flip_classifier.py
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Iterable

import numpy as np

from .config import config
from .logging_utils import get_logger
from .model_io import safe_load

logger = get_logger(__name__)


DEFAULT_FLIP_MODEL_PATH = "models/flip_logit_calibrated.joblib"


class FlipClassifier:
    """
    Thin wrapper around a calibrated sklearn classifier.

    Expected artifact:
      - Trained + calibrated classifier saved via joblib.
      - Optionally has `feature_names_in_` for deterministic column order.
    """

    def __init__(self, model_path: str | None = None) -> None:
        mp = (
            model_path
            or getattr(config, "FLIP_MODEL_PATH", None)
            or DEFAULT_FLIP_MODEL_PATH
        )
        self.model_path = mp
        self.model = safe_load(mp)

        if self.model is None:
            logger.warning(
                "flip_classifier_not_loaded",
                extra={"context": {"path": mp}},
            )
        else:
            logger.info(
                "flip_classifier_loaded",
                extra={"context": {"path": mp}},
            )

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _vectorize(self, features: Mapping[str, float]) -> np.ndarray:
        m = self.model
        if m is None:
            return np.zeros((1, 1), dtype=float)

        # If the model exposes feature_names_in_, respect it.
        cols: Iterable[str]
        if hasattr(m, "feature_names_in_"):
            cols = list(m.feature_names_in_)  # type: ignore[attr-defined]
        else:
            # Fallback: sorted keys from features.
            cols = sorted(features.keys())

        row = [float(features.get(c, 0.0)) for c in cols]
        return np.asarray([row], dtype=float)

    def predict_proba_one(self, features: Mapping[str, float]) -> float:
        """
        Return probability that this property is a 'good flip' in [0, 1].

        If no model is loaded, returns 0.5 so callers can treat it as neutral.
        """
        if self.model is None:
            return 0.5

        x = self._vectorize(features)
        try:
            proba = self.model.predict_proba(x)[0, 1]
        except Exception as e:
            logger.warning(
                "flip_classifier_predict_failed",
                extra={"context": {"error": str(e)}},
            )
            return 0.5

        p = float(proba)
        logger.info(
            "flip_classifier_scored",
            extra={"context": {"p_good": p}},
        )
        return p
