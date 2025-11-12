# src/haven/adapters/model_io.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib

from .logging_utils import get_logger

logger = get_logger(__name__)


def _resolve(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Model path is required.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model artifact not found: {p}")
    return p


def load_pickle(path: str | Path) -> Any:
    p = _resolve(path)
    logger.info(
        "model_io_load_pickle",
        extra={"context": {"path": str(p)}},
    )
    with p.open("rb") as f:
        return pickle.load(f)


def load_joblib(path: str | Path) -> Any:
    p = _resolve(path)
    logger.info(
        "model_io_load_joblib",
        extra={"context": {"path": str(p)}},
    )
    return joblib.load(p)


def safe_load(path: str | Path | None) -> Any | None:
    """
    Best-effort loader:
    - Returns None instead of exploding if file is missing/broken.
    - Tries joblib first, then pickle.
    """
    if path is None:
        return None

    try:
        s = str(path)
        if s.endswith(".joblib"):
            return load_joblib(path)
        if s.endswith(".pkl") or s.endswith(".pickle"):
            # Try joblib first to support joblib-saved .pkl, then fallback.
            try:
                return load_joblib(path)
            except Exception:
                return load_pickle(path)
        # Fallback generic
        return load_pickle(path)
    except FileNotFoundError:
        logger.warning(
            "model_io_missing_artifact",
            extra={"context": {"path": str(path)}},
        )
    except Exception as e:
        logger.warning(
            "model_io_failed_to_load",
            extra={"context": {"path": str(path), "error": str(e)}},
        )

    return None
