# src/haven/services/drift.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class DriftResult:
    feature: str
    psi: float
    warning: bool
    alert: bool


def _psi_for_feature(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index for one feature.
    """
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(baseline, quantiles))
    if len(cuts) <= 2:
        return 0.0

    def _bucket_counts(x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(cuts, x, side="right") - 1
        idx = np.clip(idx, 0, len(cuts) - 2)
        counts = np.bincount(idx, minlength=len(cuts) - 1).astype(float)
        return counts / max(counts.sum(), eps)

    p = _bucket_counts(baseline)
    q = _bucket_counts(current)

    psi = np.sum((q - p) * np.log((q + eps) / (p + eps)))
    return float(max(psi, 0.0))


def compute_psi(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    features: Iterable[str],
    warn_threshold: float = 0.1,
    alert_threshold: float = 0.25,
) -> Dict[str, DriftResult]:
    """
    Compare baseline vs. current feature distributions.

    Typical interpretation:
      - psi < 0.1  → stable
      - 0.1-0.25   → moderate drift (watch)
      - > 0.25     → major drift (retrain / investigate)
    """
    results: Dict[str, DriftResult] = {}

    for f in features:
        if f not in baseline.columns or f not in current.columns:
            continue

        psi = _psi_for_feature(
            baseline[f].to_numpy(dtype=float),
            current[f].to_numpy(dtype=float),
        )
        results[f] = DriftResult(
            feature=f,
            psi=psi,
            warning=psi >= warn_threshold,
            alert=psi >= alert_threshold,
        )

    return results
