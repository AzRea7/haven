# src/haven/adapters/rehab_estimator.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RehabEstimatorConfig:
    """
    Simple heuristic rehab model for flip underwriting.

    You can later swap this out for an ML model; keep the interface the same.
    """
    base_cost_per_sqft: float = 35.0
    old_house_year: int = 1970
    old_house_multiplier: float = 1.25
    luxury_sqft_threshold: int = 2500
    luxury_multiplier: float = 1.15
    min_rehab: float = 15000.0
    max_rehab: float = 250000.0


class RehabEstimator:
    """
    Estimate rehab budget given basic listing features.

    Inputs:
      - sqft (required-ish)
      - year_built (optional)
      - needs_gut_renovation (optional)

    Output:
      - rehab budget in dollars.
    """

    def __init__(self, cfg: RehabEstimatorConfig | None = None) -> None:
        self.cfg = cfg or RehabEstimatorConfig()

    def estimate(
        self,
        sqft: float | None,
        year_built: int | None = None,
        needs_gut_renovation: bool | None = None,
    ) -> float:
        # Fallback if we have no usable sqft
        if not sqft or sqft <= 0:
            return self.cfg.min_rehab

        cost_per_sqft = self.cfg.base_cost_per_sqft

        # Older houses: knob-and-tube, lead, weird walls, etc.
        if year_built and year_built < self.cfg.old_house_year:
            cost_per_sqft *= self.cfg.old_house_multiplier

        # Big / “luxury” houses tend to have more expensive finishes per sqft
        if sqft >= self.cfg.luxury_sqft_threshold:
            cost_per_sqft *= self.cfg.luxury_multiplier

        # Optional “full gut” toggle if you feed it from UI later
        if needs_gut_renovation:
            cost_per_sqft *= 1.5

        raw = sqft * cost_per_sqft

        # Clamp to a sane band
        return float(
            max(self.cfg.min_rehab, min(raw, self.cfg.max_rehab))
        )
