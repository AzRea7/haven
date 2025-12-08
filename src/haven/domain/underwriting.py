from dataclasses import dataclass
from typing import Literal, Dict, List, Optional

Strategy = Literal["rental", "flip"]

@dataclass
class ScenarioMetrics:
    arv: float              # scenario ARV
    rent: float             # monthly rent
    noi: float              # annual net operating income
    dscr: float             # debt-service coverage ratio
    coc: float              # cash-on-cash return (annual)
    cap_rate: float         # NOI / purchase price
    breakeven_occ: float    # breakeven occupancy %
    monthly_cashflow: float # monthly cashflow after all costs

@dataclass
class DealEvaluation:
    address: str
    city: str
    state: str
    zipcode: str
    list_price: float
    strategy: Strategy

    # Scenarios
    downside: ScenarioMetrics
    base: ScenarioMetrics
    upside: ScenarioMetrics

    # Labels & confidence
    label: Literal["buy", "maybe", "pass"]
    risk_tier: Literal["low", "medium", "high"]
    confidence: float  # 0.0–1.0

    # Model info (for transparency)
    arv_quantiles: Dict[str, float]  # {"p10":..., "p50":..., "p90":...}
    rent_quantiles: Dict[str, float]
    model_versions: Dict[str, str]   # {"arv": "...", "rent": "...", "flip": "..."}

    # Diagnostics
    warnings: List[str]
    hard_flags: List[str]            # reasons this can never be "buy"
