# src/haven/domain/assumptions.py
from pydantic import BaseModel

class UnderwritingAssumptions(BaseModel):
    vacancy_rate: float
    maintenance_rate: float
    property_mgmt_rate: float
    capex_rate: float
    closing_cost_pct: float
    min_dscr_good: float
