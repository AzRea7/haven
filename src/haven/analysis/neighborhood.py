# src/haven/analysis/neighborhood.py
from __future__ import annotations


def adjust_rank_for_neighborhood(
    base_rank: float,
    walk_score: float | None = None,
    school_score: float | None = None,
    crime_index: float | None = None,
    rent_demand_index: float | None = None,
) -> float:
    """
    Convert 0-100 neighborhood scores into a small additive adjustment
    to rank_score.

    Convention:
      - higher walk_score, school_score, rent_demand_index = better
      - higher crime_index = worse
    """
    rank = float(base_rank)

    def nz(x: float | None) -> float:
        return float(x) if x is not None else 50.0  # neutral mid-point

    walk = nz(walk_score)
    schools = nz(school_score)
    crime = nz(crime_index)
    rent_dem = nz(rent_demand_index)

    # Scale from [-5, +5] roughly
    rank += (walk - 50.0) / 10.0
    rank += (schools - 50.0) / 10.0
    rank -= (crime - 50.0) / 10.0
    rank += (rent_dem - 50.0) / 10.0

    return rank
