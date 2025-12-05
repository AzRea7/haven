# scripts/debug_rehab_estimator.py
import pandas as pd

from haven.adapters.rehab_estimator import RehabEstimator


def main() -> None:
    df = pd.read_parquet("data/curated/properties.parquet")

    est = RehabEstimator()

    sample = df.head(15).copy()

    def estimate_row(row):
        sqft = float(row.get("sqft") or 0.0)
        year = row.get("year_built")
        year_built = int(year) if year not in (None, "") else None
        return est.estimate(sqft=sqft, year_built=year_built)

    sample["rehab_model_est"] = sample.apply(estimate_row, axis=1)

    cols = [c for c in ["address","city","state","zip","sqft","year_built","rehab_model_est"] if c in sample.columns]

    print("=== Sample rehab estimates ===")
    print(sample[cols])

    print("\nSummary stats:")
    print(sample["rehab_model_est"].describe())


if __name__ == "__main__":
    main()
