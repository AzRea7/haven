import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from haven.adapters.sql_repo import SqlPropertyRepository
from haven.adapters.config import config


def get_distinct_zips(repo: SqlPropertyRepository) -> list[str]:
    # Adjust to your ORM / schema
    with repo.SessionLocal() as s:  # type: ignore[attr-defined]
        rows = s.execute("SELECT DISTINCT zipcode FROM properties").fetchall()
    return [r[0] for r in rows if r[0]]


def build_features_for_zip(zipcode: str) -> pd.DataFrame:
    repo = SqlPropertyRepository(uri=config.DB_URI)
    props = repo.search(zipcode=zipcode, limit=10_000)

    rows = []
    for p in props:
        # Example engineered features; match your existing formulas:
        # NOI, cap rate, DSCR inputs, etc.
        est_rent = getattr(p, "est_rent", None) or 0
        taxes = getattr(p, "taxes", 0) or 0
        hoa = getattr(p, "hoa_fee", 0) or 0
        price = float(p.list_price)

        noi = est_rent * 12 - (taxes + hoa)
        cap_rate = noi / price if price > 0 else 0.0

        rows.append(
            {
                "property_id": p.id,
                "zipcode": zipcode,
                "price": price,
                "est_rent": est_rent,
                "noi": noi,
                "cap_rate": cap_rate,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.perf_counter()
    repo = SqlPropertyRepository(uri=config.DB_URI)
    zips = get_distinct_zips(repo)

    print(f"Building features in parallel for {len(zips)} ZIPs...")

    dfs = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(build_features_for_zip, z): z for z in zips}
        for fut in as_completed(futures):
            df = fut.result()
            print(f"Completed ZIP {df['zipcode'].iloc[0]} "
                  f"({len(df)} rows)")
            dfs.append(df)

    if not dfs:
        print("No features built.")
        return

    final = pd.concat(dfs, ignore_index=True)
    final.to_parquet("data/curated/features.parquet", index=False)

    dt = time.perf_counter() - t0
    print(f"Saved {len(final)} rows to data/curated/features.parquet "
          f"in {dt:.2f}s")


if __name__ == "__main__":
    main()
