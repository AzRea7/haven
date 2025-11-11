import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from haven.adapters.config import config
from haven.adapters.sql_repo import SqlPropertyRepository

OUT_DIR = Path("data/curated")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------ DB + ZIP HELPERS ------------------


def get_engine():
    """Create a SQLAlchemy engine from the configured DB URI."""
    return create_engine(config.DB_URI)


def get_distinct_zips() -> list[str]:
    """
    Return distinct ZIP codes from the properties table.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT zipcode "
                "FROM properties "
                "WHERE zipcode IS NOT NULL"
            )
        ).fetchall()
    return [r[0] for r in rows if r[0]]


# ------------------ PER-ZIP FEATURE BUILD ------------------


def build_features_for_zip(zipcode: str) -> pd.DataFrame:
    """
    Build a feature dataframe for all properties in a given ZIP:
    - Pulls rows via SqlPropertyRepository.search
    - Derives est_rent, NOI, cap_rate
    - Encodes zipcode and property_type
    """
    repo = SqlPropertyRepository(uri=config.DB_URI)
    props = repo.search(zipcode=zipcode, limit=100_000)

    rows = []
    for p in props:
        price = float(getattr(p, "list_price", 0.0) or 0.0)
        taxes = float(getattr(p, "taxes", 0.0) or 0.0)
        hoa = float(getattr(p, "hoa_fee", 0.0) or 0.0)
        bedrooms = getattr(p, "bedrooms", None)
        bathrooms = getattr(p, "bathrooms", None)
        sqft = getattr(p, "sqft", None)
        property_type = getattr(p, "property_type", None)
        zipcode_val = getattr(p, "zipcode", zipcode)

        # est_rent: real if present, else heuristic
        est_rent_raw = getattr(p, "est_rent", None)
        if est_rent_raw is None or float(est_rent_raw or 0.0) <= 0.0:
            est_rent = price * 0.008 if price > 0 else 0.0
        else:
            est_rent = float(est_rent_raw)

        noi = est_rent * 12 - (taxes + hoa)
        cap_rate = noi / price if price > 0 else 0.0

        rows.append(
            {
                "id": getattr(p, "id", None),
                "address": getattr(p, "address", None),
                "zipcode": zipcode_val,
                "list_price": price,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft": sqft,
                "property_type": property_type,
                "est_rent": est_rent,
                "taxes": taxes,
                "hoa_fee": hoa,
                "noi": noi,
                "cap_rate": cap_rate,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Coerce numeric fields to numeric (LightGBM + sklearn friendly)
    for col in ["bedrooms", "bathrooms", "sqft"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Encodings
    df["zipcode_encoded"] = df["zipcode"].astype("category").cat.codes

    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].fillna("unknown").astype(str)
        df["property_type_encoded"] = (
            df["property_type"].astype("category").cat.codes
        )
    else:
        df["property_type"] = "unknown"
        df["property_type_encoded"] = -1

    return df


# ------------------ MAIN PIPELINE ------------------


def main() -> None:
    t0 = time.perf_counter()

    zips = get_distinct_zips()
    if not zips:
        print("No ZIPs found in properties table. Run ingestion first.")
        return

    print(f"Building features in parallel for {len(zips)} ZIPs...")

    dfs = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(build_features_for_zip, z): z for z in zips}
        for fut in as_completed(futures):
            df_zip = fut.result()
            if df_zip is not None and not df_zip.empty:
                z = df_zip["zipcode"].iloc[0]
                print(f"ZIP {z}: {len(df_zip)} rows")
                dfs.append(df_zip)

    if not dfs:
        print("No feature rows produced.")
        return

    final = pd.concat(dfs, ignore_index=True)

    # --- Robust flip_success label ---
    # Start from cap_rate distribution:
    cap = final["cap_rate"].fillna(0)

    # Try percentile-based threshold so we always get some positives:
    if (cap > 0).sum() > 0:
        thr = cap[cap > 0].quantile(0.7)
        flip = (cap >= thr).astype(int)
    else:
        # Degenerate: all zero cap_rate -> mark top 10% of NOI as successes
        noi = final["noi"].fillna(0)
        order = noi.sort_values(ascending=False).index
        flip = pd.Series(0, index=final.index)
        if len(order) > 0:
            k = max(1, len(order) // 10)
            flip.iloc[order[:k]] = 1

    # Guarantee both classes exist
    if flip.nunique() < 2:
        # Force at least one positive and one negative
        order = cap.sort_values(ascending=False).index
        flip = pd.Series(0, index=final.index)
        if len(order) > 0:
            flip.iloc[order[0]] = 1

    final["flip_success"] = flip.astype(int)

    # --- Save properties.parquet for flip model ---
    props_path = OUT_DIR / "properties.parquet"
    final.to_parquet(props_path, index=False)
    print(f"Wrote {len(final)} rows to {props_path}")

    # --- Save rent_training.parquet for rent quantile model ---
    # Use est_rent heuristic as 'rent' (numeric).
    rent_df = final.copy()
    rent_df = rent_df.rename(columns={"est_rent": "rent"})
    rent_path = OUT_DIR / "rent_training.parquet"
    rent_df.to_parquet(rent_path, index=False)
    print(f"Wrote {len(rent_df)} rows to {rent_path}")

    dt = time.perf_counter() - t0
    print(f"Feature build completed in {dt:.2f}s")


if __name__ == "__main__":
    main()
