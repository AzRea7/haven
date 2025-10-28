import pandas as pd

def normalize_sold(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["property_id","lat","lon","zip","beds","baths","sqft","year_built",
            "list_price","sold_price","sold_date","dom","property_type"]
    df = df[keep].copy()
    df["sold_date"] = pd.to_datetime(df["sold_date"], errors="coerce")
    df = df.dropna(subset=["sold_price","sqft","sold_date","zip"])
    df = df[df["sold_price"] > 0]
    df["psf"] = df["sold_price"] / df["sqft"].clip(lower=300)
    return df
