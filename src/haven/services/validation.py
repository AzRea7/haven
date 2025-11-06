from typing import Any

REQUIRED_FIELDS = [
    "property_type",
    "address", "city", "state", "zipcode",
    "list_price",
    "down_payment_pct",
    "interest_rate_annual",
    "loan_term_years",
    "taxes_annual",
    "insurance_annual",
]

def _to_num(val, field_name: str) -> float:
    """
    Coerce values like "250000", "6.5", "6.5%", 0.07, etc. to float.
    """
    if val is None:
        raise ValueError(f"Missing required numeric field: {field_name}")
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(val.strip().replace("%", ""))
    raise ValueError(f"Invalid type for {field_name}: {type(val)}")

def _to_num_optional(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(val.strip().replace("%", ""))
    return 0.0

def validate_and_prepare_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """
    - Ensure required core fields exist.
    - Normalize numeric/percent fields.
    - Provide sane defaults for optional fields like hoa_monthly.
    - Leave units as list[dict] for now; we'll cast to Unit in deal_analyzer.
    """
    for field in REQUIRED_FIELDS:
        if field not in raw:
            raise ValueError(f"Missing required field: {field}")

    cleaned = dict(raw)

    cleaned["list_price"] = _to_num(raw["list_price"], "list_price")
    cleaned["down_payment_pct"] = _to_num(raw["down_payment_pct"], "down_payment_pct")
    cleaned["interest_rate_annual"] = _to_num(raw["interest_rate_annual"], "interest_rate_annual")
    cleaned["loan_term_years"] = int(raw["loan_term_years"])
    cleaned["taxes_annual"] = _to_num(raw["taxes_annual"], "taxes_annual")
    cleaned["insurance_annual"] = _to_num(raw["insurance_annual"], "insurance_annual")

    cleaned["hoa_monthly"] = _to_num_optional(raw.get("hoa_monthly"))
    cleaned["sqft"] = _to_num_optional(raw.get("sqft"))

    # Fix % style mistakes: someone types 25 instead of 0.25 for down_payment_pct
    if cleaned["down_payment_pct"] > 1.0:
        cleaned["down_payment_pct"] /= 100.0
    if cleaned["interest_rate_annual"] > 1.0:
        cleaned["interest_rate_annual"] /= 100.0

    # Optional rent for single-door assets
    if "est_market_rent" in raw:
        cleaned["est_market_rent"] = _to_num_optional(raw.get("est_market_rent"))

    return cleaned
