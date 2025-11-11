# src/haven/services/validation.py

from typing import Any

# Core fields that are truly required to reason about a deal
REQUIRED_CORE_FIELDS = [
    "property_type",
    "address",
    "city",
    "state",
    "zipcode",
    "list_price",
]

# Defaults aligned with README + api/http.py
DEFAULT_DOWN_PAYMENT_PCT = 0.25
DEFAULT_INTEREST_RATE = 0.065
DEFAULT_LOAN_TERM_YEARS = 30
DEFAULT_TAXES_ANNUAL = 3000.0
DEFAULT_INSURANCE_ANNUAL = 1200.0


def _to_num(val: Any, field_name: str) -> float:
    """
    Coerce values like:
      - 250000
      - "250000"
      - "6.5"
      - "6.5%"
      - 0.065
    into float.
    """
    if val is None:
        raise ValueError(f"Missing required numeric field: {field_name}")
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if s.endswith("%"):
            # strip '%' but leave normalization decision to caller
            s = s[:-1]
        return float(s)
    raise ValueError(f"Invalid type for {field_name}: {type(val)}")


def _to_num_optional(val: Any) -> float:
    """
    Lenient converter for optional numeric fields.
    Returns 0.0 when missing/blank/garbage.
    """
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return 0.0
        if s.endswith("%"):
            s = s[:-1]
        try:
            return float(s)
        except ValueError:
            return 0.0
    return 0.0


def validate_and_prepare_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize incoming payload for deal analysis.

    Responsibilities:
      - Ensure core address/price/type fields exist.
      - Normalize numeric/percent fields.
      - Apply sane underwriting defaults when values are omitted
        (this is required for tests using analyze_deal_with_defaults).
      - Keep `units` as list[dict] (casting to Unit happens in deal_analyzer).
    """
    # 1. Check core required fields
    for field in REQUIRED_CORE_FIELDS:
        if field not in raw:
            raise ValueError(f"Missing required field: {field}")

    cleaned: dict[str, Any] = dict(raw)

    # 2. Required core numeric: list_price
    cleaned["list_price"] = _to_num(raw["list_price"], "list_price")

    # 3. Underwriting inputs with defaults if missing

    # down_payment_pct
    dp_raw = raw.get("down_payment_pct", DEFAULT_DOWN_PAYMENT_PCT)
    dp = _to_num(dp_raw, "down_payment_pct")
    # If someone passes 25 instead of 0.25, normalize.
    if dp > 1.0:
        dp /= 100.0
    cleaned["down_payment_pct"] = dp

    # interest_rate_annual
    ir_raw = raw.get("interest_rate_annual", DEFAULT_INTEREST_RATE)
    ir = _to_num(ir_raw, "interest_rate_annual")
    if ir > 1.0:
        ir /= 100.0
    cleaned["interest_rate_annual"] = ir

    # loan_term_years
    lt_raw = raw.get("loan_term_years", DEFAULT_LOAN_TERM_YEARS)
    try:
        cleaned["loan_term_years"] = int(lt_raw)
    except (TypeError, ValueError):
        raise ValueError("Invalid loan_term_years")

    # taxes_annual
    tax_raw = raw.get("taxes_annual", DEFAULT_TAXES_ANNUAL)
    cleaned["taxes_annual"] = _to_num(tax_raw, "taxes_annual")

    # insurance_annual
    ins_raw = raw.get("insurance_annual", DEFAULT_INSURANCE_ANNUAL)
    cleaned["insurance_annual"] = _to_num(ins_raw, "insurance_annual")

    # 4. Optional fields

    # HOA monthly default = 0 if missing
    cleaned["hoa_monthly"] = _to_num_optional(raw.get("hoa_monthly"))

    # Property size optional (used for psf-style pricing, etc)
    cleaned["sqft"] = _to_num_optional(raw.get("sqft"))

    # Optional single-door rent
    if "est_market_rent" in raw:
        cleaned["est_market_rent"] = _to_num_optional(raw.get("est_market_rent"))

    # units: leave as-is (deal_analyzer will turn list[dict] -> list[Unit])
    # if "units" in raw: cleaned["units"] = raw["units"]

    return cleaned
