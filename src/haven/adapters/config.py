# src/haven/adapters/config.py
import os
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    # App & logging
    ENV: str = Field(default="dev")
    LOG_LEVEL: str = Field(default="INFO")

    # persistence
    DB_URI: str = Field(default="sqlite:///haven.db")

    # Underwriting default assumptions
    VACANCY_RATE: float = Field(default=0.05)
    MAINTENANCE_RATE: float = Field(default=0.08)
    PROPERTY_MGMT_RATE: float = Field(default=0.10)
    CAPEX_RATE: float = Field(default=0.05)

    DEFAULT_CLOSING_COST_PCT: float = Field(default=0.03)
    MIN_DSCR_GOOD: float = Field(default=1.20)

    # -----------------------------
    # RentCast integration
    # -----------------------------
    RENTCAST_API_KEY: str | None = Field(default=None)
    RENTCAST_BASE_URL: str = Field(default="https://api.rentcast.io/v1")

    # If true, RentCast becomes the runtime rent estimator for analysis
    RENTCAST_USE_FOR_RENT_ESTIMATES: bool = Field(default=False)

    # -----------------------------
    # Leads defaults
    # -----------------------------
    LEADS_DEFAULT_LIMIT: int = Field(default=200)

    model_config = SettingsConfigDict(
        env_prefix="HAVEN_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator(
        "VACANCY_RATE",
        "MAINTENANCE_RATE",
        "PROPERTY_MGMT_RATE",
        "CAPEX_RATE",
        "DEFAULT_CLOSING_COST_PCT",
        mode="before",
    )
    @classmethod
    def _to_non_negative_fraction(cls, v: Any) -> Any:
        if v is None:
            return v
        if isinstance(v, str):
            v = v.strip().replace("%", "")
        try:
            f = float(v)
        except Exception as err:
            raise ValueError("rate must be numeric or percent-like") from err
        if f > 1.0:
            f = f / 100.0
        if f < 0:
            raise ValueError("rate must be non-negative")
        return f

    @field_validator("MIN_DSCR_GOOD", mode="before")
    @classmethod
    def _dscr_positive(cls, v: Any) -> Any:
        f = float(v)
        if f <= 0:
            raise ValueError("MIN_DSCR_GOOD must be > 0")
        return f


config = AppConfig()
