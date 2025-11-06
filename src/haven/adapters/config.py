from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    # App & logging
    ENV: str = Field(default="dev")
    LOG_LEVEL: str = Field(default="INFO")

    # (future) persistence
    DB_URI: str = Field(default="sqlite:///haven.db")

    # Underwriting default assumptions
    VACANCY_RATE: float = Field(default=0.05)
    MAINTENANCE_RATE: float = Field(default=0.08)
    PROPERTY_MGMT_RATE: float = Field(default=0.10)
    CAPEX_RATE: float = Field(default=0.05)

    DEFAULT_CLOSING_COST_PCT: float = Field(default=0.03)
    MIN_DSCR_GOOD: float = Field(default=1.20)

    # pydantic v2 settings config (env support, case-insensitive, etc.)
    model_config = SettingsConfigDict(
        env_prefix="HAVEN_",           # e.g., HAVEN_VACANCY_RATE=0.06
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator(
        "VACANCY_RATE", "MAINTENANCE_RATE", "PROPERTY_MGMT_RATE", "CAPEX_RATE",
        "DEFAULT_CLOSING_COST_PCT", mode="before"
    )
    @classmethod
    def _to_non_negative_fraction(cls, v):
        # accept strings like "8%" or "0.08" or numbers; normalize to fraction
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
    def _dscr_positive(cls, v):
        f = float(v)
        if f <= 0:
            raise ValueError("MIN_DSCR_GOOD must be > 0")
        return f

config = AppConfig()
