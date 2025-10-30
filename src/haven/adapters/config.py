from pydantic import BaseSettings, Field, validator

class AppConfig(BaseSettings):
    ENV: str = Field(default="dev")
    LOG_LEVEL: str = Field(default="INFO")

    # Database connection string if you're persisting deals
    DB_URI: str = Field(default="sqlite:///haven.db")

    # Global underwriting assumptions.
    # These are industry-standard rule-of-thumb defaults.
    VACANCY_RATE: float = Field(default=0.05, description="5% vacancy assumption")
    MAINTENANCE_RATE: float = Field(default=0.08, description="8% of rent to repairs/maintenance")
    PROPERTY_MGMT_RATE: float = Field(default=0.10, description="10% of rent to property mgmt")
    CAPEX_RATE: float = Field(default=0.05, description="5% of rent to long-term capital reserves")

    DEFAULT_CLOSING_COST_PCT: float = Field(
        default=0.03,
        description="Assume 3% of purchase price in closing costs"
    )

    MIN_DSCR_GOOD: float = Field(
        default=1.20,
        description="Banks like DSCR >= 1.20"
    )

    @validator("VACANCY_RATE", "MAINTENANCE_RATE", "PROPERTY_MGMT_RATE", "CAPEX_RATE")
    def _non_negative(cls, v):
        if v < 0:
            raise ValueError("rate must be non-negative")
        return v

config = AppConfig()
