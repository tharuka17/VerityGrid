import logging
from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ENVIRONMENT values that mean a real deployment. Shared with app/api/access.py.
PRODUCTION_ENVIRONMENTS = frozenset({"production", "prod"})


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and an optional .env file.

    Real environment variables take precedence over values in .env.
    Field names map to variable names case-insensitively
    (for example, database_url is read from DATABASE_URL).
    """

    app_name: str = "VerityGrid"
    environment: str = "development"
    debug: bool = False
    database_url: str = "sqlite:///./veritygrid.db"

    # Upload limits applied by POST /scans (Phase 5C). Defaults match IngestionLimits.
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1)
    max_upload_rows: int = Field(default=50_000, ge=1)
    max_upload_columns: int = Field(default=200, ge=1)

    # AI classification (Phase 6B). The mock provider is the default and needs no key.
    ai_provider: Literal["mock", "gemini"] = "mock"
    gemini_api_key: SecretStr | None = None  # never printed or logged; blank means "not set"
    gemini_model: str = ""  # no built-in default: choose a current ID from Google's models page
    gemini_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)
    gemini_max_attempts: int = Field(default=2, ge=1, le=3)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _debug_is_ignored_in_production(self) -> Self:
        """DEBUG writes tracebacks, and exception text can contain data values: never in production."""
        if self.debug and self.environment.strip().lower() in PRODUCTION_ENVIRONMENTS:
            logger.error("DEBUG is ignored because ENVIRONMENT says production.")
            self.debug = False
        return self


@lru_cache
def get_settings() -> Settings:
    """Return one shared Settings instance for the whole process."""
    return Settings()