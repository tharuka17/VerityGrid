"""Access mode for write operations.

  development  Uploads, scans and reviews are OPEN to anyone who can reach the server. For local work only.
  demo         The deployment is read-only: every request that is not GET, HEAD or OPTIONS is rejected with
               HTTP 403 and the shared error shape. This is the default.

The mode comes from the environment variable VERITYGRID_MODE (or the project's .env file). A missing or
invalid value means demo: the check fails closed. Development mode is also REFUSED (demo is used instead) when
ENVIRONMENT says production, because those are two separate settings and a mismatch must not open public writes.
This is deny-by-method, not authentication: it has no users, tokens or roles, and it does not make the open
mode safe to expose.

The middleware is added last in app/main.py, so it runs first, before the upload-size check or any body is
read. announce_mode() is called at startup (the middleware stack itself is only built on the first request).
Logs hold only the HTTP method, never the path, the body or a configured value.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import PRODUCTION_ENVIRONMENTS, get_settings

logger = logging.getLogger(__name__)

DEVELOPMENT = "development"
DEMO = "demo"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
READ_ONLY_CODE = "read_only_mode"
READ_ONLY_MESSAGE = "This deployment is read-only. Uploads, scans and reviews are disabled."


class SecuritySettings(BaseSettings):
    """Reads only VERITYGRID_MODE; other VERITYGRID_* variables and everything else in .env are ignored."""

    mode: Literal["development", "demo"] = DEMO

    model_config = SettingsConfigDict(
        env_prefix="VERITYGRID_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


def read_mode(env_file: str | None = ".env") -> str:
    """The configured mode. Any problem with the configuration means demo (fail closed).

    Only the exception class is logged, never the configured value.
    """
    try:
        return SecuritySettings(_env_file=env_file).mode
    except Exception as error:
        logger.error("Invalid access mode configuration, using demo: error_type=%s", type(error).__name__)
        return DEMO


@lru_cache(maxsize=1)
def load_mode() -> str:
    """The effective mode, read once per process (restart the API after changing it). Tests clear the cache."""
    mode = read_mode()
    if mode == DEVELOPMENT and get_settings().environment.strip().lower() in PRODUCTION_ENVIRONMENTS:
        logger.error(
            "Development mode is refused because ENVIRONMENT says production: using demo (read-only)."
        )
        return DEMO
    return mode


def announce_mode() -> str:
    """Log the effective mode once at startup, so an operator can see which one is active."""
    mode = load_mode()
    if mode == DEVELOPMENT:
        logger.warning(
            "Access mode: development. Uploads, scans and reviews are OPEN to anyone who can reach "
            "this server. Use it for local work only."
        )
    else:
        logger.info("Access mode: demo (read-only). Uploads, scans and reviews are disabled.")
    return mode


def check_demo_database(database_url: str) -> None:
    """In demo mode, refuse to start when the SQLite file does not exist.

    Without this check a wrong DATABASE_URL silently creates an EMPTY database, and the demo looks empty instead
    of failing. Development mode and other databases (in-memory, PostgreSQL) are not checked. The message holds
    neither the path nor the URL.
    """
    if load_mode() != DEMO:
        return
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or url.database in (None, "", ":memory:"):
        return
    if not Path(url.database).is_file():
        raise RuntimeError(
            "Demo mode needs a seeded SQLite database, but the file named by DATABASE_URL does not exist. "
            "Run `python -m scripts.seed_demo` and point DATABASE_URL at the file it creates, "
            "or set VERITYGRID_MODE=development for local work."
        )
        

class ReadOnlyModeMiddleware:
    """Reject every non-safe HTTP method unless the effective mode is development."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in SAFE_METHODS or load_mode() == DEVELOPMENT:
            await self.app(scope, receive, send)
            return
        method = str(scope["method"])
        logger.warning(
            "Write request rejected in read-only mode: method=%s",
            method if method.isalpha() and len(method) <= 16 else "other",
        )
        response = JSONResponse(
            {"error": {"code": READ_ONLY_CODE, "message": READ_ONLY_MESSAGE}}, status_code=403
        )
        await response(scope, receive, send)