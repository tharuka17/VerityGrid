"""Shared FastAPI dependencies and documentation helpers for the API routes."""

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Path, Query
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.api import ErrorResponse
from app.services.ingestion import IngestionLimits

# Largest value of a 32-bit signed integer: safe for SQLite and PostgreSQL integer columns.
# Ids above it are rejected with 422 instead of reaching the database.
MAX_ID = 2_147_483_647
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
MAX_OFFSET = 1_000_000

# One session per request. The dependency never commits; services commit explicitly.
DbSession = Annotated[Session, Depends(get_db)]

IdPath = Annotated[int, Path(ge=1, le=MAX_ID, description="Record id.")]
OptionalIdQuery = Annotated[int | None, Query(ge=1, le=MAX_ID)]


@dataclass(frozen=True, slots=True)
class PageParams:
    limit: int
    offset: int


def get_page_params(
    limit: Annotated[
        int, Query(ge=1, le=MAX_PAGE_SIZE, description="Maximum number of items to return.")
    ] = DEFAULT_PAGE_SIZE,
    offset: Annotated[
        int, Query(ge=0, le=MAX_OFFSET, description="Number of items to skip.")
    ] = 0,
) -> PageParams:
    return PageParams(limit=limit, offset=offset)


Pagination = Annotated[PageParams, Depends(get_page_params)]


def get_ingestion_limits() -> IngestionLimits:
    """Upload limits from settings (MAX_UPLOAD_BYTES, MAX_UPLOAD_ROWS, MAX_UPLOAD_COLUMNS)."""
    settings = get_settings()
    return IngestionLimits(
        max_bytes=settings.max_upload_bytes,
        max_rows=settings.max_upload_rows,
        max_columns=settings.max_upload_columns,
    )


Limits = Annotated[IngestionLimits, Depends(get_ingestion_limits)]

_ERROR_DESCRIPTIONS = {
    404: "The requested record does not exist.",
    413: "The uploaded file is larger than the configured limit.",
    422: "The request failed validation (parameters, body, or CSV content).",
    500: "The scan failed after its records were created. The failure is recorded on the scan.",
}


def error_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI documentation for error statuses, all using the shared ErrorResponse shape."""
    return {
        code: {"model": ErrorResponse, "description": _ERROR_DESCRIPTIONS[code]} for code in codes
    }