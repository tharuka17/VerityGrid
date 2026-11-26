"""Predictable error responses for the whole API.

Every error uses one JSON shape (ErrorResponse in app/schemas/api.py):

    {"error": {"code": "...", "message": "...", "issues": [...]}}

`code` is stable and machine-readable; `message` is safe to show to a user. `issues` appears
only for validation errors. Messages never contain uploaded content: ingestion messages hold
positions and counts, validation issues hold a field location and a short rule text (never the
submitted value), and unexpected errors return a generic message.

Mapping:
    NotFoundError              404  not_found
    FileTooLargeError          413  file_too_large
    other IngestionError       422  the error's own code (malformed_csv, invalid_header, ...)
    ScanFailedError            500  scan_failed
    RequestValidationError     422  validation_error
    unknown route, wrong verb  404 / 405  not_found / method_not_allowed
    anything else              500  internal_error
"""

import logging
from http import HTTPStatus
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import get_settings
from app.schemas.api import ErrorDetail, ErrorResponse, ValidationIssue
from app.services.ingestion import FileTooLargeError, IngestionError
from app.services.query_service import NotFoundError
from app.services.scan_service import ScanFailedError

logger = logging.getLogger(__name__)

MAX_ISSUES = 20
MAX_LOCATION_LENGTH = 100
MAX_ISSUE_MESSAGE_LENGTH = 200

_HTTP_ERROR_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed"}


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    issues: list[ValidationIssue] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a response in the shared error shape."""
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, issues=issues))
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


# --- Domain errors -------------------------------------------------------------


async def _not_found(request: Request, exc: Exception) -> JSONResponse:
    return error_response(404, "not_found", str(exc))


async def _file_too_large(request: Request, exc: Exception) -> JSONResponse:
    return error_response(413, FileTooLargeError.code, str(exc))


async def _ingestion_error(request: Request, exc: Exception) -> JSONResponse:
    code = exc.code if isinstance(exc, IngestionError) else IngestionError.code
    return error_response(422, code, str(exc))


async def _scan_failed(request: Request, exc: Exception) -> JSONResponse:
    # Only the service's own message is returned: it names the exception class and the scan
    # run id, never data. The original exception is chained to it for local debugging only.
    return error_response(500, "scan_failed", str(exc))


# --- Framework errors ----------------------------------------------------------


def _format_location(location: tuple[object, ...]) -> str:
    return ".".join(str(part) for part in location)[:MAX_LOCATION_LENGTH]


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    # Only the location and the rule text are used. The submitted value is never echoed.
    issues = [
        ValidationIssue(
            location=_format_location(tuple(item.get("loc", ()))),
            message=str(item.get("msg", "Invalid value"))[:MAX_ISSUE_MESSAGE_LENGTH],
        )
        for item in cast(RequestValidationError, exc).errors()[:MAX_ISSUES]
    ]
    return error_response(422, "validation_error", "The request failed validation.", issues=issues)


def _status_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "HTTP error"


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    http_exc = cast(StarletteHTTPException, exc)
    status_code = http_exc.status_code
    return error_response(
        status_code,
        _HTTP_ERROR_CODES.get(status_code, f"http_{status_code}"),
        _status_phrase(status_code),  # a standard phrase, not the exception's detail text
        headers=http_exc.headers,
    )


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    # Log the route template and the exception class only. Exception messages can contain data
    # values, so a traceback is written only when DEBUG is enabled (local development).
    route = request.scope.get("route")
    logger.error(
        "Unhandled error: method=%s route=%s error_type=%s",
        request.method,
        getattr(route, "path", "unmatched"),
        type(exc).__name__,
        exc_info=exc if get_settings().debug else None,
    )
    return error_response(500, "internal_error", "An unexpected error occurred.")


def register_exception_handlers(app: FastAPI) -> None:
    """Install every handler. Starlette picks the most specific class for each exception."""
    app.add_exception_handler(NotFoundError, _not_found)
    app.add_exception_handler(FileTooLargeError, _file_too_large)
    app.add_exception_handler(IngestionError, _ingestion_error)
    app.add_exception_handler(ScanFailedError, _scan_failed)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled)