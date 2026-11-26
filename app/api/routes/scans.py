"""Scan endpoints: upload a CSV and run a scan, and read a scan's status."""

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError

from app.api.deps import DbSession, IdPath, Limits, error_responses
from app.schemas.api import ScanDetail
from app.schemas.rules import MAX_RULES_CHARS, InvalidRulesError, check_rule_columns, parse_rules
from app.services import query_service
from app.services.ingestion import FileTooLargeError, ingest_csv
from app.services.scan_service import ScanResult, run_scan

router = APIRouter(prefix="/scans", tags=["scans"])

DEFAULT_DATASET_NAME = "uploaded_dataset"


def _resolve_dataset_name(dataset_name: str | None, filename: str | None) -> str:
    """Use the given name, else the upload's file name without folders or extension.

    Both are untrusted: control characters are removed here, and run_scan trims and
    truncates the result.
    """
    candidate = (dataset_name or "").strip()
    if not candidate:
        base = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        candidate = base.rsplit(".", 1)[0] if "." in base else base
    cleaned = "".join(char for char in candidate if char.isprintable()).strip()
    return cleaned or DEFAULT_DATASET_NAME


def _as_validation_error(error: InvalidRulesError) -> RequestValidationError:
    """Report rejected rules through the API's shared 422 response (location body.rules...)."""
    return RequestValidationError(
        [
            {"loc": ("body", "rules", *issue.location), "msg": issue.message, "type": "value_error"}
            for issue in error.issues
        ]
    )


# A plain `def` handler: FastAPI runs it in a worker thread, so the blocking work (parsing,
# pandas, database commits) does not stall the event loop.
@router.post(
    "",
    response_model=ScanResult,
    status_code=201,
    summary="Upload a CSV file and run a scan",
    description=(
        "Validates the CSV (UTF-8, size, row and column limits), profiles it, runs the "
        "deterministic detectors and stores the results. The file is processed in memory and "
        "is not stored; only column profiles, counts and safe evidence are kept. Findings "
        "start with review status 'pending'. The optional `rules` field is a JSON object that "
        "configures the detectors for this scan only (numeric ranges, date formats, thresholds); "
        "it is validated, size-limited and not stored, and it may only name columns that the "
        "file has."
    ),
    responses=error_responses(413, 422, 500),
)
def create_scan(
    db: DbSession,
    limits: Limits,
    file: Annotated[UploadFile, File(description="The CSV file to scan (UTF-8).")],
    dataset_name: Annotated[
        str | None,
        Form(max_length=255, description="Optional name; defaults to the file name."),
    ] = None,
    rules: Annotated[
        str | None,
        Form(
            max_length=MAX_RULES_CHARS,
            description="Optional JSON object with detection rules for this scan, for example "
            '{"numeric_rules": {"quantity": {"min_value": 0}}}.',
        ),
    ] = None,
) -> ScanResult:
    try:
        config = parse_rules(rules)
    except InvalidRulesError as error:
        raise _as_validation_error(error) from None

    # Bounded read: never load more than one byte past the limit into memory.
    data = file.file.read(limits.max_bytes + 1)
    if len(data) > limits.max_bytes:
        raise FileTooLargeError(f"The file exceeds the limit of {limits.max_bytes} bytes.")

    ingested = ingest_csv(data, limits)
    if config is not None:
        try:
            check_rule_columns(config, ingested.dataframe.columns)
        except InvalidRulesError as error:
            raise _as_validation_error(error) from None
    return run_scan(
        db,
        ingested.dataframe,
        dataset_name=_resolve_dataset_name(dataset_name, file.filename),
        source_filename=file.filename,
        config=config,
    )


@router.get(
    "/{scan_id}",
    response_model=ScanDetail,
    summary="Get a scan's status and safe summary",
    responses=error_responses(404, 422),
)
def get_scan(scan_id: IdPath, db: DbSession) -> ScanDetail:
    return query_service.get_scan(db, scan_id)