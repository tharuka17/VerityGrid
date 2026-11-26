"""Request and response schemas for the HTTP API.

Every response model is built from persisted, safe data (counts, names, rule evidence). None
of these models has a field that could carry an uploaded row or a matched sensitive value.

The only free text a client can supply is a review comment and a reviewer name. Both are
stored and returned exactly as given (after trimming), so users should not type sensitive
values into them.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Generic, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from app.db.models import ReviewStatus, Severity


def _ensure_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes (stored as UTC); label them so output is consistent."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]

T = TypeVar("T")


class ApiModel(BaseModel):
    """Base for response models: immutable, and buildable from ORM objects."""

    model_config = ConfigDict(from_attributes=True, frozen=True)


# --- Errors -------------------------------------------------------------------


class ValidationIssue(ApiModel):
    location: str  # for example "query.limit" or "body.decision"
    message: str


class ErrorDetail(ApiModel):
    code: str  # stable and machine-readable, for example "not_found"
    message: str
    issues: list[ValidationIssue] | None = None  # only for validation errors


class ErrorResponse(ApiModel):
    """The single error shape used by every endpoint."""

    error: ErrorDetail


# --- Pagination ---------------------------------------------------------------


class Page(ApiModel, Generic[T]):
    items: list[T]
    total: int  # matching rows across all pages
    limit: int
    offset: int


# --- Datasets and profiles ----------------------------------------------------


class DatasetOut(ApiModel):
    id: int
    name: str
    source_filename: str | None
    row_count: int
    column_count: int
    created_at: UtcDatetime
    updated_at: UtcDatetime


class DatasetDetail(DatasetOut):
    scan_count: int
    finding_count: int
    pending_review_count: int
    latest_scan_id: int | None
    latest_scan_status: str | None


class ColumnProfileOut(ApiModel):
    id: int
    column_name: str
    inferred_type: str
    null_count: int
    null_percent: float
    distinct_count: int
    classification_label: str | None
    classification_source: str | None
    classification_explanation: str | None


class ColumnProfilesResponse(ApiModel):
    dataset_id: int
    row_count: int
    columns: list[ColumnProfileOut]


# --- Scans ---------------------------------------------------------------------
# POST /scans returns ScanResult from app/services/scan_service.py.


class ScanDetail(ApiModel):
    id: int
    dataset_id: int
    status: str
    started_at: UtcDatetime
    finished_at: UtcDatetime | None
    summary: dict[str, Any] | None  # aggregate counts only
    error_details: str | None  # generic text for failed scans (exception class name only)


# --- Findings and reviews -------------------------------------------------------


class FindingOut(ApiModel):
    id: int
    dataset_id: int
    scan_run_id: int
    category: str
    severity: Severity
    column_name: str | None
    title: str | None
    description: str | None
    safe_evidence: dict[str, Any] | None  # counts, percentages and rule names only
    detection_method: str
    recommendation: str | None
    review_status: ReviewStatus
    created_at: UtcDatetime


class ReviewDecisionOut(ApiModel):
    id: int
    finding_id: int
    decision: ReviewStatus
    comment: str | None
    reviewer: str | None  # self-reported; there is no authentication yet
    created_at: UtcDatetime


class FindingDetail(FindingOut):
    review_decisions: list[ReviewDecisionOut]  # oldest first; the latest drives review_status


class ReviewAction(StrEnum):
    """What a reviewer may submit. 'pending' is the default state, never a decision."""

    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewRequest(BaseModel):
    """Body of POST /findings/{finding_id}/review."""

    model_config = ConfigDict(extra="forbid")

    decision: ReviewAction
    comment: str | None = Field(default=None, max_length=1000)
    reviewer: str | None = Field(default=None, max_length=100)

    @field_validator("comment", "reviewer")
    @classmethod
    def _trim_and_blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class ReviewOut(ApiModel):
    decision: ReviewDecisionOut
    finding: FindingOut


# --- Metrics -------------------------------------------------------------------


class MetricsSummary(ApiModel):
    """Counts calculated from persisted records only."""

    dataset_count: int
    scan_count: int
    scans_by_status: dict[str, int]  # every status always present
    finding_count: int
    findings_by_severity: dict[str, int]  # every severity always present
    findings_by_category: dict[str, int]  # only categories that occur
    findings_by_review_status: dict[str, int]  # every status always present
    pending_high_severity_count: int
    review_decision_count: int
    reviewed_finding_percent: float | None  # None when there are no findings