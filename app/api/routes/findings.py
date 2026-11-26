"""Finding endpoints: list and filter findings, read one, and record a human review."""

from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import DbSession, IdPath, OptionalIdQuery, Pagination, error_responses
from app.db.models import ReviewStatus, Severity
from app.schemas.api import FindingDetail, FindingOut, Page, ReviewOut, ReviewRequest
from app.schemas.findings import FindingCategory
from app.services import query_service, review_service

router = APIRouter(prefix="/findings", tags=["findings"])


class DetectionMethod(StrEnum):
    """The detection methods the deterministic detectors record in Finding.detection_method.

    Defined here, next to the route that validates it, so an unknown value returns a 422 instead
    of an empty list. It must stay in step with app/services/detectors.py: a test checks that
    every method the detectors produce is listed here.
    """

    NULL_PERCENT_THRESHOLD = "null_percent_threshold"
    PANDAS_DUPLICATED = "pandas_duplicated"
    REGEX_EMAIL_PATTERN = "regex_email_pattern"
    REGEX_PHONE_PATTERN = "regex_phone_pattern"
    CONFIGURED_NUMERIC_BOUNDS = "configured_numeric_bounds"
    STRPTIME_FORMAT_CHECK = "strptime_format_check"


@router.get(
    "",
    response_model=Page[FindingOut],
    summary="List findings (most severe first)",
    description=(
        "Filters combine with AND. A filter that matches nothing returns an empty page. "
        "Evidence contains counts, percentages and rule names only, never matched values."
    ),
    responses=error_responses(422),
)
def list_findings(
    db: DbSession,
    page: Pagination,
    dataset_id: OptionalIdQuery = None,
    scan_id: OptionalIdQuery = None,
    severity: Annotated[Severity | None, Query(description="low, medium or high")] = None,
    review_status: Annotated[
        ReviewStatus | None, Query(description="pending, approved or rejected")
    ] = None,
    category: Annotated[FindingCategory | None, Query(description="Finding category")] = None,
    detection_method: Annotated[
        DetectionMethod | None, Query(description="How the finding was detected")
    ] = None,
) -> Page[FindingOut]:
    return query_service.list_findings(
        db,
        limit=page.limit,
        offset=page.offset,
        dataset_id=dataset_id,
        scan_id=scan_id,
        severity=severity,
        review_status=review_status,
        category=category.value if category is not None else None,
        detection_method=detection_method.value if detection_method is not None else None,
    )


@router.get(
    "/{finding_id}",
    response_model=FindingDetail,
    summary="Get one finding with its review history",
    responses=error_responses(404, 422),
)
def get_finding(finding_id: IdPath, db: DbSession) -> FindingDetail:
    return query_service.get_finding(db, finding_id)


@router.post(
    "/{finding_id}/review",
    response_model=ReviewOut,
    status_code=201,
    summary="Approve or reject a finding's recommendation",
    description=(
        "Records a human decision and an audit event. This changes only the review records: "
        "it never modifies, deletes or remediates data. The reviewer name is self-reported "
        "(there is no authentication yet), and comments are stored as typed, so do not enter "
        "sensitive values in them."
    ),
    responses=error_responses(404, 422),
)
def review_finding(finding_id: IdPath, review: ReviewRequest, db: DbSession) -> ReviewOut:
    return review_service.review_finding(db, finding_id, review)