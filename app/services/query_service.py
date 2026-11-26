"""Read-only queries behind the API.

Every function reads persisted records and returns a response schema. Nothing here writes to
the database, and nothing can return an uploaded value, because none is stored. Route
handlers call these functions; they contain no query logic of their own.

Raises NotFoundError for a missing record (the API layer maps it to HTTP 404).
"""

from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    ColumnProfile,
    Dataset,
    Finding,
    ReviewDecision,
    ReviewStatus,
    ScanRun,
    ScanStatus,
    Severity,
)
from app.schemas.api import (
    ColumnProfileOut,
    ColumnProfilesResponse,
    DatasetDetail,
    DatasetOut,
    FindingDetail,
    FindingOut,
    MetricsSummary,
    Page,
    ReviewDecisionOut,
    ScanDetail,
)

# Order findings with the most severe first.
_SEVERITY_RANK = case(
    (Finding.severity == Severity.HIGH.value, 0),
    (Finding.severity == Severity.MEDIUM.value, 1),
    else_=2,
)


class NotFoundError(Exception):
    """A requested record does not exist. The message contains only the type and numeric id."""

    def __init__(self, entity: str, entity_id: int) -> None:
        super().__init__(f"{entity.capitalize()} {entity_id} was not found.")
        self.entity = entity
        self.entity_id = entity_id


# --- Helpers -------------------------------------------------------------------


def _count(db: Session, model: type, *conditions: Any) -> int:
    statement = select(func.count()).select_from(model)
    if conditions:
        statement = statement.where(*conditions)
    return db.scalar(statement) or 0


def _grouped_counts(db: Session, column: Any) -> dict[str, int]:
    rows = db.execute(select(column, func.count()).group_by(column)).all()
    return {str(key): int(count) for key, count in rows}


def _require(db: Session, model: type, entity: str, entity_id: int) -> Any:
    record = db.get(model, entity_id)
    if record is None:
        raise NotFoundError(entity, entity_id)
    return record


# --- Scans ---------------------------------------------------------------------


def get_scan(db: Session, scan_id: int) -> ScanDetail:
    return ScanDetail.model_validate(_require(db, ScanRun, "scan", scan_id))


# --- Datasets ------------------------------------------------------------------


def list_datasets(db: Session, *, limit: int, offset: int) -> Page[DatasetOut]:
    """Newest dataset first."""
    rows = db.scalars(
        select(Dataset).order_by(Dataset.id.desc()).limit(limit).offset(offset)
    ).all()
    return Page[DatasetOut](
        items=[DatasetOut.model_validate(row) for row in rows],
        total=_count(db, Dataset),
        limit=limit,
        offset=offset,
    )


def get_dataset(db: Session, dataset_id: int) -> DatasetDetail:
    dataset = _require(db, Dataset, "dataset", dataset_id)
    latest_scan = db.scalars(
        select(ScanRun)
        .where(ScanRun.dataset_id == dataset_id)
        .order_by(ScanRun.id.desc())
        .limit(1)
    ).first()
    return DatasetDetail(
        **DatasetOut.model_validate(dataset).model_dump(),
        scan_count=_count(db, ScanRun, ScanRun.dataset_id == dataset_id),
        finding_count=_count(db, Finding, Finding.dataset_id == dataset_id),
        pending_review_count=_count(
            db,
            Finding,
            Finding.dataset_id == dataset_id,
            Finding.review_status == ReviewStatus.PENDING.value,
        ),
        latest_scan_id=latest_scan.id if latest_scan else None,
        latest_scan_status=latest_scan.status if latest_scan else None,
    )


def get_column_profiles(db: Session, dataset_id: int) -> ColumnProfilesResponse:
    """Persisted profiles in the original column order. Not paginated (at most 200 columns)."""
    dataset = _require(db, Dataset, "dataset", dataset_id)
    profiles = db.scalars(
        select(ColumnProfile).where(ColumnProfile.dataset_id == dataset_id).order_by(ColumnProfile.id)
    ).all()
    row_count = dataset.row_count
    return ColumnProfilesResponse(
        dataset_id=dataset.id,
        row_count=row_count,
        columns=[
            ColumnProfileOut(
                id=profile.id,
                column_name=profile.column_name,
                inferred_type=profile.inferred_type,
                null_count=profile.null_count,
                null_percent=round(100.0 * profile.null_count / row_count, 2) if row_count else 0.0,
                distinct_count=profile.distinct_count,
                classification_label=profile.classification_label,
                classification_source=profile.classification_source,
                classification_explanation=profile.classification_explanation,
            )
            for profile in profiles
        ],
    )


# --- Findings ------------------------------------------------------------------


def list_findings(
    db: Session,
    *,
    limit: int,
    offset: int,
    dataset_id: int | None = None,
    scan_id: int | None = None,
    severity: Severity | None = None,
    review_status: ReviewStatus | None = None,
    category: str | None = None,
    detection_method: str | None = None,
) -> Page[FindingOut]:
    """Most severe first, then oldest first. Filters combine with AND.

    A filter that matches nothing, including a dataset or scan id that does not exist,
    returns an empty page rather than 404.
    """
    conditions: list[Any] = []
    if dataset_id is not None:
        conditions.append(Finding.dataset_id == dataset_id)
    if scan_id is not None:
        conditions.append(Finding.scan_run_id == scan_id)
    if severity is not None:
        conditions.append(Finding.severity == severity.value)
    if review_status is not None:
        conditions.append(Finding.review_status == review_status.value)
    if category is not None:
        conditions.append(Finding.category == category)
    if detection_method is not None:
        conditions.append(Finding.detection_method == detection_method)

    statement = select(Finding)
    if conditions:
        statement = statement.where(*conditions)
    rows = db.scalars(
        statement.order_by(_SEVERITY_RANK, Finding.id).limit(limit).offset(offset)
    ).all()
    return Page[FindingOut](
        items=[FindingOut.model_validate(row) for row in rows],
        total=_count(db, Finding, *conditions),
        limit=limit,
        offset=offset,
    )


def get_finding(db: Session, finding_id: int) -> FindingDetail:
    finding = _require(db, Finding, "finding", finding_id)
    decisions = db.scalars(
        select(ReviewDecision)
        .where(ReviewDecision.finding_id == finding_id)
        .order_by(ReviewDecision.created_at, ReviewDecision.id)
    ).all()
    return FindingDetail(
        **FindingOut.model_validate(finding).model_dump(),
        review_decisions=[ReviewDecisionOut.model_validate(decision) for decision in decisions],
    )


# --- Metrics -------------------------------------------------------------------


def get_metrics_summary(db: Session) -> MetricsSummary:
    """Counts from persisted tables only. Nothing is estimated, sampled or hard-coded."""
    scan_counts = _grouped_counts(db, ScanRun.status)
    severity_counts = _grouped_counts(db, Finding.severity)
    review_counts = _grouped_counts(db, Finding.review_status)

    finding_count = sum(review_counts.values())
    pending_count = review_counts.get(ReviewStatus.PENDING.value, 0)

    return MetricsSummary(
        dataset_count=_count(db, Dataset),
        scan_count=sum(scan_counts.values()),
        scans_by_status={s.value: scan_counts.get(s.value, 0) for s in ScanStatus},
        finding_count=finding_count,
        findings_by_severity={s.value: severity_counts.get(s.value, 0) for s in Severity},
        findings_by_category=dict(sorted(_grouped_counts(db, Finding.category).items())),
        findings_by_review_status={s.value: review_counts.get(s.value, 0) for s in ReviewStatus},
        pending_high_severity_count=_count(
            db,
            Finding,
            Finding.severity == Severity.HIGH.value,
            Finding.review_status == ReviewStatus.PENDING.value,
        ),
        review_decision_count=_count(db, ReviewDecision),
        reviewed_finding_percent=(
            round(100.0 * (finding_count - pending_count) / finding_count, 2)
            if finding_count
            else None
        ),
    )