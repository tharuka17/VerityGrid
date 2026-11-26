"""Scan service: validated in-memory data -> Dataset, ScanRun, ColumnProfile and Finding records.

Transaction boundaries. This service owns its transactions and commits itself. Pass a session
with no pending changes: the first commit would otherwise include them.

  Transaction 1  Create the Dataset (counts only) and a ScanRun with status "running"; commit.
                 If this fails: roll back and re-raise. Nothing is recorded.
  Compute        Run the bounded scan workflow (app/workflows/scan_graph.py): profiling,
                 deterministic checks and, only if a classifier was passed, optional AI
                 classification. No database access, so no transaction is held open during
                 slow provider calls.
  Transaction 2  Add ColumnProfile and Finding rows, flush, mark the ScanRun "completed" with
                 its summary, then commit ONCE. A scan is only ever "completed" together with
                 all of its profiles and findings.
  Failure path   A FAILED workflow, or any exception in Transaction 2: roll back (discarding
                 every partial row), then Transaction 3 marks the ScanRun "failed" with a
                 generic message (a reason only: an exception class name or a stable issue
                 code), and ScanFailedError is raised. The Dataset and the failed ScanRun
                 remain as a record. If Transaction 3 also fails, the ScanRun stays "running"
                 (a stale marker for a future cleanup job) and the failure is logged.

Partial workflows. A PARTIALLY_COMPLETED workflow (for example the AI provider failed) still
persists its valid deterministic results. The ScanRun status stays "completed" because the
status set has no partial value; ScanRun.summary["workflow"] holds the workflow status, counts
and issue codes. That block is added only when classification was requested or the workflow did
not complete cleanly, so the summary of an ordinary scan is unchanged.

Classification on ColumnProfile (existing columns, no schema change):
  not attempted        label, source and explanation are all NULL
  attempted, failed    label NULL, source "none", explanation = the generic fallback text
  succeeded            label, source ("mock" or "llm") and explanation as validated

Privacy: only counts, names and safe evidence are persisted, never uploaded rows. Log lines and
stored error details contain identifiers, counts and reasons only, never values. ScanFailedError
is chained to an internal exception that carries the same reason only; an API layer must return
only the ScanFailedError message to clients.
"""

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.ai.classifier import ClassificationResult, ColumnClassifier, ResultStatus
from app.ai.provider import ClassificationSource
from app.db.models import (
    ColumnProfile,
    Dataset,
    Finding,
    ScanRun,
    ScanStatus,
    Severity,
    utcnow,
)
from app.schemas.findings import DetectedFinding, DetectionConfig
from app.services.profiling import DatasetProfileResult
from app.workflows.scan_graph import (
    WorkflowOptions,
    WorkflowResult,
    WorkflowStatus,
    run_scan_workflow,
)

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 255  # matches the String(255) columns for dataset and column names


# --- Result and error types --------------------------------------------------


class ScanFindingSummary(BaseModel):
    """One finding in compact form (no evidence, no description)."""

    model_config = ConfigDict(frozen=True)

    id: int
    category: str
    severity: str
    column_name: str | None
    title: str
    review_status: str


class ScanResult(BaseModel):
    """Compact result of a completed scan, suitable for an API response."""

    model_config = ConfigDict(frozen=True)

    dataset_id: int
    scan_run_id: int
    status: str
    started_at: datetime
    finished_at: datetime
    row_count: int
    column_count: int
    finding_count: int
    findings_by_severity: dict[str, int]
    findings_by_category: dict[str, int]
    findings: list[ScanFindingSummary]


class ScanFailedError(Exception):
    """A scan failed after its records were created. The failure was recorded where possible."""

    def __init__(self, message: str, *, dataset_id: int, scan_run_id: int) -> None:
        super().__init__(message)
        self.dataset_id = dataset_id
        self.scan_run_id = scan_run_id


class _WorkflowFailed(Exception):
    """Internal: the workflow reported FAILED. Carries a stable reason only (an exception class
    name or an issue code), never messages or data."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --- Helpers -----------------------------------------------------------------


def _validate_input(df: object, dataset_name: object) -> str:
    """Reject unusable input before anything is written. Returns the cleaned dataset name."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("run_scan expects a pandas DataFrame.")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")

    names = [str(column) for column in df.columns]
    if len(set(names)) != len(names):
        raise ValueError("Column names must be unique to be stored as column profiles.")
    if any(len(name) > MAX_NAME_LENGTH for name in names):
        raise ValueError(f"Column names must not exceed {MAX_NAME_LENGTH} characters.")
    return dataset_name.strip()[:MAX_NAME_LENGTH]


def _safe_filename(filename: str | None) -> str | None:
    """Keep only the final path component, without control characters, or None if empty."""
    if filename is None:
        return None
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(char for char in base if char.isprintable()).strip()
    return cleaned[:MAX_NAME_LENGTH] or None


def _as_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes (stored as UTC); label them so results are consistent."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _failure_reason(workflow: WorkflowResult) -> str:
    """A stable reason for a failed workflow: the first issue's exception class name or code."""
    if workflow.issues:
        issue = workflow.issues[0]
        return issue.error_type or issue.code
    return "WorkflowFailed"


def _classification_fields(result: ClassificationResult | None) -> dict[str, str | None]:
    """Map one classification result onto the existing ColumnProfile columns."""
    if result is None:  # classification was not attempted for this column
        return {
            "classification_label": None,
            "classification_source": None,
            "classification_explanation": None,
        }
    if result.status is ResultStatus.OK:
        return {
            "classification_label": result.label.value,
            "classification_source": result.source.value,
            "classification_explanation": result.explanation,
        }
    return {  # attempted but failed: keep the honest "no suggestion" marker, never a label
        "classification_label": None,
        "classification_source": ClassificationSource.NONE.value,
        "classification_explanation": result.explanation,
    }


def _profile_rows(
    dataset_id: int,
    profile: DatasetProfileResult,
    classifications: Mapping[str, ClassificationResult],
) -> list[ColumnProfile]:
    rows = []
    for position, column in enumerate(profile.columns, start=1):
        rows.append(
            ColumnProfile(
                dataset_id=dataset_id,
                column_name=column.column_name,
                inferred_type=column.inferred_type.value,
                null_count=column.null_count,
                # The ORM column is not nullable. None (unhashable values) cannot occur for CSV
                # input; for other callers it is stored as 0.
                distinct_count=column.distinct_count if column.distinct_count is not None else 0,
                **_classification_fields(classifications.get(f"col_{position}")),
            )
        )
    return rows


def _finding_rows(
    dataset_id: int, scan_run_id: int, detected: list[DetectedFinding]
) -> list[Finding]:
    return [
        Finding(
            dataset_id=dataset_id,
            scan_run_id=scan_run_id,
            category=item.category.value,
            severity=item.severity.value,
            column_name=item.column_name,
            title=item.title,
            description=item.description,
            safe_evidence=dict(item.safe_evidence),
            detection_method=item.detection_method,
            recommendation=item.recommendation,
            review_status=item.review_status.value,
        )
        for item in detected
    ]


def _build_summary(profile: DatasetProfileResult, detected: list[DetectedFinding]) -> dict[str, Any]:
    """Aggregate counts only. Severity keys are always all present, for a predictable shape."""
    by_severity = {severity.value: 0 for severity in Severity}
    by_category: dict[str, int] = {}
    for item in detected:
        by_severity[item.severity.value] += 1
        by_category[item.category.value] = by_category.get(item.category.value, 0) + 1
    return {
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "finding_count": len(detected),
        "findings_by_severity": by_severity,
        "findings_by_category": by_category,
    }


def _record_failure(db: Session, scan_run_id: int, reason: str) -> None:
    """Transaction 3: mark the scan failed. Never raises; the original failure is the one to report."""
    try:
        scan_run = db.get(ScanRun, scan_run_id)
        if scan_run is None:
            return
        scan_run.status = ScanStatus.FAILED.value
        scan_run.finished_at = utcnow()
        scan_run.summary = None
        scan_run.error_details = (
            f"Scan failed ({reason}). "
            "No column profiles or findings were saved for this scan."
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.error("Could not record scan failure: scan_run_id=%d", scan_run_id)


# --- Public API --------------------------------------------------------------


def run_scan(
    db: Session,
    df: pd.DataFrame,
    *,
    dataset_name: str,
    source_filename: str | None = None,
    config: DetectionConfig | None = None,
    classifier: ColumnClassifier | None = None,
) -> ScanResult:
    """Scan a validated DataFrame and persist the results.

    dataset_name and source_filename are untrusted text: the name is trimmed and truncated, and
    the filename is reduced to its final path component. Every call creates a new Dataset and
    ScanRun; re-scanning the same data produces a separate, independent scan.

    `classifier` is optional and off by default. When given, each column (up to the workflow's
    cap) is classified through it, which may call an external provider with sanitized metadata.

    Raises TypeError or ValueError for unusable input (nothing is written), ScanFailedError if
    the scan fails after its records were created, and database errors if the first
    transaction fails.
    """
    name = _validate_input(df, dataset_name)

    # --- Transaction 1: create the dataset and a running scan ---
    try:
        dataset = Dataset(
            name=name,
            source_filename=_safe_filename(source_filename),
            row_count=len(df),
            column_count=df.shape[1],
        )
        db.add(dataset)
        db.flush()
        scan_run = ScanRun(dataset_id=dataset.id, status=ScanStatus.RUNNING.value)
        db.add(scan_run)
        db.flush()
        dataset_id, scan_run_id = dataset.id, scan_run.id
        db.commit()
    except Exception:
        db.rollback()
        raise

    logger.info(
        "Scan started: dataset_id=%d scan_run_id=%d rows=%d columns=%d",
        dataset_id,
        scan_run_id,
        len(df),
        df.shape[1],
    )

    try:
        # --- Compute step: the bounded workflow, in memory only ---
        workflow = run_scan_workflow(
            df,
            dataset_name=name,
            options=WorkflowOptions(classify=classifier is not None, detection_config=config),
            classifier=classifier,
        )
        if workflow.status is WorkflowStatus.FAILED or workflow.profile is None:
            raise _WorkflowFailed(_failure_reason(workflow))
        profile = workflow.profile
        detected = list(workflow.findings)
        classifications = {result.column_ref: result for result in workflow.classifications}

        # --- Transaction 2: persist everything, then finalize the scan, then commit once ---
        db.add_all(_profile_rows(dataset_id, profile, classifications))
        finding_rows = _finding_rows(dataset_id, scan_run_id, detected)
        db.add_all(finding_rows)
        db.flush()  # assigns ids and surfaces constraint errors before the scan is finalized

        summary = _build_summary(profile, detected)
        if classifier is not None or workflow.status is not WorkflowStatus.COMPLETED:
            summary["workflow"] = workflow.to_summary()
        scan_run = db.get(ScanRun, scan_run_id)
        if scan_run is None:
            raise RuntimeError("The scan run record disappeared during the scan.")
        started_at = _as_utc(scan_run.started_at)
        finished_at = utcnow()
        scan_run.status = ScanStatus.COMPLETED.value
        scan_run.finished_at = finished_at
        scan_run.summary = summary  # assigned as a new dict so the change is tracked

        finding_summaries = [
            ScanFindingSummary(
                id=row.id,
                category=item.category.value,
                severity=item.severity.value,
                column_name=item.column_name,
                title=item.title,
                review_status=item.review_status.value,
            )
            for row, item in zip(finding_rows, detected, strict=True)
        ]
        db.commit()
    except Exception as error:
        db.rollback()  # discards every partial profile and finding
        reason = error.reason if isinstance(error, _WorkflowFailed) else type(error).__name__
        _record_failure(db, scan_run_id, reason)
        logger.error(
            "Scan failed: dataset_id=%d scan_run_id=%d error_type=%s",
            dataset_id,
            scan_run_id,
            reason,
        )
        raise ScanFailedError(
            f"The scan failed ({reason}). The failure was recorded for scan "
            f"run {scan_run_id}; no partial results were saved.",
            dataset_id=dataset_id,
            scan_run_id=scan_run_id,
        ) from error

    logger.info(
        "Scan completed: dataset_id=%d scan_run_id=%d findings=%d workflow=%s",
        dataset_id,
        scan_run_id,
        len(detected),
        workflow.status.value,
    )
    return ScanResult(
        dataset_id=dataset_id,
        scan_run_id=scan_run_id,
        status=ScanStatus.COMPLETED.value,
        started_at=started_at,
        finished_at=_as_utc(finished_at),
        row_count=summary["row_count"],
        column_count=summary["column_count"],
        finding_count=summary["finding_count"],
        findings_by_severity=summary["findings_by_severity"],
        findings_by_category=summary["findings_by_category"],
        findings=finding_summaries,
    )