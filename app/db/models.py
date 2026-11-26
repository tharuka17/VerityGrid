"""SQLAlchemy 2.x ORM models for VerityGrid.

Privacy rule: these tables never store raw uploaded rows or raw sensitive values.
Evidence and metadata columns hold counts, percentages, rule names and column names only.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base class shared by all models."""


# --- Allowed values -------------------------------------------------------
# Stored as plain strings in the database. Validation against these values
# happens in application code (Pydantic schemas from Phase 5 onward).


class ScanStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReviewStatus(StrEnum):
    """Used for Finding.review_status and ReviewDecision.decision (decisions are never 'pending')."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# --- Models ---------------------------------------------------------------


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    source_filename: Mapped[str | None] = mapped_column(String(255))
    row_count: Mapped[int] = mapped_column(default=0)
    column_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    column_profiles: Mapped[list["ColumnProfile"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    scan_runs: Mapped[list["ScanRun"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    findings: Mapped[list["Finding"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class ColumnProfile(Base):
    __tablename__ = "column_profiles"
    __table_args__ = (UniqueConstraint("dataset_id", "column_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    column_name: Mapped[str] = mapped_column(String(255))
    inferred_type: Mapped[str] = mapped_column(String(50))
    null_count: Mapped[int] = mapped_column(default=0)
    distinct_count: Mapped[int] = mapped_column(default=0)
    # Classification metadata (filled by deterministic rules or AI suggestions later).
    classification_label: Mapped[str | None] = mapped_column(String(50))
    classification_source: Mapped[str | None] = mapped_column(String(50))
    classification_explanation: Mapped[str | None] = mapped_column(Text)

    dataset: Mapped["Dataset"] = relationship(back_populates="column_profiles")


class ScanRun(Base):
    __tablename__ = "scan_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(20), default=ScanStatus.PENDING.value)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # JSON: aggregate counts only. Assign a new dict to change it; in-place edits are not tracked.
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_details: Mapped[str | None] = mapped_column(Text)

    dataset: Mapped["Dataset"] = relationship(back_populates="scan_runs")
    findings: Mapped[list["Finding"]] = relationship(
        back_populates="scan_run", cascade="all, delete-orphan"
    )


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    scan_run_id: Mapped[int] = mapped_column(
        ForeignKey("scan_runs.id", ondelete="CASCADE"), index=True
    )
    # Values come from FindingCategory in app/schemas/findings.py (stored as plain strings).
    category: Mapped[str] = mapped_column(String(50))
    severity: Mapped[str] = mapped_column(String(20), default=Severity.MEDIUM.value)
    column_name: Mapped[str | None] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    # JSON: counts, percentages and rule names only. Never raw values.
    safe_evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    detection_method: Mapped[str] = mapped_column(String(100))
    recommendation: Mapped[str | None] = mapped_column(Text)
    review_status: Mapped[str] = mapped_column(String(20), default=ReviewStatus.PENDING.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    dataset: Mapped["Dataset"] = relationship(back_populates="findings")
    scan_run: Mapped["ScanRun"] = relationship(back_populates="findings")
    review_decisions: Mapped[list["ReviewDecision"]] = relationship(
        back_populates="finding", cascade="all, delete-orphan"
    )


class ReviewDecision(Base):
    __tablename__ = "review_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    finding_id: Mapped[int] = mapped_column(
        ForeignKey("findings.id", ondelete="CASCADE"), index=True
    )
    decision: Mapped[str] = mapped_column(String(20))
    comment: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    finding: Mapped["Finding"] = relationship(back_populates="review_decisions")


class AuditEvent(Base):
    """Append-only record of consequential actions.

    entity_type/entity_id are loose references (no foreign key), so audit history
    survives even if the referenced record is removed.
    """

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor: Mapped[str] = mapped_column(String(100))
    event_type: Mapped[str] = mapped_column(String(100))
    entity_type: Mapped[str | None] = mapped_column(String(50))
    entity_id: Mapped[int | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Named safe_metadata because "metadata" is reserved by SQLAlchemy's declarative base.
    safe_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)