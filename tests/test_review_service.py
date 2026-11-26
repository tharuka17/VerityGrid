from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    AuditEvent,
    ColumnProfile,
    Dataset,
    Finding,
    ReviewDecision,
    ReviewStatus,
    ScanRun,
)
from app.db.session import build_engine, init_db
from app.schemas.api import ReviewAction, ReviewRequest
from app.services import query_service as q
from app.services import review_service
from app.services.query_service import NotFoundError
from app.services.review_service import review_finding
from app.services.scan_service import ScanResult, run_scan


@pytest.fixture()
def engine(tmp_path: Path) -> Generator[Engine, None, None]:
    """A fresh SQLite database in a pytest temporary folder, never your local database."""
    test_engine = build_engine(f"sqlite:///{(tmp_path / 'test_review.db').as_posix()}")
    init_db(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture()
def session(engine: Engine) -> Generator[Session, None, None]:
    with sessionmaker(bind=engine)() as db:
        yield db


@pytest.fixture()
def scan(session: Session) -> ScanResult:
    """One scan with two HIGH findings: missing_values (column n) and potential_email (email)."""
    df = pd.DataFrame({"email": ["a@example.com", "b@example.com"], "n": [1, None]})
    return run_scan(session, df, dataset_name="review_fixture")


def finding_id(result: ScanResult, category: str) -> int:
    return next(f.id for f in result.findings if f.category == category)


def make_request(decision: ReviewAction, **kwargs: Any) -> ReviewRequest:
    return ReviewRequest(decision=decision, **kwargs)


def count_rows(db: Session, model: type) -> int:
    return db.scalar(select(func.count()).select_from(model)) or 0


# --- Decisions ---------------------------------------------------------------


def test_approve_updates_finding_and_returns_decision(session: Session, scan: ScanResult) -> None:
    fid = finding_id(scan, "potential_email")

    out = review_finding(
        session,
        fid,
        make_request(ReviewAction.APPROVED, comment="Looks right", reviewer="tester"),
    )
    session.expire_all()

    assert out.decision.decision == ReviewStatus.APPROVED
    assert out.decision.finding_id == fid
    assert out.decision.comment == "Looks right"
    assert out.decision.reviewer == "tester"
    assert out.finding.id == fid
    assert out.finding.review_status == ReviewStatus.APPROVED
    assert session.get(Finding, fid).review_status == "approved"  # type: ignore[union-attr]
    stored = session.scalars(select(ReviewDecision)).one()
    assert stored.finding_id == fid
    assert stored.decision == "approved"
    assert stored.created_at is not None
    assert count_rows(session, AuditEvent) == 1


def test_reject_updates_finding(session: Session, scan: ScanResult) -> None:
    fid = finding_id(scan, "missing_values")

    out = review_finding(session, fid, make_request(ReviewAction.REJECTED))
    session.expire_all()

    assert out.decision.decision == ReviewStatus.REJECTED
    assert out.decision.comment is None
    assert out.decision.reviewer is None
    assert session.get(Finding, fid).review_status == "rejected"  # type: ignore[union-attr]


# --- Audit -------------------------------------------------------------------


def test_review_records_audit_event(session: Session, scan: ScanResult) -> None:
    fid = finding_id(scan, "potential_email")

    review_finding(
        session,
        fid,
        make_request(ReviewAction.APPROVED, comment="ok", reviewer="alice"),
    )
    event = session.scalars(select(AuditEvent)).one()

    assert event.actor == "alice"
    assert event.event_type == "review.approved"
    assert event.entity_type == "finding"
    assert event.entity_id == fid
    assert event.created_at is not None
    assert event.safe_metadata == {
        "dataset_id": scan.dataset_id,
        "scan_run_id": scan.scan_run_id,
        "category": "potential_email",
        "severity": "high",
        "previous_status": "pending",
        "new_status": "approved",
        "has_comment": True,
        "actor_self_reported": True,
    }


def test_audit_event_never_contains_comment_text(session: Session, scan: ScanResult) -> None:
    review_finding(
        session,
        finding_id(scan, "potential_email"),
        make_request(ReviewAction.REJECTED, comment="SECRET-COMMENT-TEXT"),
    )

    audit_dump = repr([tuple(row) for row in session.execute(select(AuditEvent.__table__)).all()])

    assert "SECRET-COMMENT-TEXT" not in audit_dump
    assert "has_comment" in audit_dump


def test_missing_reviewer_uses_anonymous_actor(session: Session, scan: ScanResult) -> None:
    review_finding(session, finding_id(scan, "potential_email"), make_request(ReviewAction.APPROVED))

    event = session.scalars(select(AuditEvent)).one()

    assert event.actor == "anonymous"
    assert event.safe_metadata is not None
    assert event.safe_metadata["has_comment"] is False


# --- History -----------------------------------------------------------------


def test_re_review_keeps_history_and_updates_status(session: Session, scan: ScanResult) -> None:
    fid = finding_id(scan, "potential_email")

    review_finding(session, fid, make_request(ReviewAction.APPROVED, reviewer="alice"))
    review_finding(session, fid, make_request(ReviewAction.REJECTED, reviewer="bob"))
    session.expire_all()
    detail = q.get_finding(session, fid)
    events = session.scalars(select(AuditEvent).order_by(AuditEvent.id)).all()

    assert detail.review_status == ReviewStatus.REJECTED
    assert [d.decision for d in detail.review_decisions] == [
        ReviewStatus.APPROVED,
        ReviewStatus.REJECTED,
    ]
    assert [d.reviewer for d in detail.review_decisions] == ["alice", "bob"]
    assert [e.event_type for e in events] == ["review.approved", "review.rejected"]
    assert events[1].safe_metadata is not None
    assert events[1].safe_metadata["previous_status"] == "approved"
    assert events[1].safe_metadata["new_status"] == "rejected"


def test_same_decision_twice_is_recorded_twice(session: Session, scan: ScanResult) -> None:
    fid = finding_id(scan, "potential_email")

    review_finding(session, fid, make_request(ReviewAction.APPROVED))
    review_finding(session, fid, make_request(ReviewAction.APPROVED))

    assert count_rows(session, ReviewDecision) == 2
    assert count_rows(session, AuditEvent) == 2
    assert session.get(Finding, fid).review_status == "approved"  # type: ignore[union-attr]


# --- Failure handling --------------------------------------------------------


def test_unknown_finding_raises_not_found_and_writes_nothing(session: Session, scan: ScanResult) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        review_finding(session, 999_999, make_request(ReviewAction.APPROVED))

    assert str(excinfo.value) == "Finding 999999 was not found."
    assert count_rows(session, ReviewDecision) == 0
    assert count_rows(session, AuditEvent) == 0


def test_failure_rolls_back_decision_audit_and_status(
    session: Session, scan: ScanResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rows are already flushed when this fails, so the rollback must remove them."""
    fid = finding_id(scan, "potential_email")

    class ExplodingSchema:
        @staticmethod
        def model_validate(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    with monkeypatch.context() as patch:
        patch.setattr(review_service, "ReviewDecisionOut", ExplodingSchema)
        with pytest.raises(RuntimeError):
            review_finding(session, fid, make_request(ReviewAction.APPROVED, comment="x"))
    session.expire_all()

    assert count_rows(session, ReviewDecision) == 0
    assert count_rows(session, AuditEvent) == 0
    assert session.get(Finding, fid).review_status == "pending"  # type: ignore[union-attr]

    # The session is still usable afterwards.
    out = review_finding(session, fid, make_request(ReviewAction.APPROVED))
    assert out.finding.review_status == ReviewStatus.APPROVED


def test_review_changes_nothing_but_review_records(session: Session, scan: ScanResult) -> None:
    reviewed_id = finding_id(scan, "potential_email")
    other_id = finding_id(scan, "missing_values")
    evidence_before = dict(session.get(Finding, reviewed_id).safe_evidence or {})  # type: ignore[union-attr]
    counts_before = {
        model: count_rows(session, model) for model in (Dataset, ScanRun, ColumnProfile, Finding)
    }

    review_finding(session, reviewed_id, make_request(ReviewAction.REJECTED, comment="not an issue"))
    session.expire_all()

    assert {
        model: count_rows(session, model) for model in (Dataset, ScanRun, ColumnProfile, Finding)
    } == counts_before
    reviewed = session.get(Finding, reviewed_id)
    assert reviewed is not None
    assert reviewed.safe_evidence == evidence_before
    assert reviewed.category == "potential_email"
    assert session.get(Finding, other_id).review_status == "pending"  # type: ignore[union-attr]


# --- Request schema ----------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": "pending"},
        {"decision": "deleted"},
        {"decision": "approved", "comment": "x" * 1001},
        {"decision": "approved", "remediate": True},
    ],
    ids=["pending-is-not-a-decision", "unknown-decision", "comment-too-long", "extra-field"],
)
def test_review_request_rejects_invalid_input(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ReviewRequest(**payload)


def test_review_request_trims_text() -> None:
    request = ReviewRequest(decision=ReviewAction.APPROVED, comment="  checked  ", reviewer="   ")

    assert request.comment == "checked"
    assert request.reviewer is None
    assert ReviewRequest(decision=ReviewAction.APPROVED, comment="   ").comment is None


# --- Integration with the query service --------------------------------------


def test_metrics_reflect_reviews(session: Session, scan: ScanResult) -> None:
    review_finding(session, finding_id(scan, "potential_email"), make_request(ReviewAction.APPROVED))
    halfway = q.get_metrics_summary(session)

    assert halfway.finding_count == 2
    assert halfway.findings_by_review_status == {"pending": 1, "approved": 1, "rejected": 0}
    assert halfway.review_decision_count == 1
    assert halfway.reviewed_finding_percent == 50.0
    assert halfway.pending_high_severity_count == 1

    review_finding(session, finding_id(scan, "missing_values"), make_request(ReviewAction.REJECTED))
    done = q.get_metrics_summary(session)

    assert done.findings_by_review_status == {"pending": 0, "approved": 1, "rejected": 1}
    assert done.reviewed_finding_percent == 100.0
    assert done.pending_high_severity_count == 0


def test_review_status_filter_after_review(session: Session, scan: ScanResult) -> None:
    review_finding(session, finding_id(scan, "potential_email"), make_request(ReviewAction.APPROVED))

    approved = q.list_findings(session, limit=10, offset=0, review_status=ReviewStatus.APPROVED)
    pending = q.list_findings(session, limit=10, offset=0, review_status=ReviewStatus.PENDING)

    assert approved.total == 1
    assert approved.items[0].category == "potential_email"
    assert pending.total == 1
    assert pending.items[0].category == "missing_values"