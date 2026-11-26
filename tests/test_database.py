from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    AuditEvent,
    ColumnProfile,
    Dataset,
    Finding,
    ReviewDecision,
    ReviewStatus,
    ScanRun,
    ScanStatus,
    Severity,
)
from app.db.session import build_engine, get_db, init_db

EXPECTED_TABLES = {
    "datasets",
    "column_profiles",
    "scan_runs",
    "findings",
    "review_decisions",
    "audit_events",
}


@pytest.fixture()
def engine(tmp_path: Path) -> Generator[Engine, None, None]:
    """A fresh SQLite database in a pytest temporary folder, never your local database."""
    test_engine = build_engine(f"sqlite:///{(tmp_path / 'test_veritygrid.db').as_posix()}")
    init_db(test_engine)
    yield test_engine
    test_engine.dispose()  # releases the file so Windows can delete it


@pytest.fixture()
def session(engine: Engine) -> Generator[Session, None, None]:
    with sessionmaker(bind=engine)() as db:
        yield db


def create_sample_graph(db: Session) -> None:
    """Persist one dataset with a column profile, scan run, finding and review decision."""
    dataset = Dataset(
        name="customers_synthetic",
        source_filename="customers.csv",
        row_count=100,
        column_count=2,
    )
    dataset.column_profiles.append(
        ColumnProfile(column_name="email", inferred_type="string", null_count=3, distinct_count=97)
    )
    scan = ScanRun(status=ScanStatus.COMPLETED.value, summary={"findings": 1})
    dataset.scan_runs.append(scan)
    finding = Finding(
        dataset=dataset,
        scan_run=scan,
        category="potential_email",
        severity=Severity.HIGH.value,
        column_name="email",
        detection_method="regex",
        safe_evidence={"match_count": 97, "match_percent": 97.0},
        recommendation="Review whether this column should be classified as confidential.",
    )
    finding.review_decisions.append(
        ReviewDecision(decision=ReviewStatus.APPROVED.value, comment="Looks right", reviewer="tester")
    )
    db.add(dataset)
    db.commit()


def count_rows(db: Session, model: type) -> int:
    return db.scalar(select(func.count()).select_from(model)) or 0


def test_init_db_creates_all_tables(engine: Engine) -> None:
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES


def test_init_db_is_idempotent(engine: Engine) -> None:
    init_db(engine)  # second call must not fail or change anything
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES


def test_dataset_persists_with_defaults_and_timestamps(session: Session) -> None:
    dataset = Dataset(name="orders_synthetic")
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    assert dataset.id is not None
    assert dataset.row_count == 0
    assert dataset.column_count == 0
    assert dataset.created_at is not None
    first_updated_at = dataset.updated_at

    dataset.name = "orders_synthetic_v2"
    session.commit()
    session.refresh(dataset)

    assert dataset.updated_at >= first_updated_at


def test_relationships_navigate_in_both_directions(session: Session) -> None:
    create_sample_graph(session)
    session.expire_all()

    finding = session.scalars(select(Finding)).one()

    assert finding.dataset.name == "customers_synthetic"
    assert finding.scan_run.dataset is finding.dataset
    assert finding.dataset.column_profiles[0].column_name == "email"
    assert finding.review_decisions[0].finding is finding
    assert finding.review_decisions[0].decision == ReviewStatus.APPROVED.value
    assert finding.review_status == ReviewStatus.PENDING.value  # default; set by review logic later


def test_json_columns_round_trip(session: Session) -> None:
    create_sample_graph(session)
    session.expire_all()

    assert session.scalars(select(ScanRun)).one().summary == {"findings": 1}
    assert session.scalars(select(Finding)).one().safe_evidence == {
        "match_count": 97,
        "match_percent": 97.0,
    }


def test_audit_event_is_loosely_coupled(session: Session) -> None:
    """Audit events reference entities by type and id only, so no matching row is required."""
    session.add(
        AuditEvent(
            actor="system",
            event_type="review.approved",
            entity_type="finding",
            entity_id=999,
            safe_metadata={"category": "potential_email"},
        )
    )
    session.commit()
    session.expire_all()

    event = session.scalars(select(AuditEvent)).one()
    assert event.entity_id == 999
    assert event.safe_metadata == {"category": "potential_email"}
    assert event.created_at is not None


def test_foreign_keys_are_enforced(session: Session) -> None:
    session.add(ColumnProfile(dataset_id=9999, column_name="orphan", inferred_type="string"))

    with pytest.raises(IntegrityError):
        session.commit()


def test_deleting_a_dataset_removes_its_dependent_rows(session: Session) -> None:
    create_sample_graph(session)
    dataset = session.scalars(select(Dataset)).one()

    session.delete(dataset)
    session.commit()

    for model in (Dataset, ColumnProfile, ScanRun, Finding, ReviewDecision):
        assert count_rows(session, model) == 0


def test_get_db_yields_a_session_and_closes_it() -> None:
    """Uses the default engine, but never opens a connection, so no database file is touched."""
    generator = get_db()
    db = next(generator)

    assert isinstance(db, Session)
    generator.close()  # runs the dependency's cleanup (db.close())