import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, ColumnProfile, Dataset, Finding, ScanRun
from app.db.session import build_engine, init_db
from app.schemas.findings import DateRule, DetectionConfig, FindingCategory, NumericRule
from app.services import scan_service
from app.workflows import scan_graph
from app.services.scan_service import ScanFailedError, ScanResult, run_scan

ISSUE_CONFIG = DetectionConfig(
    numeric_rules={"qty": NumericRule(min_value=0)},
    date_rules={"counted": DateRule()},
)
EXPECTED_CATEGORIES = {
    FindingCategory.DUPLICATE_ROWS.value,
    FindingCategory.INVALID_DATE.value,
    FindingCategory.MISSING_VALUES.value,
    FindingCategory.NUMERIC_RULE_VIOLATION.value,
    FindingCategory.POTENTIAL_EMAIL.value,
}


@pytest.fixture()
def engine(tmp_path: Path) -> Generator[Engine, None, None]:
    """A fresh SQLite database in a pytest temporary folder, never your local database."""
    test_engine = build_engine(f"sqlite:///{(tmp_path / 'test_scan.db').as_posix()}")
    init_db(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture()
def session(engine: Engine) -> Generator[Session, None, None]:
    with sessionmaker(bind=engine)() as db:
        yield db


def issue_frame() -> pd.DataFrame:
    """Triggers exactly one finding from each of five rules when ISSUE_CONFIG is used."""
    return pd.DataFrame(
        {
            "sku": ["A", "B", "C", "A"],
            "qty": [5, -2, 7, 5],
            "contact": ["a@example.com", "b@example.com", "c@example.com", "a@example.com"],
            "notes": [None, None, None, None],
            "counted": ["2026-09-01", "2026-02-30", "2026-09-03", "2026-09-01"],
        }
    )


def count_rows(db: Session, model: type) -> int:
    return db.scalar(select(func.count()).select_from(model)) or 0


def dump_database(db: Session) -> str:
    """Every row of every table as text, to prove raw values were never stored."""
    parts = []
    for table in Base.metadata.sorted_tables:
        for row in db.execute(select(table)).all():
            parts.append(repr(tuple(row)))
    return "\n".join(parts)


def explode(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("detector blew up on SECRET-VALUE")


# --- Successful scans --------------------------------------------------------


def test_successful_scan_returns_compact_result(session: Session) -> None:
    result = run_scan(
        session,
        issue_frame(),
        dataset_name="inventory_synthetic",
        source_filename="inventory.csv",
        config=ISSUE_CONFIG,
    )

    assert isinstance(result, ScanResult)
    assert result.status == "completed"
    assert (result.row_count, result.column_count, result.finding_count) == (4, 5, 5)
    assert result.findings_by_severity == {"low": 0, "medium": 0, "high": 5}
    assert result.findings_by_category == {category: 1 for category in EXPECTED_CATEGORIES}
    assert {f.category for f in result.findings} == EXPECTED_CATEGORIES
    assert {f.review_status for f in result.findings} == {"pending"}
    assert result.finished_at >= result.started_at


def test_dataset_and_scan_records_are_linked(session: Session) -> None:
    result = run_scan(
        session,
        issue_frame(),
        dataset_name="inventory_synthetic",
        source_filename="inventory.csv",
        config=ISSUE_CONFIG,
    )
    session.expire_all()

    dataset = session.scalars(select(Dataset)).one()
    scan = session.scalars(select(ScanRun)).one()

    assert dataset.id == result.dataset_id
    assert scan.id == result.scan_run_id
    assert dataset.name == "inventory_synthetic"
    assert dataset.source_filename == "inventory.csv"
    assert (dataset.row_count, dataset.column_count) == (4, 5)
    assert scan.dataset_id == dataset.id
    assert scan.status == "completed"
    assert scan.error_details is None
    assert scan.started_at is not None
    assert scan.finished_at is not None
    assert scan.finished_at >= scan.started_at
    assert len(dataset.scan_runs) == 1


def test_column_profiles_are_persisted_and_linked(session: Session) -> None:
    result = run_scan(session, issue_frame(), dataset_name="d", config=ISSUE_CONFIG)
    session.expire_all()

    profiles = {p.column_name: p for p in session.scalars(select(ColumnProfile)).all()}

    assert set(profiles) == {"sku", "qty", "contact", "notes", "counted"}
    assert {p.dataset_id for p in profiles.values()} == {result.dataset_id}
    assert profiles["qty"].inferred_type == "integer"
    assert profiles["qty"].distinct_count == 3
    assert profiles["sku"].inferred_type == "string"
    assert profiles["sku"].distinct_count == 3
    assert profiles["notes"].inferred_type == "empty"
    assert profiles["notes"].null_count == 4
    assert profiles["notes"].distinct_count == 0
    assert profiles["counted"].inferred_type == "datetime"
    assert all(p.classification_label is None for p in profiles.values())
    assert len(session.get(Dataset, result.dataset_id).column_profiles) == 5  # type: ignore[union-attr]


def test_findings_are_persisted_with_scan_and_dataset_links(session: Session) -> None:
    result = run_scan(session, issue_frame(), dataset_name="d", config=ISSUE_CONFIG)
    session.expire_all()

    findings = session.scalars(select(Finding)).all()
    scan = session.get(ScanRun, result.scan_run_id)

    assert len(findings) == 5
    assert {f.scan_run_id for f in findings} == {result.scan_run_id}
    assert {f.dataset_id for f in findings} == {result.dataset_id}
    assert {f.id for f in findings} == {f.id for f in result.findings}
    assert {f.review_status for f in findings} == {"pending"}
    assert scan is not None
    assert len(scan.findings) == 5
    assert len(scan.dataset.findings) == 5


def test_finding_text_and_evidence_are_persisted(session: Session) -> None:
    run_scan(session, issue_frame(), dataset_name="d", config=ISSUE_CONFIG)
    session.expire_all()

    findings = {f.category: f for f in session.scalars(select(Finding)).all()}

    email = findings["potential_email"]
    assert email.column_name == "contact"
    assert email.severity == "high"
    assert email.detection_method == "regex_email_pattern"
    assert email.title == "Potential email addresses in 'contact'"
    assert email.description is not None and "4 of 4" in email.description
    assert email.recommendation
    assert email.safe_evidence == {
        "non_null_count": 4,
        "match_count": 4,
        "match_percent": 100.0,
        "pattern_family": "email",
    }

    duplicates = findings["duplicate_rows"]
    assert duplicates.column_name is None
    assert duplicates.safe_evidence is not None
    assert duplicates.safe_evidence["duplicate_row_count"] == 1


def test_scan_summary_in_database_matches_result(session: Session) -> None:
    result = run_scan(session, issue_frame(), dataset_name="d", config=ISSUE_CONFIG)
    session.expire_all()

    scan = session.get(ScanRun, result.scan_run_id)

    assert scan is not None
    assert scan.summary == {
        "row_count": 4,
        "column_count": 5,
        "finding_count": 5,
        "findings_by_severity": {"low": 0, "medium": 0, "high": 5},
        "findings_by_category": {category: 1 for category in EXPECTED_CATEGORIES},
    }


def test_no_duplicate_findings_within_a_scan(session: Session) -> None:
    run_scan(session, issue_frame(), dataset_name="d", config=ISSUE_CONFIG)

    keys = [(f.scan_run_id, f.category, f.column_name) for f in session.scalars(select(Finding))]

    assert len(keys) == len(set(keys)) == 5


def test_each_scan_creates_its_own_records(session: Session) -> None:
    first = run_scan(session, issue_frame(), dataset_name="same_name", config=ISSUE_CONFIG)
    second = run_scan(session, issue_frame(), dataset_name="same_name", config=ISSUE_CONFIG)

    assert first.dataset_id != second.dataset_id
    assert first.scan_run_id != second.scan_run_id
    assert count_rows(session, Dataset) == 2
    assert count_rows(session, ScanRun) == 2
    assert count_rows(session, ColumnProfile) == 10
    assert count_rows(session, Finding) == 10
    for scan_run_id in (first.scan_run_id, second.scan_run_id):
        keys = [
            (f.category, f.column_name)
            for f in session.scalars(select(Finding).where(Finding.scan_run_id == scan_run_id))
        ]
        assert len(keys) == len(set(keys)) == 5


def test_detection_config_is_applied(session: Session) -> None:
    """Without numeric and date rules, only the rules that need no configuration fire."""
    result = run_scan(session, issue_frame(), dataset_name="d")

    assert result.finding_count == 3
    assert {f.category for f in result.findings} == {
        "duplicate_rows",
        "missing_values",
        "potential_email",
    }


def test_clean_data_produces_no_findings(session: Session) -> None:
    result = run_scan(session, pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}), dataset_name="clean")

    assert result.status == "completed"
    assert result.finding_count == 0
    assert result.findings == []
    assert result.findings_by_severity == {"low": 0, "medium": 0, "high": 0}
    assert result.findings_by_category == {}
    assert count_rows(session, ColumnProfile) == 2
    assert count_rows(session, Finding) == 0


def test_empty_dataframe_completes_with_zero_counts(session: Session) -> None:
    result = run_scan(session, pd.DataFrame(), dataset_name="empty")

    assert result.status == "completed"
    assert (result.row_count, result.column_count, result.finding_count) == (0, 0, 0)
    assert count_rows(session, Dataset) == 1
    assert count_rows(session, ColumnProfile) == 0


# --- Privacy -----------------------------------------------------------------


def test_database_contains_no_raw_values(session: Session) -> None:
    df = pd.DataFrame(
        {
            "email": ["jane.secret@example.com", "bob.secret@example.net", "x@example.org"],
            "phone": ["503-555-0199", "503-555-0198", "n/a"],
            "note": ["SECRET-NOTE", "ok", "ok"],
            "qty": [-8675309, 5, 6],
            "ship_date": ["2026-02-30", "2026-09-01", "2026-09-02"],
        }
    )
    config = DetectionConfig(
        numeric_rules={"qty": NumericRule(min_value=0)},
        date_rules={"ship_date": DateRule()},
    )

    result = run_scan(session, df, dataset_name="sentinel_check", config=config)
    dump = dump_database(session)

    assert {f.category for f in result.findings} >= {
        "potential_email",
        "potential_phone",
        "numeric_rule_violation",
        "invalid_date",
    }
    for raw_value in (
        "jane.secret",
        "bob.secret",
        "503-555-0199",
        "503-555-0198",
        "SECRET-NOTE",
        "8675309",
        "2026-02-30",
    ):
        assert raw_value not in dump
    assert "@" not in dump


def test_result_is_compact_and_json_serializable(session: Session) -> None:
    result = run_scan(session, issue_frame(), dataset_name="d", config=ISSUE_CONFIG)

    raw_json = result.model_dump_json()
    payload = json.loads(raw_json)

    assert set(payload) == {
        "dataset_id",
        "scan_run_id",
        "status",
        "started_at",
        "finished_at",
        "row_count",
        "column_count",
        "finding_count",
        "findings_by_severity",
        "findings_by_category",
        "findings",
    }
    assert set(payload["findings"][0]) == {
        "id",
        "category",
        "severity",
        "column_name",
        "title",
        "review_status",
    }
    assert "a@example.com" not in raw_json
    assert "2026-02-30" not in raw_json
    assert "@" not in raw_json


def test_success_logging_contains_ids_and_counts_only(
    session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="app.services.scan_service")
    df = pd.DataFrame({"email": ["jane.secret@example.com"], "note": ["SECRET-VALUE"]})

    run_scan(session, df, dataset_name="log_check")

    assert "Scan started" in caplog.text
    assert "Scan completed" in caplog.text
    assert "rows=1" in caplog.text
    assert "SECRET" not in caplog.text
    assert "@" not in caplog.text


# --- Failure handling --------------------------------------------------------


def test_detector_failure_records_a_failed_scan(
    session: Session, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="app.services.scan_service")
    monkeypatch.setattr(scan_graph, "run_detectors", explode)

    with pytest.raises(ScanFailedError) as excinfo:
        run_scan(session, issue_frame(), dataset_name="will_fail")
    error = excinfo.value
    session.expire_all()
    scan = session.get(ScanRun, error.scan_run_id)

    assert scan is not None
    assert scan.status == "failed"
    assert scan.dataset_id == error.dataset_id
    assert scan.finished_at is not None
    assert scan.summary is None
    assert scan.error_details is not None and "RuntimeError" in scan.error_details
    assert "SECRET" not in scan.error_details
    assert "SECRET" not in str(error)
    assert "SECRET" not in caplog.text
    assert "Scan failed" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert count_rows(session, Dataset) == 1
    assert count_rows(session, ColumnProfile) == 0
    assert count_rows(session, Finding) == 0


def test_failure_after_writes_rolls_everything_back(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The profiles and findings are already flushed when this fails, so rollback must remove them."""

    def failing_summary(*args: Any, **kwargs: Any) -> None:
        raise ValueError("summary failed SECRET-VALUE")

    monkeypatch.setattr(scan_service, "_build_summary", failing_summary)

    with pytest.raises(ScanFailedError) as excinfo:
        run_scan(session, issue_frame(), dataset_name="rollback_check", config=ISSUE_CONFIG)
    session.expire_all()
    scan = session.get(ScanRun, excinfo.value.scan_run_id)

    assert count_rows(session, ColumnProfile) == 0
    assert count_rows(session, Finding) == 0
    assert scan is not None
    assert scan.status == "failed"
    assert scan.error_details is not None and "ValueError" in scan.error_details
    assert "SECRET" not in scan.error_details


def test_session_remains_usable_after_a_failure(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(scan_graph, "run_detectors", explode)
        with pytest.raises(ScanFailedError):
            run_scan(session, issue_frame(), dataset_name="first")

    result = run_scan(session, issue_frame(), dataset_name="second", config=ISSUE_CONFIG)

    assert result.status == "completed"
    assert sorted(session.scalars(select(ScanRun.status)).all()) == ["completed", "failed"]
    assert count_rows(session, Finding) == 5  # only the successful scan's findings exist


# --- Input validation --------------------------------------------------------


@pytest.mark.parametrize(
    ("df", "name", "error"),
    [
        pytest.param("not a dataframe", "ok", TypeError, id="not-a-dataframe"),
        pytest.param(pd.DataFrame({"a": [1]}), "   ", ValueError, id="blank-name"),
        pytest.param(
            pd.DataFrame([[1, 2]], columns=["a", "a"]), "ok", ValueError, id="duplicate-columns"
        ),
        pytest.param(pd.DataFrame({"x" * 256: [1]}), "ok", ValueError, id="long-column-name"),
    ],
)
def test_invalid_input_is_rejected_before_any_write(
    session: Session, df: Any, name: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        run_scan(session, df, dataset_name=name)

    assert count_rows(session, Dataset) == 0
    assert count_rows(session, ScanRun) == 0


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("C:\\Users\\me\\customers.csv", "customers.csv"),
        ("../../etc/passwd.csv", "passwd.csv"),
        (None, None),
        ("   ", None),
    ],
    ids=["windows-path", "parent-traversal", "none", "blank"],
)
def test_source_filename_is_reduced_to_a_safe_name(
    session: Session, given: str | None, expected: str | None
) -> None:
    run_scan(session, pd.DataFrame({"a": [1]}), dataset_name="d", source_filename=given)

    assert session.scalars(select(Dataset)).one().source_filename == expected


def test_long_dataset_name_is_truncated(session: Session) -> None:
    run_scan(session, pd.DataFrame({"a": [1]}), dataset_name="n" * 300)

    assert len(session.scalars(select(Dataset)).one().name) == 255