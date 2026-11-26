import json
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.classifier import ColumnClassifier
from app.ai.mock_provider import MockClassificationProvider
from app.ai.provider import (
    ClassificationProvider,
    ClassificationSource,
    ColumnMetadata,
    ProviderUnavailableError,
)
from app.db.models import Base, ColumnProfile, Dataset, Finding, ScanRun
from app.schemas.findings import DetectionConfig, NumericRule
from app.services import query_service as q
from app.services import scan_service
from app.services.scan_service import ScanFailedError, ScanResult, run_scan
from app.workflows import scan_graph

GOOD: dict[str, Any] = {
    "label": "internal",
    "explanation": "Operational data.",
    "supporting_refs": ["column_name"],
    "review_recommendation": "review_recommended",
}
DOWN = ProviderUnavailableError("offline")
BASE_SUMMARY_KEYS = {
    "row_count",
    "column_count",
    "finding_count",
    "findings_by_severity",
    "findings_by_category",
}
SCAN_RESULT_KEYS = {
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


class CountingProvider(ClassificationProvider):
    """Returns the queued outcomes in order, then `default`. Exceptions are raised."""

    name = "counting-stub"
    source = ClassificationSource.LLM

    def __init__(self, *outcomes: Any, default: Any = GOOD) -> None:
        self.outcomes = list(outcomes)
        self.default = default
        self.seen: list[ColumnMetadata] = []

    def classify(self, metadata: ColumnMetadata) -> Any:
        self.seen.append(metadata)
        outcome = self.outcomes.pop(0) if self.outcomes else self.default
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class BrokenClassifier:
    """A classifier object with a bug: unlike a provider failure, this raises out of the step."""

    def classify_column(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("SECRET-DETAIL")


def contact_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contact_email": ["a@example.com", "b@example.com", "c@example.com"],
            "credit_limit": [100, 200, 300],
            "city": ["x", "y", "z"],
        }
    )


def mock_classifier() -> ColumnClassifier:
    return ColumnClassifier(MockClassificationProvider())


def profiles_in_order(db: Session) -> list[ColumnProfile]:
    return list(db.scalars(select(ColumnProfile).order_by(ColumnProfile.id)).all())


def count_rows(db: Session, model: type) -> int:
    return db.scalar(select(func.count()).select_from(model)) or 0


def only_scan(db: Session) -> ScanRun:
    db.expire_all()
    return db.scalars(select(ScanRun)).one()


def dump_database(db: Session) -> str:
    parts = []
    for table in Base.metadata.sorted_tables:
        for row in db.execute(select(table)).all():
            parts.append(repr(tuple(row)))
    return "\n".join(parts)


# --- Without a classifier (the default) --------------------------------------------------


def test_scan_without_classifier_leaves_classification_empty(session: Session) -> None:
    result = run_scan(session, contact_frame(), dataset_name="d")

    assert result.status == "completed"
    for profile in profiles_in_order(session):
        assert profile.classification_label is None
        assert profile.classification_source is None
        assert profile.classification_explanation is None
    scan = only_scan(session)
    assert scan.summary is not None
    assert set(scan.summary) == BASE_SUMMARY_KEYS  # an ordinary scan's summary is unchanged


# --- Mock classification -------------------------------------------------------------------


def test_mock_classification_is_persisted_on_column_profiles(session: Session) -> None:
    run_scan(session, contact_frame(), dataset_name="d", classifier=mock_classifier())

    profiles = {p.column_name: p for p in profiles_in_order(session)}

    assert {name: p.classification_label for name, p in profiles.items()} == {
        "contact_email": "personal",
        "credit_limit": "financial",
        "city": "unknown",
    }
    for profile in profiles.values():
        assert profile.classification_source == "mock"
        assert profile.classification_explanation is not None
        assert profile.classification_explanation.startswith("[MOCK] ")


def test_mock_scan_summary_records_the_workflow(session: Session) -> None:
    run_scan(session, contact_frame(), dataset_name="d", classifier=mock_classifier())

    scan = only_scan(session)

    assert scan.status == "completed"
    assert scan.summary is not None
    assert set(scan.summary) == BASE_SUMMARY_KEYS | {"workflow"}
    workflow = scan.summary["workflow"]
    assert workflow["status"] == "completed"
    assert workflow["classification"] == {
        "requested": True,
        "attempted": 3,
        "succeeded": 3,
        "failed": 0,
        "skipped_by_cap": 0,
        "skipped_by_breaker": 0,
        "circuit_opened": False,
        "provider": "mock-rules-v1",
    }
    assert workflow["risk"]["dataset"] == "high"
    assert workflow["issues"] == []
    assert len(workflow["steps"]) == 7


def test_scan_result_shape_is_unchanged_by_classification(session: Session) -> None:
    result = run_scan(session, contact_frame(), dataset_name="d", classifier=mock_classifier())

    assert isinstance(result, ScanResult)
    assert set(result.model_dump()) == SCAN_RESULT_KEYS
    assert result.finding_count == 1


def test_llm_source_is_persisted_as_llm(session: Session) -> None:
    run_scan(session, contact_frame(), dataset_name="d", classifier=ColumnClassifier(CountingProvider()))

    for profile in profiles_in_order(session):
        assert profile.classification_label == "internal"
        assert profile.classification_source == "llm"
        assert profile.classification_explanation == "Operational data."


# --- Partial scans -----------------------------------------------------------------------------


def test_provider_failure_gives_a_partial_scan_with_deterministic_results(session: Session) -> None:
    provider = CountingProvider(default=DOWN)

    result = run_scan(session, contact_frame(), dataset_name="d", classifier=ColumnClassifier(provider))

    assert result.status == "completed"
    assert result.finding_count == 1
    assert count_rows(session, Finding) == 1
    scan = only_scan(session)
    assert scan.status == "completed"
    assert scan.error_details is None
    assert scan.summary is not None
    workflow = scan.summary["workflow"]
    assert workflow["status"] == "partially_completed"
    assert workflow["issues"] == [
        {"step": "classify_columns", "code": "classification_failed_columns", "error_type": None}
    ]


def test_failed_classification_is_marked_none_on_the_column_profile(session: Session) -> None:
    run_scan(session, contact_frame(), dataset_name="d", classifier=ColumnClassifier(CountingProvider(default=DOWN)))

    for profile in profiles_in_order(session):
        assert profile.classification_label is None  # a failure never stores a label
        assert profile.classification_source == "none"
        assert profile.classification_explanation is not None
        assert profile.classification_explanation.startswith("No suggestion available:")


def test_circuit_breaker_in_a_persisted_scan(session: Session) -> None:
    provider = CountingProvider(default=DOWN)
    df = pd.DataFrame({name: [1] for name in "abcde"})

    run_scan(session, df, dataset_name="d", classifier=ColumnClassifier(provider))

    assert len(provider.seen) == 3
    profiles = profiles_in_order(session)
    assert [p.classification_source for p in profiles] == ["none", "none", "none", None, None]
    classification = only_scan(session).summary["workflow"]["classification"]  # type: ignore[index]
    assert classification["attempted"] == 3
    assert classification["failed"] == 3
    assert classification["skipped_by_breaker"] == 2
    assert classification["circuit_opened"] is True


def test_column_cap_in_a_persisted_scan(session: Session) -> None:
    provider = CountingProvider()
    df = pd.DataFrame({f"c{i}": [1] for i in range(51)})  # one more than the default cap of 50

    run_scan(session, df, dataset_name="d", classifier=ColumnClassifier(provider))

    assert len(provider.seen) == 50
    profiles = profiles_in_order(session)
    assert [p.classification_source for p in profiles[:50]] == ["llm"] * 50
    assert profiles[50].classification_source is None
    assert profiles[50].classification_label is None
    summary = only_scan(session).summary
    assert summary is not None
    assert summary["workflow"]["status"] == "partially_completed"
    assert summary["workflow"]["classification"]["skipped_by_cap"] == 1


def test_classifier_bug_gives_a_partial_scan(session: Session) -> None:
    result = run_scan(
        session,
        contact_frame(),
        dataset_name="d",
        classifier=BrokenClassifier(),  # type: ignore[arg-type]
    )

    assert result.status == "completed"
    assert result.finding_count == 1
    scan = only_scan(session)
    assert scan.summary is not None
    workflow = scan.summary["workflow"]
    assert workflow["status"] == "partially_completed"
    assert workflow["issues"] == [
        {"step": "classify_columns", "code": "step_failed", "error_type": "RuntimeError"}
    ]
    assert all(p.classification_source is None for p in profiles_in_order(session))
    assert "SECRET" not in dump_database(session)


# --- Failures -------------------------------------------------------------------------------------


def test_workflow_failure_fails_the_scan_without_calling_the_provider(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("SECRET-DETAIL")

    provider = CountingProvider()
    monkeypatch.setattr(scan_graph, "run_detectors", explode)

    with pytest.raises(ScanFailedError) as excinfo:
        run_scan(session, contact_frame(), dataset_name="d", classifier=ColumnClassifier(provider))
    scan = only_scan(session)

    assert "RuntimeError" in str(excinfo.value)
    assert "SECRET" not in str(excinfo.value)
    assert scan.status == "failed"
    assert scan.summary is None
    assert scan.error_details is not None and "RuntimeError" in scan.error_details
    assert "SECRET" not in scan.error_details
    assert count_rows(session, ColumnProfile) == 0
    assert count_rows(session, Finding) == 0
    assert provider.seen == []  # nothing was sent to the provider


def test_failure_while_persisting_rolls_back_classifications(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_summary(*args: Any, **kwargs: Any) -> None:
        raise ValueError("summary failed SECRET-VALUE")

    provider = CountingProvider()
    monkeypatch.setattr(scan_service, "_build_summary", failing_summary)

    with pytest.raises(ScanFailedError):
        run_scan(session, contact_frame(), dataset_name="d", classifier=ColumnClassifier(provider))
    scan = only_scan(session)

    assert len(provider.seen) == 3  # the provider calls already happened (and cost something)
    assert count_rows(session, ColumnProfile) == 0  # but no classification was kept
    assert count_rows(session, Finding) == 0
    assert scan.status == "failed"
    assert scan.error_details is not None and "ValueError" in scan.error_details
    assert "SECRET" not in scan.error_details


@pytest.mark.parametrize(
    ("df", "name", "error"),
    [
        pytest.param("not a frame", "d", TypeError, id="not-a-dataframe"),
        pytest.param(pd.DataFrame([[1, 2]], columns=["a", "a"]), "d", ValueError, id="duplicate-columns"),
    ],
)
def test_invalid_input_never_reaches_the_workflow_or_the_provider(
    session: Session, df: Any, name: str, error: type[Exception]
) -> None:
    provider = CountingProvider()

    with pytest.raises(error):
        run_scan(session, df, dataset_name=name, classifier=ColumnClassifier(provider))

    assert provider.seen == []
    assert count_rows(session, Dataset) == 0
    assert count_rows(session, ScanRun) == 0


# --- Privacy ----------------------------------------------------------------------------------------


def test_database_contains_no_raw_values_with_classification(session: Session) -> None:
    df = pd.DataFrame(
        {
            "contact_email": ["SECRET-CELL-1@example.com", "SECRET-CELL-2@example.com"],
            "note": ["SECRET-NOTE-VALUE", None],
        }
    )
    provider = CountingProvider()

    run_scan(session, df, dataset_name="d", classifier=ColumnClassifier(provider))

    assert "SECRET" not in dump_database(session)
    assert "SECRET" not in " ".join(m.model_dump_json() for m in provider.seen)


def test_summary_contains_no_column_names_or_values(session: Session) -> None:
    run_scan(session, contact_frame(), dataset_name="d", classifier=mock_classifier())

    text = json.dumps(only_scan(session).summary)

    for forbidden in ("contact_email", "credit_limit", "city", "a@example.com"):
        assert forbidden not in text


# --- Reading it back ---------------------------------------------------------------------------------


def test_persisted_classifications_are_visible_through_the_query_service(session: Session) -> None:
    result = run_scan(session, contact_frame(), dataset_name="d", classifier=mock_classifier())

    columns = q.get_column_profiles(session, result.dataset_id).columns

    assert [c.classification_label for c in columns] == ["personal", "financial", "unknown"]
    assert {c.classification_source for c in columns} == {"mock"}
    assert all((c.classification_explanation or "").startswith("[MOCK]") for c in columns)


def test_scan_detail_exposes_the_workflow_summary(session: Session) -> None:
    result = run_scan(session, contact_frame(), dataset_name="d", classifier=mock_classifier())

    detail = q.get_scan(session, result.scan_run_id)

    assert detail.status == "completed"
    assert detail.summary is not None
    assert detail.summary["workflow"]["status"] == "completed"


def test_repeat_scans_keep_their_own_classifications(session: Session) -> None:
    provider = CountingProvider()
    classifier = ColumnClassifier(provider)

    first = run_scan(session, contact_frame(), dataset_name="same", classifier=classifier)
    second = run_scan(session, contact_frame(), dataset_name="same", classifier=classifier)

    assert first.dataset_id != second.dataset_id
    assert count_rows(session, ColumnProfile) == 6
    assert len(provider.seen) == 6
    for dataset_id in (first.dataset_id, second.dataset_id):
        columns = q.get_column_profiles(session, dataset_id).columns
        assert len(columns) == 3
        assert {c.classification_source for c in columns} == {"llm"}


# --- Other behavior still works with a classifier ------------------------------------------------------


def test_detection_config_still_passes_through(session: Session) -> None:
    df = pd.DataFrame({"qty": [5, -2]})
    config = DetectionConfig(numeric_rules={"qty": NumericRule(min_value=0)})

    result = run_scan(session, df, dataset_name="d", config=config, classifier=mock_classifier())

    assert "numeric_rule_violation" in {f.category for f in result.findings}
    assert count_rows(session, Finding) == 1


def test_empty_dataframe_with_a_classifier_completes(session: Session) -> None:
    result = run_scan(session, pd.DataFrame(), dataset_name="empty", classifier=mock_classifier())

    assert result.status == "completed"
    assert count_rows(session, ColumnProfile) == 0
    summary = only_scan(session).summary
    assert summary is not None
    assert summary["workflow"]["status"] == "completed"
    assert summary["workflow"]["classification"]["attempted"] == 0