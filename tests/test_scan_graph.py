import dataclasses
import json
import logging
from typing import Any

import pandas as pd
import pytest

from app.ai.classifier import ColumnClassifier, ResultStatus
from app.ai.mock_provider import MockClassificationProvider
from app.ai.provider import (
    ClassificationLabel,
    ClassificationProvider,
    ClassificationSource,
    ColumnMetadata,
    ProviderUnavailableError,
    ReviewRecommendation,
)
from app.schemas.findings import DetectionConfig, NumericRule
from app.services.detectors import run_detectors
from app.services.profiling import profile_dataframe
from app.workflows import scan_graph
from app.workflows.scan_graph import (
    ClassificationStats,
    RiskLevel,
    WorkflowIssue,
    WorkflowOptions,
    WorkflowResult,
    WorkflowStatus,
    build_scan_graph,
    run_scan_workflow,
)

GOOD: dict[str, Any] = {
    "label": "internal",
    "explanation": "Operational data.",
    "supporting_refs": ["column_name"],
    "review_recommendation": "review_recommended",
}
DOWN = ProviderUnavailableError("offline")

STEPS_PLAIN = ["validate_input", "profile_dataset", "run_checks", "consolidate", "apply_policy", "finalize"]
STEPS_CLASSIFIED = [
    "validate_input",
    "profile_dataset",
    "run_checks",
    "classify_columns",
    "consolidate",
    "apply_policy",
    "finalize",
]


class CountingProvider(ClassificationProvider):
    """Returns the queued outcomes in order, then `default`. An outcome that is an exception is
    raised. Records every ColumnMetadata it receives."""

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
    """One HIGH email finding on contact_email under the default configuration."""
    return pd.DataFrame(
        {
            "contact_email": ["a@example.com", "b@example.com", "c@example.com"],
            "credit_limit": [100, 200, 300],
            "city": ["x", "y", "z"],
        }
    )


def frame(*names: str) -> pd.DataFrame:
    """Columns of one integer each: no findings under the default configuration."""
    return pd.DataFrame({name: [1] for name in names})


def run(
    df: pd.DataFrame, provider: ClassificationProvider | None = None, **options: Any
) -> WorkflowResult:
    classifier = ColumnClassifier(provider) if provider is not None else None
    return run_scan_workflow(
        df,
        dataset_name="test",
        options=WorkflowOptions(classify=provider is not None, **options),
        classifier=classifier,
    )


def by_name(result: WorkflowResult) -> dict[str, Any]:
    return {assessment.column_name: assessment for assessment in result.assessments}


def explode(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("SECRET-DETAIL")


# --- Normal completion (deterministic only) ---------------------------------------------


def test_deterministic_run_completes_without_classification() -> None:
    result = run(contact_frame())

    assert result.status is WorkflowStatus.COMPLETED
    assert list(result.steps) == STEPS_PLAIN
    assert result.issues == ()
    assert result.classifications == ()
    assert result.stats == ClassificationStats(requested=False)
    assert result.profile is not None and result.profile.column_count == 3
    assert len(result.assessments) == 3


def test_deterministic_results_come_from_the_existing_services() -> None:
    df = contact_frame()

    result = run(df)

    assert result.findings == tuple(run_detectors(df))
    assert result.profile == profile_dataframe(df)
    assert [f.category.value for f in result.findings] == ["potential_email"]


def test_detection_config_is_passed_through() -> None:
    df = pd.DataFrame({"qty": [5, -2]})
    config = DetectionConfig(numeric_rules={"qty": NumericRule(min_value=0)})

    result = run(df, detection_config=config)

    assert result.findings == tuple(run_detectors(df, config))
    assert "numeric_rule_violation" in {f.category.value for f in result.findings}


def test_assessments_without_classification() -> None:
    assessments = by_name(run(contact_frame()))

    email = assessments["contact_email"]
    assert (email.risk, email.needs_review) == (RiskLevel.HIGH, True)
    assert email.finding_categories == ("potential_email",)
    for name in ("credit_limit", "city"):
        assert (assessments[name].risk, assessments[name].needs_review) == (RiskLevel.LOW, False)
        assert assessments[name].finding_categories == ()
    for assessment in assessments.values():
        assert assessment.label is None
        assert assessment.review_recommendation is None
        assert assessment.classification_status is None


def test_dataset_level_findings_raise_dataset_risk() -> None:
    df = pd.DataFrame({"a": list(range(19)) + [0]})  # one repeated row in twenty

    result = run(df)

    assert [(f.category.value, f.column_name) for f in result.findings] == [("duplicate_rows", None)]
    assert result.dataset_risk is RiskLevel.MEDIUM
    assert (result.assessments[0].risk, result.assessments[0].needs_review) == (RiskLevel.LOW, False)


def test_empty_dataframe_completes() -> None:
    result = run(pd.DataFrame())

    assert result.status is WorkflowStatus.COMPLETED
    assert result.profile is not None and result.profile.column_count == 0
    assert result.findings == () and result.assessments == ()
    assert result.dataset_risk is RiskLevel.LOW
    assert list(result.steps) == STEPS_PLAIN


def test_results_are_deterministic() -> None:
    first = run(contact_frame(), MockClassificationProvider())
    second = run(contact_frame(), MockClassificationProvider())

    for field in ("status", "profile", "findings", "classifications", "assessments", "dataset_risk", "stats", "issues", "steps"):
        assert getattr(first, field) == getattr(second, field)


# --- Mock mode -----------------------------------------------------------------------------


def test_mock_mode_classifies_every_column() -> None:
    result = run(contact_frame(), MockClassificationProvider())

    assert result.status is WorkflowStatus.COMPLETED
    assert list(result.steps) == STEPS_CLASSIFIED
    assert [r.column_ref for r in result.classifications] == ["col_1", "col_2", "col_3"]
    assert {r.column_name: r.label for r in result.classifications} == {
        "contact_email": ClassificationLabel.PERSONAL,
        "credit_limit": ClassificationLabel.FINANCIAL,
        "city": ClassificationLabel.UNKNOWN,
    }
    assert result.stats == ClassificationStats(requested=True, attempted=3, succeeded=3)
    assert result.issues == ()


def test_mock_mode_assessments() -> None:
    assessments = by_name(run(contact_frame(), MockClassificationProvider()))

    email, credit, city = assessments["contact_email"], assessments["credit_limit"], assessments["city"]
    assert (email.risk, email.needs_review, email.label) == (RiskLevel.HIGH, True, ClassificationLabel.PERSONAL)
    # A financial suggestion raises risk to medium and asks for review, with no finding at all.
    assert (credit.risk, credit.needs_review, credit.label) == (RiskLevel.MEDIUM, True, ClassificationLabel.FINANCIAL)
    # An uncertain label needs review but does not by itself raise risk.
    assert (city.risk, city.needs_review, city.label) == (RiskLevel.LOW, True, ClassificationLabel.UNKNOWN)
    assert {a.classification_status for a in assessments.values()} == {ResultStatus.OK}
    assert {a.review_recommendation for a in assessments.values()} == {ReviewRecommendation.REQUIRED}


def test_mock_results_are_labelled_mock() -> None:
    result = run(contact_frame(), MockClassificationProvider())

    for classification in result.classifications:
        assert classification.source is ClassificationSource.MOCK
        assert classification.explanation.startswith("[MOCK] ")
        assert "not an LLM" in classification.display_source


def test_mock_mode_with_no_columns() -> None:
    result = run(pd.DataFrame(), MockClassificationProvider())

    assert result.status is WorkflowStatus.COMPLETED
    assert result.classifications == ()
    assert result.stats == ClassificationStats(requested=True)


# --- Model failure -------------------------------------------------------------------------------


def test_provider_unavailable_keeps_deterministic_results() -> None:
    df = contact_frame()
    baseline = run(df)

    result = run(df, CountingProvider(default=DOWN))

    assert result.status is WorkflowStatus.PARTIALLY_COMPLETED
    assert result.findings == baseline.findings
    assert result.profile == baseline.profile
    assert {c.status for c in result.classifications} == {ResultStatus.PROVIDER_UNAVAILABLE}
    assert {c.source for c in result.classifications} == {ClassificationSource.NONE}
    assert result.stats.failed == 3 and result.stats.succeeded == 0
    assert [i.code for i in result.issues] == ["classification_failed_columns"]
    # Deterministic policy still applies without any usable suggestion.
    assessments = by_name(result)
    assert (assessments["contact_email"].risk, assessments["contact_email"].needs_review) == (RiskLevel.HIGH, True)
    assert (assessments["credit_limit"].risk, assessments["credit_limit"].needs_review) == (RiskLevel.LOW, False)
    assert all(a.label is None for a in assessments.values())
    assert all(a.classification_status is ResultStatus.PROVIDER_UNAVAILABLE for a in assessments.values())


def test_circuit_breaker_stops_calling_the_provider() -> None:
    provider = CountingProvider(default=DOWN)

    result = run(frame("a", "b", "c", "d", "e"), provider)

    assert len(provider.seen) == 3
    assert result.stats == ClassificationStats(
        requested=True, attempted=3, failed=3, skipped_by_breaker=2, circuit_opened=True
    )
    assert len(result.classifications) == 3
    assert {i.code for i in result.issues} == {"classification_failed_columns", "classification_circuit_open"}
    assert result.status is WorkflowStatus.PARTIALLY_COMPLETED
    assert len(result.assessments) == 5


def test_a_success_resets_the_failure_streak() -> None:
    provider = CountingProvider(GOOD, DOWN, GOOD, DOWN, GOOD)

    result = run(frame("a", "b", "c", "d", "e"), provider, max_consecutive_failures=2)

    assert len(provider.seen) == 5
    assert result.stats == ClassificationStats(requested=True, attempted=5, succeeded=3, failed=2)
    assert not result.stats.circuit_opened
    assert [i.code for i in result.issues] == ["classification_failed_columns"]
    assert result.status is WorkflowStatus.PARTIALLY_COMPLETED


def test_column_cap_limits_provider_calls() -> None:
    provider = CountingProvider()

    result = run(frame("a", "b", "c", "d"), provider, max_classified_columns=2)

    assert len(provider.seen) == 2
    assert result.stats == ClassificationStats(requested=True, attempted=2, succeeded=2, skipped_by_cap=2)
    assert [i.code for i in result.issues] == ["classification_skipped_by_cap"]
    assert result.status is WorkflowStatus.PARTIALLY_COMPLETED
    assert [a.label for a in result.assessments] == [
        ClassificationLabel.INTERNAL,
        ClassificationLabel.INTERNAL,
        None,
        None,
    ]
    assert result.assessments[2].classification_status is None


def test_invalid_provider_output_is_partial() -> None:
    result = run(frame("a", "b"), CountingProvider(default="not a mapping"))

    assert result.status is WorkflowStatus.PARTIALLY_COMPLETED
    assert {c.status for c in result.classifications} == {ResultStatus.INVALID_OUTPUT}
    assert result.stats.failed == 2


def test_classifier_bug_is_a_warning_not_a_failure() -> None:
    df = contact_frame()
    baseline = run(df)

    result = run_scan_workflow(
        df,
        dataset_name="test",
        options=WorkflowOptions(classify=True),
        classifier=BrokenClassifier(),  # type: ignore[arg-type]
    )

    assert result.status is WorkflowStatus.PARTIALLY_COMPLETED
    assert result.issues == (WorkflowIssue("classify_columns", "step_failed", "RuntimeError"),)
    assert result.findings == baseline.findings
    assert result.classifications == ()
    assert result.stats == ClassificationStats(requested=True)
    assert list(result.steps) == STEPS_CLASSIFIED  # later steps still ran
    assert len(result.assessments) == 3
    assert "SECRET" not in repr(result)


def test_provider_exception_text_is_not_recorded(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    result = run(frame("a"), CountingProvider(default=RuntimeError("SECRET-DETAIL")))

    assert result.status is WorkflowStatus.PARTIALLY_COMPLETED
    assert result.classifications[0].status is ResultStatus.PROVIDER_ERROR
    assert "SECRET" not in repr(result)
    assert "SECRET" not in json.dumps(result.to_summary())
    assert "SECRET" not in caplog.text


# --- Fatal failures -------------------------------------------------------------------------------


def test_detection_failure_fails_the_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = CountingProvider()
    monkeypatch.setattr(scan_graph, "run_detectors", explode)

    result = run(contact_frame(), provider)

    assert result.status is WorkflowStatus.FAILED
    assert list(result.steps) == ["validate_input", "profile_dataset", "run_checks", "finalize"]
    assert result.issues == (WorkflowIssue("run_checks", "step_failed", "RuntimeError"),)
    assert result.findings == () and result.classifications == () and result.assessments == ()
    assert result.profile is not None  # profiling had succeeded before the failure
    assert result.stats.requested is True and result.stats.attempted == 0
    assert provider.seen == []  # nothing was sent to the provider
    assert "SECRET" not in repr(result)


def test_profiling_failure_fails_the_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scan_graph, "profile_dataframe", explode)

    result = run(contact_frame())

    assert result.status is WorkflowStatus.FAILED
    assert list(result.steps) == ["validate_input", "profile_dataset", "finalize"]
    assert result.issues == (WorkflowIssue("profile_dataset", "step_failed", "RuntimeError"),)
    assert result.profile is None


def test_duplicate_column_names_fail_validation() -> None:
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])

    result = run(df)

    assert result.status is WorkflowStatus.FAILED
    assert list(result.steps) == ["validate_input", "finalize"]
    assert result.issues == (WorkflowIssue("validate_input", "invalid_input"),)
    assert result.profile is None


def test_failure_logs_contain_class_names_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="app.workflows.scan_graph")
    monkeypatch.setattr(scan_graph, "run_detectors", explode)

    run(contact_frame())

    assert "Workflow step failed: step=run_checks error_type=RuntimeError" in caplog.text
    assert "Workflow finished: status=failed" in caplog.text
    assert "SECRET" not in caplog.text
    assert "contact_email" not in caplog.text


# --- Policy table ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "recommendation", "risk", "needs_review"),
    [
        pytest.param("personal", "review_required", RiskLevel.MEDIUM, True, id="personal"),
        pytest.param("financial", "review_required", RiskLevel.MEDIUM, True, id="financial"),
        pytest.param("unknown", "review_required", RiskLevel.LOW, True, id="unknown"),
        pytest.param("internal", "review_recommended", RiskLevel.LOW, False, id="internal"),
        pytest.param("public", "review_optional", RiskLevel.LOW, False, id="public"),
    ],
)
def test_label_policy_table(label: str, recommendation: str, risk: RiskLevel, needs_review: bool) -> None:
    answer = {**GOOD, "label": label, "review_recommendation": recommendation}

    result = run(frame("c"), CountingProvider(default=answer))

    assessment = result.assessments[0]
    assert assessment.label is ClassificationLabel(label)
    assert (assessment.risk, assessment.needs_review) == (risk, needs_review)


# --- Input validation ------------------------------------------------------------------------------------


def test_non_dataframe_is_rejected() -> None:
    with pytest.raises(TypeError):
        run_scan_workflow("not a frame", dataset_name="test")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "   ", None], ids=["empty", "spaces", "none"])
def test_blank_dataset_name_is_rejected(name: Any) -> None:
    with pytest.raises(ValueError):
        run_scan_workflow(contact_frame(), dataset_name=name)


def test_classification_requires_a_classifier() -> None:
    with pytest.raises(ValueError):
        run_scan_workflow(contact_frame(), dataset_name="test", options=WorkflowOptions(classify=True))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_classified_columns": 0},
        {"max_classified_columns": 201},
        {"max_classified_columns": True},
        {"max_consecutive_failures": 0},
        {"max_consecutive_failures": 11},
        {"classify": "yes"},
    ],
    ids=["cap-zero", "cap-too-large", "cap-boolean", "streak-zero", "streak-too-large", "classify-not-bool"],
)
def test_option_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        WorkflowOptions(**kwargs)


# --- Structure and safety ----------------------------------------------------------------------------------


def test_step_order_without_classification() -> None:
    assert list(run(contact_frame()).steps) == STEPS_PLAIN


def test_step_order_with_classification() -> None:
    assert list(run(contact_frame(), MockClassificationProvider()).steps) == STEPS_CLASSIFIED


def test_step_durations_are_recorded() -> None:
    result = run(contact_frame(), MockClassificationProvider())

    assert set(result.durations_ms) == set(result.steps) - {"finalize"}
    assert all(duration >= 0.0 for duration in result.durations_ms.values())


def test_workflow_is_bounded() -> None:
    result = run(contact_frame(), MockClassificationProvider())

    assert scan_graph.RECURSION_LIMIT <= 25
    assert len(result.steps) <= 7


def test_summary_is_json_safe_and_counts_only() -> None:
    summary = run(contact_frame(), MockClassificationProvider()).to_summary()

    json.dumps(summary)  # must be serializable
    assert set(summary) == {"status", "steps", "classification", "risk", "issues", "durations_ms"}
    assert summary["status"] == "completed"
    assert summary["steps"] == STEPS_CLASSIFIED
    assert summary["classification"] == {
        "requested": True,
        "attempted": 3,
        "succeeded": 3,
        "failed": 0,
        "skipped_by_cap": 0,
        "skipped_by_breaker": 0,
        "circuit_opened": False,
        "provider": "mock-rules-v1",
    }
    assert summary["risk"] == {
        "dataset": "high",
        "columns": {"low": 1, "medium": 1, "high": 1},
        "needs_review_columns": 3,
    }
    assert summary["issues"] == []


def test_summary_for_a_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scan_graph, "run_detectors", explode)

    summary = run(contact_frame()).to_summary()

    json.dumps(summary)
    assert summary["status"] == "failed"
    assert summary["classification"]["provider"] is None
    assert summary["risk"]["dataset"] is None
    assert summary["issues"] == [{"step": "run_checks", "code": "step_failed", "error_type": "RuntimeError"}]


def test_no_raw_values_in_results_or_provider_input() -> None:
    df = pd.DataFrame(
        {
            "contact_email": ["SECRET-CELL-1@example.com", "SECRET-CELL-2@example.com"],
            "note": ["SECRET-NOTE-VALUE", None],
        }
    )
    provider = CountingProvider()

    result = run(df, provider)

    assert len(provider.seen) == 2
    assert "SECRET" not in repr(result)
    assert "SECRET" not in json.dumps(result.to_summary())
    assert "SECRET" not in " ".join(metadata.model_dump_json() for metadata in provider.seen)


def test_result_has_no_dataframe_field() -> None:
    result = run(contact_frame(), MockClassificationProvider())

    for field in dataclasses.fields(WorkflowResult):
        assert not isinstance(getattr(result, field.name), pd.DataFrame)


def test_build_scan_graph_returns_an_invokable_graph() -> None:
    graph = build_scan_graph(contact_frame(), options=WorkflowOptions(), classifier=None)

    assert callable(graph.invoke)


def test_workflow_does_not_modify_the_input_frame() -> None:
    df = contact_frame()
    snapshot = df.copy(deep=True)

    run(df, MockClassificationProvider())

    pd.testing.assert_frame_equal(df, snapshot)