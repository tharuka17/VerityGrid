import logging
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from app.ai.classifier import (
    NOTE_CONFLICT,
    NOTE_SENSITIVE,
    ClassificationResult,
    ColumnClassifier,
    ResultStatus,
    build_column_metadata,
    sanitize_text,
)
from app.ai.mock_provider import MockClassificationProvider
from app.ai.provider import (
    MAX_COLUMN_NAME_LENGTH,
    MOCK_PREFIX,
    ClassificationLabel,
    ClassificationProvider,
    ClassificationSource,
    ClassificationSuggestion,
    ColumnMetadata,
    FindingSummary,
    ProviderUnavailableError,
    ReviewRecommendation,
    has_control_characters,
)
from app.db.models import Severity
from app.schemas.findings import DetectedFinding, FindingCategory
from app.services.detectors import run_detectors
from app.services.profiling import ColumnProfileResult, InferredType, profile_dataframe

EMAIL = FindingCategory.POTENTIAL_EMAIL
PHONE = FindingCategory.POTENTIAL_PHONE

GOOD: dict[str, Any] = {
    "label": "personal",
    "explanation": "Email-like values were found.",
    "supporting_refs": ["column_name"],
    "review_recommendation": "review_required",
}


# --- Helpers and test doubles ---------------------------------------------------


class StubProvider(ClassificationProvider):
    name = "stub"
    source = ClassificationSource.LLM

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[ColumnMetadata] = []

    def classify(self, metadata: ColumnMetadata) -> Any:
        self.calls.append(metadata)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class MockSourceStub(StubProvider):
    name = "mock-stub"
    source = ClassificationSource.MOCK


class NoneSourceStub(StubProvider):
    source = ClassificationSource.NONE


class StringSourceStub(StubProvider):
    source = "llm"  # type: ignore[assignment]


class SequenceStub(StubProvider):
    """Returns (or raises) one outcome per call, in order."""

    def __init__(self, *outcomes: Any) -> None:
        super().__init__(None)
        self.outcomes = list(outcomes)

    def classify(self, metadata: ColumnMetadata) -> Any:
        self.calls.append(metadata)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_profile(
    name: str = "email",
    *,
    inferred: InferredType = InferredType.STRING,
    rows: int = 10,
    nulls: int = 2,
    distinct: int | None = 4,
) -> ColumnProfileResult:
    return ColumnProfileResult(
        column_name=name,
        pandas_dtype="object",
        inferred_type=inferred,
        row_count=rows,
        null_count=nulls,
        null_percent=round(100.0 * nulls / rows, 2) if rows else 0.0,
        distinct_count=distinct,
    )


def make_detected(
    category: FindingCategory = EMAIL, column: str | None = "email", **overrides: Any
) -> DetectedFinding:
    data: dict[str, Any] = {
        "category": category,
        "severity": Severity.HIGH,
        "column_name": column,
        "title": "TITLE-TEXT",
        "description": "DESCRIPTION-TEXT",
        "safe_evidence": {"match_count": 3, "match_percent": 100.0, "pattern_family": "email"},
        "detection_method": "regex_email_pattern",
        "recommendation": "RECOMMENDATION-TEXT",
    }
    data.update(overrides)
    return DetectedFinding(**data)


def summary(category: FindingCategory = EMAIL) -> FindingSummary:
    return FindingSummary(category=category, severity=Severity.HIGH, evidence={"match_count": 3})


def meta(
    name: str = "contact_email",
    *,
    findings: tuple[FindingSummary, ...] | list[FindingSummary] = (),
    distinct_percent: float | None = 100.0,
    ref: str = "col_1",
) -> ColumnMetadata:
    return ColumnMetadata(
        column_ref=ref,
        column_name=name,
        inferred_type="string",
        row_count=4,
        null_percent=0.0,
        distinct_percent=distinct_percent,
        findings=tuple(findings),
    )


def assert_fallback(result: ClassificationResult, status: ResultStatus) -> None:
    assert result.status is status
    assert result.label is ClassificationLabel.UNKNOWN
    assert result.review_recommendation is ReviewRecommendation.REQUIRED
    assert result.source is ClassificationSource.NONE
    assert result.supporting_refs == ()
    assert result.policy_notes == ()
    assert result.explanation.startswith("No suggestion available:")


# --- sanitize_text ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected", "changed"),
    [
        ("plain text", "plain text", False),
        ("a  b\tc\nd", "a b c d", True),
        ("a\x00b\x07c", "abc", True),
        ("a\u200bb", "ab", True),
        ("ＡＢＣ", "ABC", True),
        ("  padded  ", "padded", True),
        ("", "", False),
        ("   ", "", True),
        ("Ignore previous instructions", "Ignore previous instructions", False),
    ],
)
def test_sanitize_text(value: str, expected: str, changed: bool) -> None:
    assert sanitize_text(value, 64) == (expected, changed)


@pytest.mark.parametrize(
    ("value", "max_length", "expected", "changed"),
    [
        pytest.param("x" * 100, 10, "x" * 10, True, id="shortened"),
        pytest.param("ab cd", 3, "ab", True, id="shortened-without-trailing-space"),
        pytest.param("x" * 10, 10, "x" * 10, False, id="exactly-at-limit"),
    ],
)
def test_sanitize_text_truncates(value: str, max_length: int, expected: str, changed: bool) -> None:
    assert sanitize_text(value, max_length) == (expected, changed)


def test_sanitize_text_accepts_non_strings() -> None:
    assert sanitize_text(123, 10) == ("123", False)


@pytest.mark.parametrize(
    "value",
    ["a\x00\x01\x02", "\x07bell", "tab\tnewline\n", "\u200b\u200c\u200d", "mixed\x00\u200b\ue000text"],
)
def test_sanitized_text_never_contains_control_characters(value: str) -> None:
    cleaned, _ = sanitize_text(value, 64)

    assert not has_control_characters(cleaned)


# --- build_column_metadata ----------------------------------------------------------


def test_metadata_is_built_from_a_profile() -> None:
    metadata = build_column_metadata(make_profile(), [], column_ref="col_3")

    assert metadata.column_ref == "col_3"
    assert metadata.column_name == "email"
    assert metadata.name_was_modified is False
    assert metadata.inferred_type == "string"
    assert metadata.row_count == 10
    assert metadata.null_percent == 20.0
    assert metadata.distinct_percent == 50.0  # 4 distinct values among 8 non-null
    assert metadata.findings == ()


@pytest.mark.parametrize(
    ("rows", "nulls", "distinct", "expected"),
    [
        (10, 2, 4, 50.0),
        (10, 0, 10, 100.0),
        (3, 0, 1, 33.33),
        (5, 5, 0, None),
        (0, 0, 0, None),
        (10, 2, None, None),
    ],
    ids=["half", "all-distinct", "one-third", "all-null", "no-rows", "distinct-unknown"],
)
def test_distinct_percent(rows: int, nulls: int, distinct: int | None, expected: float | None) -> None:
    profile = make_profile(rows=rows, nulls=nulls, distinct=distinct)

    metadata = build_column_metadata(profile, [], column_ref="col_1")

    if expected is None:
        assert metadata.distinct_percent is None
    else:
        assert metadata.distinct_percent == pytest.approx(expected)


@pytest.mark.parametrize("inferred", list(InferredType))
def test_inferred_type_is_passed_through(inferred: InferredType) -> None:
    metadata = build_column_metadata(make_profile(inferred=inferred), [], column_ref="col_1")

    assert metadata.inferred_type == inferred.value


def test_untrusted_name_is_sanitized() -> None:
    metadata = build_column_metadata(make_profile("contact\x00 name"), [], column_ref="col_1")

    assert metadata.column_name == "contact name"
    assert metadata.name_was_modified is True


def test_long_name_is_truncated() -> None:
    metadata = build_column_metadata(make_profile("n" * 100), [], column_ref="col_1")

    assert len(metadata.column_name) == MAX_COLUMN_NAME_LENGTH
    assert metadata.name_was_modified is True


def test_name_made_only_of_invisible_characters_falls_back_to_the_reference() -> None:
    metadata = build_column_metadata(make_profile("\u200b\u200b"), [], column_ref="col_7")

    assert metadata.column_name == "col_7"
    assert metadata.name_was_modified is True


def test_only_findings_for_this_column_are_included() -> None:
    findings = [
        make_detected(EMAIL, "email"),
        make_detected(FindingCategory.MISSING_VALUES, "other", safe_evidence={"null_count": 2}),
        make_detected(FindingCategory.DUPLICATE_ROWS, None, safe_evidence={"duplicate_row_count": 1}),
    ]

    metadata = build_column_metadata(make_profile("email"), findings, column_ref="col_1")

    assert len(metadata.findings) == 1
    item = metadata.findings[0]
    assert item.category is EMAIL
    assert item.severity is Severity.HIGH
    assert item.evidence == {"match_count": 3, "match_percent": 100.0, "pattern_family": "email"}


def test_finding_titles_descriptions_and_recommendations_are_not_included() -> None:
    metadata = build_column_metadata(make_profile("email"), [make_detected()], column_ref="col_1")

    dump = metadata.model_dump_json()

    for text in ("TITLE-TEXT", "DESCRIPTION-TEXT", "RECOMMENDATION-TEXT", "regex_email_pattern"):
        assert text not in dump


def test_evidence_text_is_sanitized() -> None:
    finding = make_detected(safe_evidence={"pattern_family": "em\x00ail"})

    metadata = build_column_metadata(make_profile("email"), [finding], column_ref="col_1")

    assert metadata.findings[0].evidence == {"pattern_family": "email"}


def test_offered_references_follow_the_findings() -> None:
    metadata = build_column_metadata(make_profile("email"), [make_detected()], column_ref="col_1")

    assert metadata.available_refs == {
        "column_name",
        "inferred_type",
        "null_percent",
        "distinct_percent",
        "finding:potential_email",
    }


# --- Classifier: construction and successful answers -----------------------------------


@pytest.mark.parametrize(
    "provider",
    [
        pytest.param("not a provider", id="not-a-provider"),
        pytest.param(NoneSourceStub(GOOD), id="source-none"),
        pytest.param(StringSourceStub(GOOD), id="source-not-an-enum"),
    ],
)
def test_constructor_rejects_unusable_providers(provider: Any) -> None:
    with pytest.raises(TypeError):
        ColumnClassifier(provider)


def test_valid_answer_becomes_a_result() -> None:
    result = ColumnClassifier(StubProvider(GOOD)).classify_metadata(
        meta(findings=[summary()])
    )

    assert result.status is ResultStatus.OK
    assert result.column_ref == "col_1"
    assert result.column_name == "contact_email"
    assert result.label is ClassificationLabel.PERSONAL
    assert result.explanation == "Email-like values were found."
    assert result.supporting_refs == ("column_name",)
    assert result.review_recommendation is ReviewRecommendation.REQUIRED
    assert result.source is ClassificationSource.LLM
    assert result.provider_name == "stub"
    assert result.policy_notes == ()


def test_provider_receives_the_metadata_object_and_nothing_else() -> None:
    stub = StubProvider(GOOD)
    metadata = meta()

    ColumnClassifier(stub).classify_metadata(metadata)

    assert len(stub.calls) == 1
    assert stub.calls[0] is metadata
    assert isinstance(stub.calls[0], ColumnMetadata)


def test_a_validated_suggestion_object_is_accepted() -> None:
    stub = StubProvider(ClassificationSuggestion(**GOOD))

    result = ColumnClassifier(stub).classify_metadata(meta())

    assert result.status is ResultStatus.OK


# --- Classifier: invalid output -------------------------------------------------------------


INVALID_OUTPUTS = {
    "string": "not a mapping",
    "none": None,
    "number": 42,
    "list": [1, 2],
    "empty-mapping": {},
    "missing-fields": {"label": "public"},
    "label-outside-enum": {**GOOD, "label": "top-secret"},
    "provenance-spoof": {**GOOD, "source": "mock"},
    "confidence-field": {**GOOD, "confidence": 0.99},
    "explanation-control-char": {**GOOD, "explanation": "bad\x07text"},
    "explanation-too-long": {**GOOD, "explanation": "x" * 301},
    "malformed-reference": {**GOOD, "supporting_refs": ["Column Name"]},
}


@pytest.mark.parametrize("output", INVALID_OUTPUTS.values(), ids=INVALID_OUTPUTS.keys())
def test_invalid_output_becomes_a_fallback(output: Any) -> None:
    result = ColumnClassifier(StubProvider(output)).classify_metadata(meta())

    assert_fallback(result, ResultStatus.INVALID_OUTPUT)
    assert result.provider_name == "stub"


@pytest.mark.parametrize(
    ("refs", "distinct_percent"),
    [
        pytest.param(["finding:invalid_date"], 100.0, id="finding-not-in-request"),
        pytest.param(["distinct_percent"], None, id="distinct-percent-not-offered"),
        pytest.param(["column_name", "made_up_ref"], 100.0, id="invented-reference"),
    ],
)
def test_references_that_were_not_offered_are_rejected(
    refs: list[str], distinct_percent: float | None
) -> None:
    answer = {**GOOD, "supporting_refs": refs}

    result = ColumnClassifier(StubProvider(answer)).classify_metadata(
        meta(findings=[summary()], distinct_percent=distinct_percent)
    )

    assert_fallback(result, ResultStatus.INVALID_OUTPUT)


def test_duplicate_references_are_removed_in_order() -> None:
    answer = {**GOOD, "supporting_refs": ["column_name", "column_name", "inferred_type"]}

    result = ColumnClassifier(StubProvider(answer)).classify_metadata(meta())

    assert result.status is ResultStatus.OK
    assert result.supporting_refs == ("column_name", "inferred_type")


def test_offered_references_are_accepted() -> None:
    answer = {**GOOD, "supporting_refs": ["distinct_percent", "inferred_type", "null_percent"]}

    result = ColumnClassifier(StubProvider(answer)).classify_metadata(meta())

    assert result.status is ResultStatus.OK
    assert result.supporting_refs == ("distinct_percent", "inferred_type", "null_percent")


# --- Classifier: provider failures ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_error", "status"),
    [
        pytest.param(lambda: ProviderUnavailableError("down"), ResultStatus.PROVIDER_UNAVAILABLE, id="unavailable"),
        pytest.param(lambda: RuntimeError("boom"), ResultStatus.PROVIDER_ERROR, id="runtime-error"),
        pytest.param(lambda: ValueError("boom"), ResultStatus.PROVIDER_ERROR, id="value-error"),
        pytest.param(lambda: KeyError("k"), ResultStatus.PROVIDER_ERROR, id="key-error"),
        pytest.param(lambda: TimeoutError("slow"), ResultStatus.PROVIDER_ERROR, id="timeout"),
    ],
)
def test_provider_failures_become_fallbacks(make_error: Any, status: ResultStatus) -> None:
    result = ColumnClassifier(StubProvider(make_error())).classify_metadata(meta())

    assert_fallback(result, status)
    assert result.provider_name == "stub"


def test_each_failure_status_has_its_own_message() -> None:
    results = [
        ColumnClassifier(StubProvider(ProviderUnavailableError("x"))).classify_metadata(meta()),
        ColumnClassifier(StubProvider(RuntimeError("x"))).classify_metadata(meta()),
        ColumnClassifier(StubProvider("invalid")).classify_metadata(meta()),
    ]

    assert [r.status for r in results] == [
        ResultStatus.PROVIDER_UNAVAILABLE,
        ResultStatus.PROVIDER_ERROR,
        ResultStatus.INVALID_OUTPUT,
    ]
    assert len({r.explanation for r in results}) == 3
    assert all(r.explanation.endswith("manually.") for r in results)


def test_exception_text_never_reaches_the_result() -> None:
    error = RuntimeError("SECRET-DETAIL-IN-ERROR")

    result = ColumnClassifier(StubProvider(error)).classify_metadata(meta())

    assert "SECRET" not in repr(result)


def test_logs_contain_no_names_explanations_or_error_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="app.ai.classifier")
    answer = {**GOOD, "explanation": "UNIQUE-EXPLANATION-TEXT"}
    name = "secretcolumn"

    ColumnClassifier(StubProvider(answer)).classify_metadata(meta(name))
    ColumnClassifier(StubProvider(RuntimeError("SECRET-DETAIL"))).classify_metadata(meta(name))
    ColumnClassifier(StubProvider(ProviderUnavailableError("SECRET-DETAIL"))).classify_metadata(meta(name))
    ColumnClassifier(StubProvider({"label": "bad"})).classify_metadata(meta(name))

    assert "Classified: column_ref=col_1 provider=stub label=personal" in caplog.text
    assert "Provider failed: column_ref=col_1 provider=stub error_type=RuntimeError" in caplog.text
    assert "Provider unavailable: column_ref=col_1 provider=stub" in caplog.text
    assert "Invalid provider output: column_ref=col_1 provider=stub" in caplog.text
    for forbidden in (name, "UNIQUE-EXPLANATION-TEXT", "SECRET"):
        assert forbidden not in caplog.text


# --- Classifier: policy ----------------------------------------------------------------------------------


def answer_with(label: str, recommendation: str) -> dict[str, Any]:
    return {
        "label": label,
        "explanation": "Some reasoning.",
        "supporting_refs": ["column_name"],
        "review_recommendation": recommendation,
    }


@pytest.mark.parametrize("label", ["personal", "financial", "unknown"])
@pytest.mark.parametrize(
    ("recommendation", "notes"),
    [
        ("review_optional", (NOTE_SENSITIVE,)),
        ("review_recommended", (NOTE_SENSITIVE,)),
        ("review_required", ()),
    ],
)
def test_sensitive_and_uncertain_labels_always_require_review(
    label: str, recommendation: str, notes: tuple[str, ...]
) -> None:
    result = ColumnClassifier(StubProvider(answer_with(label, recommendation))).classify_metadata(meta())

    assert result.label.value == label  # policy never rewrites the label
    assert result.review_recommendation is ReviewRecommendation.REQUIRED
    assert result.policy_notes == notes


@pytest.mark.parametrize("label", ["public", "internal"])
@pytest.mark.parametrize("recommendation", ["review_optional", "review_recommended", "review_required"])
def test_public_and_internal_recommendations_are_left_alone(label: str, recommendation: str) -> None:
    result = ColumnClassifier(StubProvider(answer_with(label, recommendation))).classify_metadata(meta())

    assert result.review_recommendation.value == recommendation
    assert result.policy_notes == ()


@pytest.mark.parametrize(
    ("label", "recommendation", "category", "notes"),
    [
        pytest.param("public", "review_optional", EMAIL, (NOTE_CONFLICT,), id="public-vs-email"),
        pytest.param("internal", "review_recommended", PHONE, (NOTE_CONFLICT,), id="internal-vs-phone"),
        pytest.param("financial", "review_optional", EMAIL, (NOTE_SENSITIVE, NOTE_CONFLICT), id="financial-optional-vs-email"),
        pytest.param("financial", "review_required", EMAIL, (NOTE_CONFLICT,), id="financial-required-vs-email"),
        pytest.param("unknown", "review_required", PHONE, (NOTE_CONFLICT,), id="unknown-vs-phone"),
        pytest.param("personal", "review_optional", EMAIL, (NOTE_SENSITIVE,), id="personal-agrees-with-email"),
        pytest.param("personal", "review_required", EMAIL, (), id="personal-required-agrees"),
    ],
)
def test_labels_that_conflict_with_email_or_phone_findings_are_flagged(
    label: str, recommendation: str, category: FindingCategory, notes: tuple[str, ...]
) -> None:
    result = ColumnClassifier(StubProvider(answer_with(label, recommendation))).classify_metadata(
        meta(findings=[summary(category)])
    )

    assert result.label.value == label
    assert result.review_recommendation is ReviewRecommendation.REQUIRED
    assert result.policy_notes == notes


def test_other_findings_do_not_trigger_the_conflict_rule() -> None:
    result = ColumnClassifier(StubProvider(answer_with("public", "review_optional"))).classify_metadata(
        meta(findings=[summary(FindingCategory.MISSING_VALUES)])
    )

    assert result.review_recommendation is ReviewRecommendation.OPTIONAL
    assert result.policy_notes == ()


# --- Classifier: mock labelling and display text ---------------------------------------------------------


MARKER_ANSWER = answer_with("public", "review_optional")


def test_mock_source_explanations_always_carry_the_marker() -> None:
    answer = {**MARKER_ANSWER, "explanation": "No marker here."}

    result = ColumnClassifier(MockSourceStub(answer)).classify_metadata(meta())

    assert result.explanation == "[MOCK] No marker here."
    assert result.source is ClassificationSource.MOCK
    assert result.provider_name == "mock-stub"


def test_an_existing_marker_is_not_doubled() -> None:
    answer = {**MARKER_ANSWER, "explanation": "[MOCK] Already marked."}

    result = ColumnClassifier(MockSourceStub(answer)).classify_metadata(meta())

    assert result.explanation == "[MOCK] Already marked."


def test_the_marker_is_kept_when_the_text_is_at_the_length_limit() -> None:
    answer = {**MARKER_ANSWER, "explanation": "x" * 300}

    result = ColumnClassifier(MockSourceStub(answer)).classify_metadata(meta())

    assert result.explanation.startswith(MOCK_PREFIX)
    assert len(result.explanation) == 300


def test_llm_source_explanations_get_no_marker() -> None:
    answer = {**MARKER_ANSWER, "explanation": "No marker here."}

    result = ColumnClassifier(StubProvider(answer)).classify_metadata(meta())

    assert result.explanation == "No marker here."
    assert result.source is ClassificationSource.LLM


def test_display_text_distinguishes_mock_llm_and_no_suggestion() -> None:
    mock = ColumnClassifier(MockSourceStub(MARKER_ANSWER)).classify_metadata(meta())
    llm = ColumnClassifier(StubProvider(MARKER_ANSWER)).classify_metadata(meta())
    none = ColumnClassifier(StubProvider(RuntimeError("x"))).classify_metadata(meta())

    assert mock.display_source == "Mock suggestion (rule-based, not an LLM)"
    assert "unverified" in llm.display_source
    assert "human" in llm.display_source
    assert none.display_source == "No suggestion"


def test_results_are_immutable_and_reject_unknown_fields() -> None:
    result = ColumnClassifier(StubProvider(GOOD)).classify_metadata(meta())

    with pytest.raises(ValidationError):
        result.label = ClassificationLabel.PUBLIC  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ClassificationResult(**{**result.model_dump(), "unexpected": 1})


def test_classify_column_uses_a_reference_and_defaults_to_col_1() -> None:
    stub = StubProvider(GOOD)
    classifier = ColumnClassifier(stub)

    default = classifier.classify_column(make_profile(), [])
    custom = classifier.classify_column(make_profile(), [], column_ref="col_7")

    assert (default.column_ref, custom.column_ref) == ("col_1", "col_7")
    assert [call.column_ref for call in stub.calls] == ["col_1", "col_7"]


# --- Classifier: whole datasets ---------------------------------------------------------------------------------


def test_classify_dataset_returns_one_result_per_column_in_order() -> None:
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    stub = StubProvider(GOOD)

    results = ColumnClassifier(stub).classify_dataset(profile_dataframe(df), run_detectors(df))

    assert [(r.column_ref, r.column_name) for r in results] == [("col_1", "a"), ("col_2", "b"), ("col_3", "c")]
    assert len(stub.calls) == 3


def test_classify_dataset_with_no_columns() -> None:
    stub = StubProvider(GOOD)

    results = ColumnClassifier(stub).classify_dataset(profile_dataframe(pd.DataFrame()), [])

    assert results == []
    assert stub.calls == []


def test_each_column_receives_only_its_own_findings() -> None:
    df = pd.DataFrame({"contact_email": ["a@example.com", "b@example.com"], "qty": [1, 2]})
    stub = SequenceStub(GOOD, GOOD)

    ColumnClassifier(stub).classify_dataset(profile_dataframe(df), run_detectors(df))

    assert [call.column_name for call in stub.calls] == ["contact_email", "qty"]
    assert [f.category for f in stub.calls[0].findings] == [EMAIL]
    assert stub.calls[1].findings == ()


def test_one_failing_column_does_not_affect_the_others() -> None:
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    stub = SequenceStub(GOOD, RuntimeError("boom"), GOOD)

    results = ColumnClassifier(stub).classify_dataset(profile_dataframe(df), run_detectors(df))

    assert [r.status for r in results] == [
        ResultStatus.OK,
        ResultStatus.PROVIDER_ERROR,
        ResultStatus.OK,
    ]


def test_no_raw_cell_values_reach_the_provider() -> None:
    df = pd.DataFrame(
        {
            "contact_email": ["SECRET-CELL-1@example.com", "SECRET-CELL-2@example.com"],
            "note": ["SECRET-NOTE-VALUE", None],
        }
    )
    stub = SequenceStub(GOOD, GOOD)

    ColumnClassifier(stub).classify_dataset(profile_dataframe(df), run_detectors(df))

    payload = " ".join(call.model_dump_json() for call in stub.calls)
    assert "contact_email" in payload  # sanitized names are sent as data
    assert "SECRET" not in payload
    assert "Potential email addresses" not in payload  # finding titles are never sent


def test_mock_provider_end_to_end() -> None:
    df = pd.DataFrame(
        {"contact_email": ["a@example.com", "b@example.com"], "credit_limit": [100, 200], "city": ["x", "y"]}
    )

    results = ColumnClassifier(MockClassificationProvider()).classify_dataset(
        profile_dataframe(df), run_detectors(df)
    )

    assert [r.label for r in results] == [
        ClassificationLabel.PERSONAL,
        ClassificationLabel.FINANCIAL,
        ClassificationLabel.UNKNOWN,
    ]
    for result in results:
        assert result.status is ResultStatus.OK
        assert result.source is ClassificationSource.MOCK
        assert result.explanation.startswith(MOCK_PREFIX)