from typing import Any

import pytest
from pydantic import ValidationError

from app.ai.provider import (
    MAX_COLUMN_NAME_LENGTH,
    MAX_EXPLANATION_LENGTH,
    MAX_REFS,
    MOCK_PREFIX,
    RECOMMENDATION_RANK,
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
from app.schemas.findings import FindingCategory

VALID_METADATA: dict[str, Any] = {
    "column_ref": "col_1",
    "column_name": "email",
    "inferred_type": "string",
    "row_count": 10,
    "null_percent": 20.0,
    "distinct_percent": 50.0,
}
VALID_ANSWER: dict[str, Any] = {
    "label": "personal",
    "explanation": "Contact data.",
    "supporting_refs": ["column_name"],
    "review_recommendation": "review_required",
}


def summary(category: FindingCategory = FindingCategory.POTENTIAL_EMAIL) -> FindingSummary:
    return FindingSummary(category=category, severity=Severity.HIGH, evidence={"match_count": 3})


# --- Enumerations and constants ------------------------------------------------


def test_label_values_are_the_five_agreed_labels() -> None:
    assert [label.value for label in ClassificationLabel] == [
        "public",
        "internal",
        "personal",
        "financial",
        "unknown",
    ]


def test_review_recommendation_values_and_ranking() -> None:
    assert {item.value for item in ReviewRecommendation} == {
        "review_required",
        "review_recommended",
        "review_optional",
    }
    assert set(RECOMMENDATION_RANK) == set(ReviewRecommendation)
    assert (
        RECOMMENDATION_RANK[ReviewRecommendation.OPTIONAL]
        < RECOMMENDATION_RANK[ReviewRecommendation.RECOMMENDED]
        < RECOMMENDATION_RANK[ReviewRecommendation.REQUIRED]
    )


def test_source_values_and_mock_prefix() -> None:
    assert {item.value for item in ClassificationSource} == {"mock", "llm", "none"}
    assert MOCK_PREFIX == "[MOCK] "


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain", False),
        ("with space", False),
        ("tab\there", True),
        ("new\nline", True),
        ("bell\x07", True),
        ("zero\u200bwidth", True),
        ("café", False),
        ("emoji 😀", False),
        ("private\ue000use", True),
    ],
)
def test_control_character_detection(text: str, expected: bool) -> None:
    assert has_control_characters(text) is expected


# --- Provider base class -------------------------------------------------------


def test_provider_base_class_is_abstract() -> None:
    with pytest.raises(TypeError):
        ClassificationProvider()  # type: ignore[abstract]


def test_provider_without_classify_cannot_be_created() -> None:
    class Incomplete(ClassificationProvider):
        name = "incomplete"
        source = ClassificationSource.LLM

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_concrete_provider_can_be_created() -> None:
    class Concrete(ClassificationProvider):
        name = "concrete"
        source = ClassificationSource.LLM

        def classify(self, metadata: ColumnMetadata) -> dict[str, Any]:
            return {}

    assert Concrete().classify(ColumnMetadata(**VALID_METADATA)) == {}


def test_provider_unavailable_error_is_an_exception() -> None:
    error = ProviderUnavailableError("offline")

    assert isinstance(error, Exception)
    assert str(error) == "offline"


# --- ColumnMetadata ------------------------------------------------------------


def test_valid_metadata_uses_safe_defaults() -> None:
    metadata = ColumnMetadata(**VALID_METADATA)

    assert metadata.findings == ()
    assert metadata.name_was_modified is False


INVALID_METADATA = {
    "ref-without-number": {"column_ref": "col_"},
    "ref-wrong-prefix": {"column_ref": "c1"},
    "ref-too-many-digits": {"column_ref": "col_12345"},
    "ref-not-digits": {"column_ref": "col_a"},
    "name-empty": {"column_name": ""},
    "name-too-long": {"column_name": "n" * (MAX_COLUMN_NAME_LENGTH + 1)},
    "name-control-char": {"column_name": "bad\x07name"},
    "name-zero-width": {"column_name": "zero\u200bwidth"},
    "type-empty": {"inferred_type": ""},
    "type-too-long": {"inferred_type": "t" * 31},
    "negative-row-count": {"row_count": -1},
    "null-percent-below-zero": {"null_percent": -0.1},
    "null-percent-above-100": {"null_percent": 100.1},
    "distinct-percent-below-zero": {"distinct_percent": -1.0},
    "distinct-percent-above-100": {"distinct_percent": 100.1},
}


@pytest.mark.parametrize("overrides", INVALID_METADATA.values(), ids=INVALID_METADATA.keys())
def test_invalid_metadata_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ColumnMetadata(**{**VALID_METADATA, **overrides})


def test_metadata_boundary_values_are_accepted() -> None:
    low = ColumnMetadata(
        **{
            **VALID_METADATA,
            "column_ref": "col_9999",
            "column_name": "n" * MAX_COLUMN_NAME_LENGTH,
            "row_count": 0,
            "null_percent": 0.0,
            "distinct_percent": 0.0,
        }
    )
    high = ColumnMetadata(**{**VALID_METADATA, "null_percent": 100.0, "distinct_percent": 100.0})

    assert len(low.column_name) == MAX_COLUMN_NAME_LENGTH
    assert (high.null_percent, high.distinct_percent) == (100.0, 100.0)


def test_metadata_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ColumnMetadata(**VALID_METADATA, raw_values=["x"])


def test_metadata_is_immutable() -> None:
    metadata = ColumnMetadata(**VALID_METADATA)

    with pytest.raises(ValidationError):
        metadata.column_name = "other"  # type: ignore[misc]


def test_available_refs_without_findings() -> None:
    metadata = ColumnMetadata(**VALID_METADATA)

    assert metadata.available_refs == {
        "column_name",
        "inferred_type",
        "null_percent",
        "distinct_percent",
    }


def test_available_refs_without_distinct_percent() -> None:
    metadata = ColumnMetadata(**{**VALID_METADATA, "distinct_percent": None})

    assert metadata.available_refs == {"column_name", "inferred_type", "null_percent"}


def test_available_refs_include_one_per_finding() -> None:
    metadata = ColumnMetadata(
        **VALID_METADATA,
        findings=(summary(), summary(FindingCategory.MISSING_VALUES)),
    )

    assert {"finding:potential_email", "finding:missing_values"} <= metadata.available_refs
    assert len(metadata.available_refs) == 6


# --- FindingSummary ------------------------------------------------------------


def test_valid_finding_summary() -> None:
    item = FindingSummary(
        category=FindingCategory.POTENTIAL_EMAIL,
        severity=Severity.HIGH,
        evidence={"match_count": 3, "match_percent": 100.0, "flag": True, "pattern_family": "email"},
    )

    assert item.evidence["match_count"] == 3


INVALID_EVIDENCE = {
    "too-many-items": {f"k{i}": i for i in range(21)},
    "key-too-long": {"k" * 61: 1},
    "key-control-char": {"bad\x07": 1},
    "text-too-long": {"k": "x" * 121},
    "text-control-char": {"k": "bad\x07"},
    "list-value": {"k": [1, 2]},
    "dict-value": {"k": {"a": 1}},
    "none-value": {"k": None},
}


@pytest.mark.parametrize("evidence", INVALID_EVIDENCE.values(), ids=INVALID_EVIDENCE.keys())
def test_invalid_evidence_is_rejected(evidence: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FindingSummary(category=FindingCategory.POTENTIAL_EMAIL, severity=Severity.HIGH, evidence=evidence)


def test_evidence_boundary_values_are_accepted() -> None:
    many = {f"k{i}": i for i in range(20)}
    long_key = {"k" * 60: 1}
    long_text = {"k": "x" * 120}

    for evidence in (many, long_key, long_text):
        FindingSummary(category=FindingCategory.POTENTIAL_EMAIL, severity=Severity.HIGH, evidence=evidence)


def test_finding_summary_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        FindingSummary(category="bogus", severity=Severity.HIGH, evidence={"a": 1})  # type: ignore[arg-type]


def test_finding_summary_rejects_unknown_severity() -> None:
    with pytest.raises(ValidationError):
        FindingSummary(category=FindingCategory.POTENTIAL_EMAIL, severity="critical", evidence={"a": 1})  # type: ignore[arg-type]


def test_finding_summary_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        FindingSummary(
            category=FindingCategory.POTENTIAL_EMAIL,
            severity=Severity.HIGH,
            evidence={"a": 1},
            title="not allowed",  # type: ignore[call-arg]
        )


# --- ClassificationSuggestion ----------------------------------------------------


def test_valid_suggestion() -> None:
    suggestion = ClassificationSuggestion(**VALID_ANSWER)

    assert suggestion.label is ClassificationLabel.PERSONAL
    assert suggestion.review_recommendation is ReviewRecommendation.REQUIRED
    assert suggestion.supporting_refs == ("column_name",)  # a list is converted to a tuple


def test_suggestion_defaults_to_no_refs() -> None:
    answer = {key: value for key, value in VALID_ANSWER.items() if key != "supporting_refs"}

    assert ClassificationSuggestion(**answer).supporting_refs == ()


def test_suggestion_accepts_enum_members() -> None:
    suggestion = ClassificationSuggestion(
        label=ClassificationLabel.PUBLIC,
        explanation="Catalog data.",
        review_recommendation=ReviewRecommendation.OPTIONAL,
    )

    assert suggestion.label is ClassificationLabel.PUBLIC


def test_suggestion_is_immutable() -> None:
    suggestion = ClassificationSuggestion(**VALID_ANSWER)

    with pytest.raises(ValidationError):
        suggestion.label = ClassificationLabel.PUBLIC  # type: ignore[misc]


INVALID_ANSWERS = {
    "label-outside-enum": {"label": "top-secret"},
    "label-wrong-case": {"label": "PUBLIC"},
    "label-none": {"label": None},
    "recommendation-outside-enum": {"review_recommendation": "mandatory"},
    "explanation-empty": {"explanation": ""},
    "explanation-blank": {"explanation": "   "},
    "explanation-too-long": {"explanation": "x" * (MAX_EXPLANATION_LENGTH + 1)},
    "explanation-control-char": {"explanation": "bad\x07text"},
    "explanation-zero-width": {"explanation": "bad\u200btext"},
    "refs-too-many": {"supporting_refs": ["column_name"] * (MAX_REFS + 1)},
    "refs-uppercase-and-space": {"supporting_refs": ["Column Name"]},
    "refs-hyphen": {"supporting_refs": ["column-name"]},
    "refs-empty-string": {"supporting_refs": [""]},
    "refs-too-long": {"supporting_refs": ["a" * 31]},
    "refs-dangling-colon": {"supporting_refs": ["finding:"]},
    "refs-leading-colon": {"supporting_refs": [":x"]},
    "refs-two-colons": {"supporting_refs": ["a:b:c"]},
    "refs-trailing-space": {"supporting_refs": ["column_name "]},
    "refs-bare-string": {"supporting_refs": "column_name"},
}


@pytest.mark.parametrize("overrides", INVALID_ANSWERS.values(), ids=INVALID_ANSWERS.keys())
def test_invalid_suggestions_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ClassificationSuggestion(**{**VALID_ANSWER, **overrides})


@pytest.mark.parametrize("missing", ["label", "explanation", "review_recommendation"])
def test_required_suggestion_fields(missing: str) -> None:
    answer = {key: value for key, value in VALID_ANSWER.items() if key != missing}

    with pytest.raises(ValidationError):
        ClassificationSuggestion(**answer)


@pytest.mark.parametrize(
    "extra",
    [{"source": "mock"}, {"provider_name": "x"}, {"confidence": 0.9}],
    ids=["provenance-source", "provenance-name", "confidence"],
)
def test_suggestion_rejects_extra_fields(extra: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ClassificationSuggestion(**{**VALID_ANSWER, **extra})


@pytest.mark.parametrize(
    "value",
    ["text", None, 42, ["label"]],
    ids=["string", "none", "number", "list"],
)
def test_suggestion_rejects_non_mappings(value: Any) -> None:
    with pytest.raises(ValidationError):
        ClassificationSuggestion.model_validate(value)


def test_suggestion_boundary_values_are_accepted() -> None:
    suggestion = ClassificationSuggestion(
        label="unknown",
        explanation="x" * MAX_EXPLANATION_LENGTH,
        supporting_refs=["a" * 30] * MAX_REFS,
        review_recommendation="review_required",
    )

    assert len(suggestion.explanation) == MAX_EXPLANATION_LENGTH
    assert len(suggestion.supporting_refs) == MAX_REFS


def test_explanation_whitespace_is_normalized() -> None:
    suggestion = ClassificationSuggestion(**{**VALID_ANSWER, "explanation": "  one\ntwo\t three  "})

    assert suggestion.explanation == "one two three"


# --- Structural privacy ---------------------------------------------------------


def test_models_have_no_field_that_could_carry_cell_values() -> None:
    assert set(ColumnMetadata.model_fields) == {
        "column_ref",
        "column_name",
        "name_was_modified",
        "inferred_type",
        "row_count",
        "null_percent",
        "distinct_percent",
        "findings",
    }
    assert set(FindingSummary.model_fields) == {"category", "severity", "evidence"}
    assert set(ClassificationSuggestion.model_fields) == {
        "label",
        "explanation",
        "supporting_refs",
        "review_recommendation",
    }


def test_limits_are_the_documented_values() -> None:
    assert MAX_COLUMN_NAME_LENGTH == 64
    assert MAX_EXPLANATION_LENGTH == 300
    assert MAX_REFS == 8