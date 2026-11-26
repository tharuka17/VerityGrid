from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from app.db.models import ReviewStatus, Severity
from app.schemas.findings import (
    DateRule,
    DetectedFinding,
    DetectionConfig,
    FindingCategory,
    NumericRule,
)
from app.services.detectors import (
    dedupe_findings,
    detect_duplicate_rows,
    detect_invalid_dates,
    detect_missing_values,
    detect_numeric_rule_violations,
    detect_potential_emails,
    detect_potential_phones,
    run_detectors,
)

DEFAULT = DetectionConfig()


def make_finding(**overrides: Any) -> DetectedFinding:
    data: dict[str, Any] = {
        "category": FindingCategory.MISSING_VALUES,
        "severity": Severity.LOW,
        "column_name": "a",
        "title": "title",
        "description": "description",
        "safe_evidence": {"count": 1},
        "detection_method": "method",
        "recommendation": "next step",
    }
    data.update(overrides)
    return DetectedFinding(**data)


def frame_with_nulls(null_count: int, rows: int = 10) -> pd.DataFrame:
    values: list[float | None] = [None] * null_count + [float(i) for i in range(rows - null_count)]
    return pd.DataFrame({"a": values})


def only(findings: list[DetectedFinding]) -> DetectedFinding:
    assert len(findings) == 1
    return findings[0]


def mixed_issue_frame() -> pd.DataFrame:
    """One frame that triggers exactly one finding from five different rules."""
    return pd.DataFrame(
        {
            "sku": ["A", "B", "C", "A"],
            "qty": [5, -2, 7, 5],
            "contact": ["a@example.com", "b@example.com", "c@example.com", "a@example.com"],
            "notes": [None, None, None, None],
            "counted": ["2026-09-01", "2026-02-30", "2026-09-03", "2026-09-01"],
        }
    )


MIXED_CONFIG = DetectionConfig(
    numeric_rules={"qty": NumericRule(min_value=0)},
    date_rules={"counted": DateRule()},
)


# --- Schemas ---------------------------------------------------------------


def test_finding_defaults_to_pending_review() -> None:
    assert make_finding().review_status == ReviewStatus.PENDING


def test_column_name_required_for_column_level_categories() -> None:
    with pytest.raises(ValidationError):
        make_finding(category=FindingCategory.MISSING_VALUES, column_name=None)


def test_duplicate_rows_finding_needs_no_column() -> None:
    finding = make_finding(category=FindingCategory.DUPLICATE_ROWS, column_name=None)

    assert finding.column_name is None


def test_safe_evidence_rejects_non_scalar_values() -> None:
    with pytest.raises(ValidationError):
        make_finding(safe_evidence={"values": [1, 2, 3]})


def test_safe_evidence_rejects_long_text() -> None:
    with pytest.raises(ValidationError):
        make_finding(safe_evidence={"note": "x" * 121})


def test_safe_evidence_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        make_finding(safe_evidence={})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_finding(raw_values=["secret"])


def test_findings_are_immutable() -> None:
    finding = make_finding()

    with pytest.raises(ValidationError):
        finding.title = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"min_value": 5.0, "max_value": 1.0}],
    ids=["no-bounds", "min-above-max"],
)
def test_numeric_rule_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        NumericRule(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"missing_value_threshold_percent": -1.0},
        {"missing_value_threshold_percent": 100.5},
        {"missing_value_overrides": {"a": 150.0}},
        {"duplicate_subset": []},
        {"pattern_min_match_count": 0},
    ],
    ids=["negative-threshold", "threshold-over-100", "override-over-100", "empty-subset", "zero-min-matches"],
)
def test_detection_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DetectionConfig(**kwargs)


# --- Missing values --------------------------------------------------------


def test_missing_values_below_threshold_not_flagged() -> None:
    assert detect_missing_values(frame_with_nulls(1), DEFAULT) == []  # 10% < 20%


def test_missing_values_evidence_at_threshold() -> None:
    finding = only(detect_missing_values(frame_with_nulls(2), DEFAULT))  # exactly 20%

    assert finding.category == FindingCategory.MISSING_VALUES
    assert finding.column_name == "a"
    assert finding.severity == Severity.LOW
    assert finding.detection_method == "null_percent_threshold"
    assert finding.safe_evidence == {
        "row_count": 10,
        "null_count": 2,
        "null_percent": 20.0,
        "threshold_percent": 20.0,
    }


@pytest.mark.parametrize(
    ("null_count", "expected"),
    [
        (2, Severity.LOW),
        (3, Severity.MEDIUM),
        (4, Severity.MEDIUM),
        (5, Severity.HIGH),
        (10, Severity.HIGH),
    ],
)
def test_missing_value_severity_bands(null_count: int, expected: Severity) -> None:
    finding = only(detect_missing_values(frame_with_nulls(null_count), DEFAULT))

    assert finding.severity == expected


def test_missing_value_threshold_is_configurable() -> None:
    config = DetectionConfig(missing_value_threshold_percent=5.0)

    assert only(detect_missing_values(frame_with_nulls(1), config)).column_name == "a"


def test_missing_value_per_column_override() -> None:
    df = pd.DataFrame({"a": frame_with_nulls(1)["a"], "b": frame_with_nulls(1)["a"]})
    config = DetectionConfig(
        missing_value_threshold_percent=50.0, missing_value_overrides={"a": 10.0}
    )

    assert [f.column_name for f in detect_missing_values(df, config)] == ["a"]


def test_missing_value_threshold_uses_unrounded_percentage() -> None:
    df = pd.DataFrame({"a": [None, 1.0, 2.0]})  # 33.333...% missing

    flagged = detect_missing_values(df, DetectionConfig(missing_value_threshold_percent=33.33))
    not_flagged = detect_missing_values(df, DetectionConfig(missing_value_threshold_percent=33.34))

    assert len(flagged) == 1
    assert not_flagged == []


def test_exempt_columns_are_not_checked() -> None:
    config = DetectionConfig(missing_value_exempt_columns=["a"])

    assert detect_missing_values(frame_with_nulls(5), config) == []


def test_blank_strings_are_not_missing() -> None:
    df = pd.DataFrame({"a": ["", "", "x"]})

    assert detect_missing_values(df, DEFAULT) == []


def test_missing_values_on_empty_frame() -> None:
    assert detect_missing_values(pd.DataFrame(), DEFAULT) == []


# --- Duplicate rows --------------------------------------------------------


def test_duplicate_rows_detected() -> None:
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})

    finding = only(detect_duplicate_rows(df, DEFAULT))

    assert finding.category == FindingCategory.DUPLICATE_ROWS
    assert finding.column_name is None
    assert finding.severity == Severity.HIGH  # 25% is at least 10%
    assert finding.detection_method == "pandas_duplicated"
    assert finding.review_status == ReviewStatus.PENDING
    assert finding.safe_evidence == {
        "row_count": 4,
        "duplicate_row_count": 1,
        "duplicate_percent": 25.0,
        "comparison_scope": "all_columns",
    }


def test_no_duplicates_no_finding() -> None:
    assert detect_duplicate_rows(pd.DataFrame({"a": [1, 2, 3]}), DEFAULT) == []


def test_duplicate_severity_is_medium_when_few() -> None:
    df = pd.DataFrame({"a": list(range(19)) + [0]})  # 1 repeat in 20 rows = 5%

    assert only(detect_duplicate_rows(df, DEFAULT)).severity == Severity.MEDIUM


def test_duplicate_key_subset() -> None:
    df = pd.DataFrame({"id": [1, 1, 2], "qty": [5, 6, 7]})

    assert detect_duplicate_rows(df, DEFAULT) == []  # whole rows differ
    finding = only(detect_duplicate_rows(df, DetectionConfig(duplicate_subset=["id"])))
    assert finding.safe_evidence["duplicate_row_count"] == 1
    assert finding.safe_evidence["duplicate_percent"] == pytest.approx(33.33)
    assert finding.safe_evidence["comparison_scope"] == "selected_columns"


def test_duplicate_subset_with_unknown_column_is_skipped() -> None:
    df = pd.DataFrame({"id": [1, 1]})

    assert detect_duplicate_rows(df, DetectionConfig(duplicate_subset=["id", "nope"])) == []


def test_nan_values_compare_equal_for_duplicates() -> None:
    df = pd.DataFrame({"a": [float("nan"), float("nan")]})

    assert only(detect_duplicate_rows(df, DEFAULT)).safe_evidence["duplicate_row_count"] == 1


def test_duplicates_on_empty_frame() -> None:
    assert detect_duplicate_rows(pd.DataFrame(), DEFAULT) == []


# --- Email patterns --------------------------------------------------------


def test_email_column_is_flagged() -> None:
    df = pd.DataFrame({"contact": ["a@example.com", "b@example.org", "c@example.net", None]})

    finding = only(detect_potential_emails(df, DEFAULT))

    assert finding.category == FindingCategory.POTENTIAL_EMAIL
    assert finding.column_name == "contact"
    assert finding.severity == Severity.HIGH
    assert finding.detection_method == "regex_email_pattern"
    assert finding.safe_evidence == {
        "non_null_count": 3,
        "match_count": 3,
        "match_percent": 100.0,
        "pattern_family": "email",
    }


def test_email_embedded_in_free_text() -> None:
    df = pd.DataFrame(
        {"notes": ["Leave at back door", "Billing: ap@example.com", "", "Call dock", None]}
    )

    finding = only(detect_potential_emails(df, DEFAULT))

    assert finding.severity == Severity.MEDIUM
    assert finding.safe_evidence["non_null_count"] == 4
    assert finding.safe_evidence["match_count"] == 1
    assert finding.safe_evidence["match_percent"] == 25.0


@pytest.mark.parametrize(
    "value",
    [
        "x@example.com",
        "first.last+tag@sub.example.co.uk",
        "UPPER@EXAMPLE.COM",
        "see ap@example.org for billing",
    ],
)
def test_email_positive_variants(value: str) -> None:
    assert len(detect_potential_emails(pd.DataFrame({"c": [value]}), DEFAULT)) == 1


@pytest.mark.parametrize(
    "value",
    ["user@", "@example.com", "user@example", "a@b.c", "plain text", "12345", "2026-09-01"],
)
def test_email_false_positive_boundaries(value: str) -> None:
    assert detect_potential_emails(pd.DataFrame({"c": [value]}), DEFAULT) == []


def test_numeric_columns_are_not_scanned_for_patterns() -> None:
    df = pd.DataFrame({"n": [1, 2, 3]})

    assert detect_potential_emails(df, DEFAULT) == []
    assert detect_potential_phones(df, DEFAULT) == []


def test_pattern_columns_restrict_scanning() -> None:
    df = pd.DataFrame({"email_a": ["a@example.com"], "email_b": ["b@example.com"]})

    findings = detect_potential_emails(df, DetectionConfig(pattern_columns=["email_a"]))

    assert [f.column_name for f in findings] == ["email_a"]


def test_pattern_min_match_count() -> None:
    df = pd.DataFrame({"notes": ["Billing: ap@example.com", "nothing here"]})

    assert len(detect_potential_emails(df, DetectionConfig(pattern_min_match_count=1))) == 1
    assert detect_potential_emails(df, DetectionConfig(pattern_min_match_count=2)) == []


def test_pattern_severity_boundary_is_at_fifty_percent() -> None:
    half = pd.DataFrame({"c": ["a@example.com", "plain"]})
    third = pd.DataFrame({"c": ["a@example.com", "plain", "other"]})

    assert only(detect_potential_emails(half, DEFAULT)).severity == Severity.HIGH
    assert only(detect_potential_emails(third, DEFAULT)).severity == Severity.MEDIUM


# --- Phone patterns --------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "503-555-0101",
        "(503) 555-0102",
        "+1 206 555 0103",
        "2065550105",
        "503.555.0106",
        "1-503-555-0107",
    ],
)
def test_phone_formats_are_detected(value: str) -> None:
    finding = only(detect_potential_phones(pd.DataFrame({"phone": [value]}), DEFAULT))

    assert finding.category == FindingCategory.POTENTIAL_PHONE
    assert finding.detection_method == "regex_phone_pattern"
    assert finding.safe_evidence["match_count"] == 1
    assert finding.safe_evidence["pattern_family"] == "phone"


@pytest.mark.parametrize(
    "value",
    [
        "2026-09-01",
        "12345",
        "C001",
        "1842.50",
        "123456789",  # 9 digits
        "50355501011",  # 11 digits not starting with 1
        "+44 20 7946 0958",  # international format: not recognized by design
        "4111111111111111",  # 16-digit run
    ],
)
def test_phone_false_positive_boundaries(value: str) -> None:
    assert detect_potential_phones(pd.DataFrame({"c": [value]}), DEFAULT) == []


def test_phone_embedded_in_free_text() -> None:
    df = pd.DataFrame(
        {"notes": ["Call 503-555-0143 on arrival", "Leave at back door", "Dock hours 6am-2pm", "No notes"]}
    )

    finding = only(detect_potential_phones(df, DEFAULT))

    assert finding.severity == Severity.MEDIUM
    assert finding.safe_evidence["match_count"] == 1
    assert finding.safe_evidence["match_percent"] == 25.0


def test_phone_numbers_stored_as_numbers_are_not_scanned() -> None:
    """Known limitation: numeric columns are skipped, so digit-only phones parsed as ints are missed."""
    df = pd.DataFrame({"phone": [2065550105, 2065550106]})

    assert detect_potential_phones(df, DEFAULT) == []


# --- Numeric rules ---------------------------------------------------------


def test_negative_values_flagged_only_for_designated_columns() -> None:
    df = pd.DataFrame({"quantity_on_hand": [10, -12, 5], "adjustment": [-5, 3, -1]})
    config = DetectionConfig(numeric_rules={"quantity_on_hand": NumericRule(min_value=0)})

    finding = only(detect_numeric_rule_violations(df, config))

    assert finding.category == FindingCategory.NUMERIC_RULE_VIOLATION
    assert finding.column_name == "quantity_on_hand"  # "adjustment" has no rule, so it is not checked
    assert finding.severity == Severity.HIGH
    assert finding.detection_method == "configured_numeric_bounds"
    assert finding.safe_evidence["checked_count"] == 3
    assert finding.safe_evidence["violation_count"] == 1
    assert finding.safe_evidence["violation_percent"] == pytest.approx(33.33)
    assert finding.safe_evidence["non_numeric_count"] == 0
    assert finding.safe_evidence["min_allowed"] == 0.0
    assert finding.safe_evidence["below_min_count"] == 1
    assert "max_allowed" not in finding.safe_evidence


def test_numeric_bounds_are_inclusive() -> None:
    at_min = pd.DataFrame({"q": [0, 1, 2]})
    at_max = pd.DataFrame({"q": [99, 100]})

    assert detect_numeric_rule_violations(at_min, DetectionConfig(numeric_rules={"q": NumericRule(min_value=0)})) == []
    assert detect_numeric_rule_violations(at_max, DetectionConfig(numeric_rules={"q": NumericRule(max_value=100)})) == []


def test_numeric_min_and_max() -> None:
    df = pd.DataFrame({"q": [-1, 50, 101]})
    config = DetectionConfig(numeric_rules={"q": NumericRule(min_value=0, max_value=100)})

    finding = only(detect_numeric_rule_violations(df, config))

    assert finding.safe_evidence["below_min_count"] == 1
    assert finding.safe_evidence["above_max_count"] == 1
    assert finding.safe_evidence["violation_count"] == 2
    assert finding.safe_evidence["violation_percent"] == pytest.approx(66.67)


def test_non_numeric_text_in_numeric_column() -> None:
    df = pd.DataFrame({"reorder_point": ["40", "TBD", "30"]})
    config = DetectionConfig(numeric_rules={"reorder_point": NumericRule(min_value=0)})

    finding = only(detect_numeric_rule_violations(df, config))

    assert finding.safe_evidence["checked_count"] == 3
    assert finding.safe_evidence["non_numeric_count"] == 1
    assert finding.safe_evidence["below_min_count"] == 0
    assert finding.safe_evidence["violation_count"] == 1


def test_nulls_are_not_numeric_violations() -> None:
    df = pd.DataFrame({"q": [None, 5.0, 6.0]})
    config = DetectionConfig(numeric_rules={"q": NumericRule(min_value=0)})

    assert detect_numeric_rule_violations(df, config) == []


def test_numeric_rule_for_missing_column_is_ignored() -> None:
    config = DetectionConfig(numeric_rules={"absent": NumericRule(min_value=0)})

    assert detect_numeric_rule_violations(pd.DataFrame({"q": [-1]}), config) == []


def test_numeric_severity_is_medium_when_few() -> None:
    df = pd.DataFrame({"q": [-1] + list(range(19))})  # 1 of 20 = 5%
    config = DetectionConfig(numeric_rules={"q": NumericRule(min_value=0)})

    assert only(detect_numeric_rule_violations(df, config)).severity == Severity.MEDIUM


# --- Date validity ---------------------------------------------------------


def test_invalid_dates_with_default_format() -> None:
    df = pd.DataFrame(
        {"d": ["2026-09-01", "2026-02-30", "2023-13-45", "01/09/2026", "not a date", None]}
    )
    config = DetectionConfig(date_rules={"d": DateRule()})

    finding = only(detect_invalid_dates(df, config))

    assert finding.category == FindingCategory.INVALID_DATE
    assert finding.column_name == "d"
    assert finding.severity == Severity.HIGH
    assert finding.detection_method == "strptime_format_check"
    assert finding.safe_evidence == {
        "checked_count": 5,
        "invalid_count": 4,
        "invalid_percent": 80.0,
        "expected_format": "%Y-%m-%d",
    }


def test_valid_dates_produce_no_finding() -> None:
    df = pd.DataFrame({"d": [" 2026-09-01 ", "2026-12-31", "2024-02-29"]})  # padded value and a leap day
    config = DetectionConfig(date_rules={"d": DateRule()})

    assert detect_invalid_dates(df, config) == []


def test_custom_date_format() -> None:
    df = pd.DataFrame({"d": ["01/09/2026", "31/02/2026"]})
    config = DetectionConfig(date_rules={"d": DateRule(date_format="%d/%m/%Y")})

    finding = only(detect_invalid_dates(df, config))

    assert finding.safe_evidence["invalid_count"] == 1
    assert finding.safe_evidence["expected_format"] == "%d/%m/%Y"


def test_datetime_dtype_column_is_skipped() -> None:
    df = pd.DataFrame({"d": pd.to_datetime(["2026-09-01", "2026-09-02"])})
    config = DetectionConfig(date_rules={"d": DateRule()})

    assert detect_invalid_dates(df, config) == []


def test_dates_not_checked_without_rule() -> None:
    df = pd.DataFrame({"d": ["not a date", "2026-02-30"]})

    assert detect_invalid_dates(df, DEFAULT) == []


# --- Combined runner -------------------------------------------------------


def test_run_detectors_combines_and_orders() -> None:
    findings = run_detectors(mixed_issue_frame(), MIXED_CONFIG)

    assert len(findings) == 5
    assert {f.severity for f in findings} == {Severity.HIGH}
    assert [f.category.value for f in findings] == [
        "duplicate_rows",
        "invalid_date",
        "missing_values",
        "numeric_rule_violation",
        "potential_email",
    ]


def test_run_detectors_is_deterministic() -> None:
    first = run_detectors(mixed_issue_frame(), MIXED_CONFIG)
    second = run_detectors(mixed_issue_frame(), MIXED_CONFIG)

    assert first == second


def test_dedupe_keeps_first_finding_per_key() -> None:
    first = make_finding(title="first")
    repeat = make_finding(title="second")  # same category and column
    other_column = make_finding(column_name="b", title="third")

    result = dedupe_findings([first, repeat, other_column])

    assert [f.title for f in result] == ["first", "third"]


def test_run_detectors_on_empty_frame() -> None:
    assert run_detectors(pd.DataFrame()) == []


def test_detector_output_contains_no_raw_values() -> None:
    df = pd.DataFrame(
        {
            "email": ["jane.doe@example.com", "SECRET-NOTE-XYZ", "bob@example.net"],
            "phone": ["503-555-0199", "n/a", "503-555-0198"],
            "qty": [-987654, 5, 6],
            "ship_date": ["2026-02-30", "2026-09-01", "TBD-SENTINEL"],
        }
    )
    config = DetectionConfig(
        numeric_rules={"qty": NumericRule(min_value=0)},
        date_rules={"ship_date": DateRule()},
    )

    findings = run_detectors(df, config)
    rendered = " ".join(repr(f) + f.model_dump_json() for f in findings)

    assert {f.category for f in findings} == {
        FindingCategory.POTENTIAL_EMAIL,
        FindingCategory.POTENTIAL_PHONE,
        FindingCategory.NUMERIC_RULE_VIOLATION,
        FindingCategory.INVALID_DATE,
    }
    for raw_value in (
        "jane.doe",
        "bob@",
        "503-555-0199",
        "503-555-0198",
        "SECRET-NOTE-XYZ",
        "987654",
        "TBD-SENTINEL",
        "2026-02-30",
    ):
        assert raw_value not in rendered
    assert "@" not in rendered


def test_detector_findings_are_pending_with_scalar_evidence() -> None:
    for finding in run_detectors(mixed_issue_frame(), MIXED_CONFIG):
        assert finding.review_status == ReviewStatus.PENDING
        assert all(isinstance(v, (int, float, bool, str)) for v in finding.safe_evidence.values())