"""Deterministic finding detectors.

Each detector is a pure function: (DataFrame, DetectionConfig) -> list[DetectedFinding].
No database access, no LLM calls, no logging of data. Detectors read raw values in memory
only to count problems; findings contain counts, percentages and rule names, never the
values that matched.

Patterns use Python's `re` module directly rather than pandas string methods, so behavior
does not depend on the installed pandas string backend.
"""

import re
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime

import pandas as pd
from pandas.api import types as pdt

from app.db.models import Severity
from app.schemas.findings import (
    DateRule,
    DetectedFinding,
    DetectionConfig,
    EvidenceValue,
    FindingCategory,
    NumericRule,
)

# Local-part@domain.tld. A top-level domain of 2+ letters is required, so "user@host" and
# "a@b.c" are not matched. Matches anywhere in the text, so emails embedded in notes count.
# The lookbehind lets a match start only at the beginning of a run of local-part characters.
# Without it, a long run with no usable "@" costs quadratic time on a single uploaded value.
_EMAIL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}", re.ASCII
)

# North American style numbers: optional +1 / 1 prefix, area code (optionally in parentheses),
# 3 digits, 4 digits, with optional space, dot or dash separators. The digit lookarounds stop
# matches inside longer digit runs (card numbers, long IDs). International formats are not
# recognized, by design.
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)", re.ASCII
)

_SEVERITY_ORDER = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}


# --- Small helpers -----------------------------------------------------------


def _iter_columns(df: pd.DataFrame) -> Iterator[tuple[str, pd.Series]]:
    """Yield (name, series) by position, so duplicate column names are handled safely."""
    for position in range(df.shape[1]):
        yield str(df.columns[position]), df.iloc[:, position]


def _percent(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def _label(name: str) -> str:
    """Shorten an untrusted column name for use in titles."""
    return name if len(name) <= 60 else name[:57] + "..."


def _build(
    *,
    category: FindingCategory,
    severity: Severity,
    column_name: str | None,
    title: str,
    description: str,
    evidence: dict[str, EvidenceValue],
    method: str,
    recommendation: str,
) -> DetectedFinding:
    return DetectedFinding(
        category=category,
        severity=severity,
        column_name=None if column_name is None else column_name[:255],
        title=title,
        description=description,
        safe_evidence=evidence,
        detection_method=method,
        recommendation=recommendation,
    )


def _two_level_severity(percent: float) -> Severity:
    """MEDIUM normally, HIGH when 10% or more of checked values are affected."""
    return Severity.HIGH if percent >= 10.0 else Severity.MEDIUM


# --- Missing values ----------------------------------------------------------


def detect_missing_values(df: pd.DataFrame, config: DetectionConfig) -> list[DetectedFinding]:
    """Report columns whose null percentage is at or above their threshold.

    "Null" means what pandas treats as missing (NaN, None, NaT, pd.NA). Empty or
    whitespace-only strings are not counted. Severity bands: LOW below 25%, MEDIUM from 25%,
    HIGH from 50%.
    """
    row_count = len(df)
    if row_count == 0:
        return []

    exempt = set(config.missing_value_exempt_columns)
    findings: list[DetectedFinding] = []
    for name, series in _iter_columns(df):
        if name in exempt:
            continue
        null_count = int(series.isna().sum())
        if null_count == 0:
            continue
        threshold = config.missing_value_overrides.get(
            name, config.missing_value_threshold_percent
        )
        # Compare the unrounded percentage so the boundary is exact.
        if 100.0 * null_count / row_count < threshold:
            continue

        null_percent = _percent(null_count, row_count)
        if null_percent >= 50.0:
            severity = Severity.HIGH
        elif null_percent >= 25.0:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        findings.append(
            _build(
                category=FindingCategory.MISSING_VALUES,
                severity=severity,
                column_name=name,
                title=f"Missing values in '{_label(name)}'",
                description=(
                    f"{null_count} of {row_count} values ({null_percent}%) are missing, "
                    f"at or above the {float(threshold)}% threshold."
                ),
                evidence={
                    "row_count": row_count,
                    "null_count": null_count,
                    "null_percent": null_percent,
                    "threshold_percent": float(threshold),
                },
                method="null_percent_threshold",
                recommendation=(
                    "Confirm whether this column may be empty. If not, correct the source "
                    "data or add validation; if it may, exempt the column from this rule."
                ),
            )
        )
    return findings


# --- Duplicate rows ----------------------------------------------------------


def detect_duplicate_rows(df: pd.DataFrame, config: DetectionConfig) -> list[DetectedFinding]:
    """Report rows that repeat an earlier row (the first occurrence is not counted).

    Compares entire rows, or only config.duplicate_subset if given. If any subset column is
    missing from the DataFrame, the rule is skipped. Missing values compare as equal.
    """
    row_count = len(df)
    if row_count == 0 or df.shape[1] == 0:
        return []

    subset = config.duplicate_subset
    if subset is None:
        duplicated = df.duplicated(keep="first")
        scope = "all_columns"
    else:
        if any(column not in df.columns for column in subset):
            return []
        duplicated = df.duplicated(subset=subset, keep="first")
        scope = "selected_columns"

    duplicate_count = int(duplicated.sum())
    if duplicate_count == 0:
        return []

    duplicate_percent = _percent(duplicate_count, row_count)
    return [
        _build(
            category=FindingCategory.DUPLICATE_ROWS,
            severity=_two_level_severity(duplicate_percent),
            column_name=None,
            title="Duplicate rows detected",
            description=(
                f"{duplicate_count} of {row_count} rows ({duplicate_percent}%) repeat an "
                f"earlier row when comparing {scope.replace('_', ' ')}."
            ),
            evidence={
                "row_count": row_count,
                "duplicate_row_count": duplicate_count,
                "duplicate_percent": duplicate_percent,
                "comparison_scope": scope,
            },
            method="pandas_duplicated",
            recommendation=(
                "Review the repeated rows and remove or consolidate them at the source "
                "if they are not intentional."
            ),
        )
    ]


# --- Email and phone patterns ------------------------------------------------


def _detect_text_pattern(
    df: pd.DataFrame,
    config: DetectionConfig,
    *,
    pattern: re.Pattern[str],
    category: FindingCategory,
    family: str,
    plural: str,
    noun: str,
    method: str,
) -> list[DetectedFinding]:
    """Count values containing the pattern in each text column. Matched text is never kept.

    Numeric, boolean and datetime columns are skipped. Severity is HIGH when at least half of
    the non-null values match (a dedicated contact column) and MEDIUM otherwise (for example
    an occasional value inside free text).
    """
    selected = None if config.pattern_columns is None else set(config.pattern_columns)
    findings: list[DetectedFinding] = []

    for name, series in _iter_columns(df):
        if selected is not None and name not in selected:
            continue
        if pdt.is_numeric_dtype(series) or pdt.is_datetime64_any_dtype(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue

        match_count = sum(1 for text in non_null.astype(str) if pattern.search(text))
        if match_count < config.pattern_min_match_count:
            continue

        non_null_count = len(non_null)
        match_percent = _percent(match_count, non_null_count)
        severity = Severity.HIGH if 2 * match_count >= non_null_count else Severity.MEDIUM

        findings.append(
            _build(
                category=category,
                severity=severity,
                column_name=name,
                title=f"Potential {plural} in '{_label(name)}'",
                description=(
                    f"{match_count} of {non_null_count} non-empty values ({match_percent}%) "
                    f"contain {noun}. Matched values are not stored."
                ),
                evidence={
                    "non_null_count": non_null_count,
                    "match_count": match_count,
                    "match_percent": match_percent,
                    "pattern_family": family,
                },
                method=method,
                recommendation=(
                    "Review whether this column holds personal contact data and, if so, "
                    "classify it accordingly (for example as confidential)."
                ),
            )
        )
    return findings


def detect_potential_emails(df: pd.DataFrame, config: DetectionConfig) -> list[DetectedFinding]:
    return _detect_text_pattern(
        df,
        config,
        pattern=_EMAIL_PATTERN,
        category=FindingCategory.POTENTIAL_EMAIL,
        family="email",
        plural="email addresses",
        noun="an email-like pattern",
        method="regex_email_pattern",
    )


def detect_potential_phones(df: pd.DataFrame, config: DetectionConfig) -> list[DetectedFinding]:
    return _detect_text_pattern(
        df,
        config,
        pattern=_PHONE_PATTERN,
        category=FindingCategory.POTENTIAL_PHONE,
        family="phone",
        plural="phone numbers",
        noun="a phone-number-like pattern",
        method="regex_phone_pattern",
    )


# --- Numeric business rules --------------------------------------------------


def _describe_bounds(rule: NumericRule) -> str:
    if rule.min_value is not None and rule.max_value is not None:
        return f"between {rule.min_value:g} and {rule.max_value:g}"
    if rule.min_value is not None:
        return f"at least {rule.min_value:g}"
    return f"at most {rule.max_value:g}"


def detect_numeric_rule_violations(
    df: pd.DataFrame, config: DetectionConfig
) -> list[DetectedFinding]:
    """Check only the columns named in config.numeric_rules (bounds are inclusive).

    Null values are not violations (the missing-value rule covers them). Non-null values that
    cannot be read as numbers, such as the text "TBD", count as violations.
    """
    findings: list[DetectedFinding] = []
    for name, series in _iter_columns(df):
        rule = config.numeric_rules.get(name)
        if rule is None:
            continue
        non_null = series.dropna()
        checked_count = len(non_null)
        if checked_count == 0:
            continue

        numeric = pd.to_numeric(non_null, errors="coerce")
        non_numeric_count = int(numeric.isna().sum())
        valid = numeric.dropna()

        evidence: dict[str, EvidenceValue] = {"checked_count": checked_count}
        violation_count = non_numeric_count
        below_count = 0
        above_count = 0
        if rule.min_value is not None:
            below_count = int((valid < rule.min_value).sum())
            violation_count += below_count
        if rule.max_value is not None:
            above_count = int((valid > rule.max_value).sum())
            violation_count += above_count
        if violation_count == 0:
            continue

        violation_percent = _percent(violation_count, checked_count)
        evidence["violation_count"] = violation_count
        evidence["violation_percent"] = violation_percent
        evidence["non_numeric_count"] = non_numeric_count
        if rule.min_value is not None:
            evidence["min_allowed"] = float(rule.min_value)
            evidence["below_min_count"] = below_count
        if rule.max_value is not None:
            evidence["max_allowed"] = float(rule.max_value)
            evidence["above_max_count"] = above_count

        findings.append(
            _build(
                category=FindingCategory.NUMERIC_RULE_VIOLATION,
                severity=_two_level_severity(violation_percent),
                column_name=name,
                title=f"Values outside the allowed range in '{_label(name)}'",
                description=(
                    f"{violation_count} of {checked_count} values ({violation_percent}%) break "
                    f"the configured rule (values must be {_describe_bounds(rule)} and numeric)."
                ),
                evidence=evidence,
                method="configured_numeric_bounds",
                recommendation=(
                    "Check these records against the business rule and correct the source "
                    "data. If the rule does not apply to this column, remove it from the "
                    "configuration."
                ),
            )
        )
    return findings


# --- Date validity -----------------------------------------------------------


def _matches_format(text: str, date_format: str) -> bool:
    try:
        datetime.strptime(text, date_format)
    except ValueError:
        return False
    return True


def detect_invalid_dates(df: pd.DataFrame, config: DetectionConfig) -> list[DetectedFinding]:
    """Check only the columns named in config.date_rules against their expected format.

    A value is invalid when it does not parse under the format, which covers both the wrong
    format ("01/09/2026" for "%Y-%m-%d") and impossible dates ("2026-02-30"). The two cases
    are not distinguished. Surrounding whitespace is ignored. Columns already stored as
    datetimes are skipped, and nulls are not counted.
    """
    findings: list[DetectedFinding] = []
    for name, series in _iter_columns(df):
        rule: DateRule | None = config.date_rules.get(name)
        if rule is None or pdt.is_datetime64_any_dtype(series):
            continue
        non_null = series.dropna()
        checked_count = len(non_null)
        if checked_count == 0:
            continue

        texts = non_null.astype(str).str.strip()
        invalid_count = sum(
            int(count)
            for value, count in texts.value_counts().items()
            if not _matches_format(str(value), rule.date_format)
        )
        if invalid_count == 0:
            continue

        invalid_percent = _percent(invalid_count, checked_count)
        findings.append(
            _build(
                category=FindingCategory.INVALID_DATE,
                severity=_two_level_severity(invalid_percent),
                column_name=name,
                title=f"Invalid dates in '{_label(name)}'",
                description=(
                    f"{invalid_count} of {checked_count} values ({invalid_percent}%) do not "
                    f"match the expected date format."
                ),
                evidence={
                    "checked_count": checked_count,
                    "invalid_count": invalid_count,
                    "invalid_percent": invalid_percent,
                    "expected_format": rule.date_format,
                },
                method="strptime_format_check",
                recommendation=(
                    "Correct or re-export the malformed dates, or update the expected "
                    "format if the column legitimately uses a different one."
                ),
            )
        )
    return findings


# --- Combining results -------------------------------------------------------

_DETECTORS: tuple[Callable[[pd.DataFrame, DetectionConfig], list[DetectedFinding]], ...] = (
    detect_missing_values,
    detect_duplicate_rows,
    detect_potential_emails,
    detect_potential_phones,
    detect_numeric_rule_violations,
    detect_invalid_dates,
)


def dedupe_findings(findings: Iterable[DetectedFinding]) -> list[DetectedFinding]:
    """Keep the first finding for each (category, column) pair, preserving order."""
    seen: set[tuple[str, str | None]] = set()
    unique: list[DetectedFinding] = []
    for finding in findings:
        if finding.dedup_key in seen:
            continue
        seen.add(finding.dedup_key)
        unique.append(finding)
    return unique


def run_detectors(
    df: pd.DataFrame, config: DetectionConfig | None = None
) -> list[DetectedFinding]:
    """Run every detector, remove duplicate findings and return a deterministic ordering.

    Order: severity (HIGH first), then category name, then column name. An exception raised
    by a detector is not caught here; the scan service (a later phase) handles failures.
    """
    active_config = config or DetectionConfig()
    combined = [
        finding for detector in _DETECTORS for finding in detector(df, active_config)
    ]
    return sorted(
        dedupe_findings(combined),
        key=lambda f: (_SEVERITY_ORDER[f.severity], f.category.value, f.column_name or ""),
    )