"""Detection rules supplied with an upload (the optional `rules` form field of POST /scans).

The text is untrusted. It is size-limited, parsed with the standard JSON parser (NaN and Infinity are
rejected), validated by DetectionConfig and then by the extra checks below: entry caps, finite numeric bounds,
printable column names of bounded length and a strict allowlist for date formats. Error locations and messages
never contain the submitted text: unknown keys and column names are replaced by placeholders, and messages hold
rule text and counts only. The rules apply to one scan and are not stored.

This module has no web-framework, database or provider imports; the route converts InvalidRulesError into the
API's shared 422 response.
"""

import json
import math
from collections.abc import Iterable
from typing import Any, NamedTuple

from pydantic import ValidationError

from app.schemas.findings import DetectionConfig

MAX_RULES_CHARS = 20_000
MAX_COLUMN_NAME_LENGTH = 255
MAX_DATE_FORMAT_LENGTH = 40
MAX_ISSUES = 20
MAX_ENTRIES = {
    "numeric_rules": 20,
    "date_rules": 20,
    "missing_value_overrides": 50,
    "missing_value_exempt_columns": 50,
    "duplicate_subset": 50,
    "pattern_columns": 50,
}

ALLOWED_DIRECTIVES = frozenset("YmdyHMSbB")
ALLOWED_SEPARATORS = frozenset("-/.: T,")

_DICT_FIELDS = ("numeric_rules", "date_rules", "missing_value_overrides")
_LIST_FIELDS = ("missing_value_exempt_columns", "duplicate_subset", "pattern_columns")
_KNOWN_NAMES = frozenset(DetectionConfig.model_fields) | {"min_value", "max_value", "date_format"}
_NAME_MESSAGE = (
    f"Column names must be 1 to {MAX_COLUMN_NAME_LENGTH} printable characters."
)


class RuleIssue(NamedTuple):
    location: tuple[str, ...]  # relative to the rules field, for example ("numeric_rules", "<column>")
    message: str


class InvalidRulesError(Exception):
    """The rules were rejected. `issues` is safe to return to a client."""

    def __init__(self, issues: list[RuleIssue]) -> None:
        super().__init__("The detection rules are invalid.")
        self.issues = issues[:MAX_ISSUES]


def _reject_constant(name: str) -> Any:
    raise ValueError("NaN and Infinity are not valid JSON")


def _safe_location(loc: tuple[Any, ...]) -> tuple[str, ...]:
    """Make a pydantic error location safe: no submitted key or column name is ever included."""
    parts: list[str] = []
    for index, segment in enumerate(loc):
        if isinstance(segment, int):
            parts.append(str(segment))
        elif index == 1 and loc[0] in _DICT_FIELDS:
            parts.append("<column>")
        elif isinstance(segment, str) and segment in _KNOWN_NAMES:
            parts.append(segment)
        else:
            parts.append("<unknown field>")
    return tuple(parts)


def _pydantic_issues(error: ValidationError) -> list[RuleIssue]:
    issues: list[RuleIssue] = []
    for item in error.errors():
        issue = RuleIssue(
            _safe_location(tuple(item.get("loc", ()))),
            str(item.get("msg", "Invalid value"))[:200],
        )
        if issue not in issues:
            issues.append(issue)
    return issues


def _bad_name(name: str) -> bool:
    return not name or len(name) > MAX_COLUMN_NAME_LENGTH or not name.isprintable()


def _date_format_problem(date_format: str) -> str | None:
    """None if the format is acceptable, otherwise a safe explanation."""
    if len(date_format) > MAX_DATE_FORMAT_LENGTH:
        return f"The date format must be at most {MAX_DATE_FORMAT_LENGTH} characters."
    directives: list[str] = []
    position = 0
    while position < len(date_format):
        char = date_format[position]
        if char == "%":
            directive = date_format[position + 1 : position + 2]
            if not directive or directive not in ALLOWED_DIRECTIVES:
                return "The date format may only use these directives: %Y %m %d %y %H %M %S %b %B."
            directives.append(directive)
            position += 2
        elif char in ALLOWED_SEPARATORS:
            position += 1
        else:
            return "The date format may only contain directives and the separators - / . : , space and T."
    if not directives:
        return "The date format needs at least one directive such as %Y."
    if len(set(directives)) != len(directives):
        return "A directive may appear only once in the date format."
    return None


def _check_sizes(raw: dict[str, Any]) -> None:
    issues = [
        RuleIssue((field,), f"At most {limit} entries are allowed.")
        for field, limit in MAX_ENTRIES.items()
        if isinstance(raw.get(field), (dict, list)) and len(raw[field]) > limit
    ]
    if issues:
        raise InvalidRulesError(issues)


def _check_values(config: DetectionConfig) -> None:
    issues: list[RuleIssue] = []

    def add(location: tuple[str, ...], message: str) -> None:
        issue = RuleIssue(location, message)
        if issue not in issues:
            issues.append(issue)

    for field in _DICT_FIELDS:
        if any(_bad_name(name) for name in getattr(config, field)):
            add((field, "<column>"), _NAME_MESSAGE)
    for field in _LIST_FIELDS:
        if any(_bad_name(name) for name in (getattr(config, field) or [])):
            add((field,), _NAME_MESSAGE)
    for rule in config.numeric_rules.values():
        for bound in (rule.min_value, rule.max_value):
            if bound is not None and not math.isfinite(bound):
                add(("numeric_rules", "<column>"), "Numeric bounds must be finite numbers.")
    for rule in config.date_rules.values():
        problem = _date_format_problem(rule.date_format)
        if problem:
            add(("date_rules", "<column>", "date_format"), problem)
    if issues:
        raise InvalidRulesError(issues)


def parse_rules(text: str | None) -> DetectionConfig | None:
    """Turn the submitted rules text into a DetectionConfig. None or blank text means "no rules"."""
    if text is None or not text.strip():
        return None
    if len(text) > MAX_RULES_CHARS:
        raise InvalidRulesError([RuleIssue((), f"The rules must be at most {MAX_RULES_CHARS} characters.")])
    try:
        raw = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise InvalidRulesError([RuleIssue((), "The rules must be valid JSON.")]) from None
    if not isinstance(raw, dict):
        raise InvalidRulesError([RuleIssue((), "The rules must be a JSON object.")])
    _check_sizes(raw)
    try:
        config = DetectionConfig.model_validate(raw)
    except ValidationError as error:
        raise InvalidRulesError(_pydantic_issues(error)) from None
    _check_values(config)
    return config


def check_rule_columns(config: DetectionConfig, columns: Iterable[str]) -> None:
    """Reject rules that name columns the uploaded file does not have.

    The detectors ignore such rules silently, which would make a mistyped name look like clean data.
    Only a count is reported, never the name.
    """
    present = {str(column) for column in columns}
    issues: list[RuleIssue] = []
    for field in (*_DICT_FIELDS, *_LIST_FIELDS):
        names = getattr(config, field) or []
        missing = sum(1 for name in names if name not in present)
        if missing:
            issues.append(RuleIssue((field,), f"{missing} column name(s) in this rule are not in the uploaded file."))
    if issues:
        raise InvalidRulesError(issues)