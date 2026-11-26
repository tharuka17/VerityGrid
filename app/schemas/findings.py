"""Validated data structures for deterministic finding detection.

Nothing here touches the database or an LLM. DetectedFinding is the in-memory result of a
detector; mapping it onto the ORM Finding row happens in a later phase.

Privacy: DetectedFinding.safe_evidence may hold only small scalar values (counts, percentages,
rule names). The schema rejects nested values and long strings, so raw cell values cannot be
carried through it by accident.
"""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# One source of truth for these allowed values: they are also used by the database models.
from app.db.models import ReviewStatus, Severity

MAX_EVIDENCE_KEY_LENGTH = 60
MAX_EVIDENCE_TEXT_LENGTH = 120

EvidenceValue = int | float | bool | str
Percent = Annotated[float, Field(ge=0.0, le=100.0)]


class FindingCategory(StrEnum):
    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    POTENTIAL_EMAIL = "potential_email"
    POTENTIAL_PHONE = "potential_phone"
    NUMERIC_RULE_VIOLATION = "numeric_rule_violation"
    INVALID_DATE = "invalid_date"


class DetectedFinding(BaseModel):
    """One deterministic finding. Immutable once created."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: FindingCategory
    severity: Severity
    column_name: str | None = Field(default=None, max_length=255)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=600)
    safe_evidence: dict[str, EvidenceValue] = Field(min_length=1)
    detection_method: str = Field(min_length=1, max_length=100)
    recommendation: str = Field(min_length=1, max_length=400)
    review_status: ReviewStatus = ReviewStatus.PENDING

    @field_validator("safe_evidence")
    @classmethod
    def _evidence_must_be_small_and_scalar(
        cls, value: dict[str, EvidenceValue]
    ) -> dict[str, EvidenceValue]:
        for key, item in value.items():
            if len(key) > MAX_EVIDENCE_KEY_LENGTH:
                raise ValueError(f"evidence key longer than {MAX_EVIDENCE_KEY_LENGTH} characters")
            if isinstance(item, str) and len(item) > MAX_EVIDENCE_TEXT_LENGTH:
                raise ValueError(
                    f"evidence text longer than {MAX_EVIDENCE_TEXT_LENGTH} characters"
                )
        return value

    @model_validator(mode="after")
    def _column_required_for_column_level_categories(self) -> Self:
        if self.category != FindingCategory.DUPLICATE_ROWS and self.column_name is None:
            raise ValueError(f"column_name is required for category '{self.category.value}'")
        return self

    @property
    def dedup_key(self) -> tuple[str, str | None]:
        """Two findings with the same key describe the same problem in the same scan."""
        return (self.category.value, self.column_name)


class NumericRule(BaseModel):
    """Inclusive bounds for one explicitly designated numeric column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_value: float | None = None
    max_value: float | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.min_value is None and self.max_value is None:
            raise ValueError("set at least one of min_value or max_value")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must not be greater than max_value")
        return self


class DateRule(BaseModel):
    """Expected strptime format for one explicitly designated date column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date_format: str = Field(default="%Y-%m-%d", min_length=2, max_length=50)


class DetectionConfig(BaseModel):
    """Which rules apply to which columns, and with what limits.

    Columns named here that are not present in the DataFrame are ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Missing values: a column is reported when its null percentage is at or above its threshold.
    missing_value_threshold_percent: Percent = 20.0
    missing_value_overrides: dict[str, Percent] = Field(default_factory=dict)
    missing_value_exempt_columns: list[str] = Field(default_factory=list)

    # Duplicate rows: None compares entire rows; a list compares only those key columns.
    duplicate_subset: list[str] | None = None

    # Email/phone patterns: None scans every text column; a list restricts scanning to those.
    pattern_columns: list[str] | None = None
    pattern_min_match_count: int = Field(default=1, ge=1)

    # Explicit per-column business rules. Columns not listed are never checked.
    numeric_rules: dict[str, NumericRule] = Field(default_factory=dict)
    date_rules: dict[str, DateRule] = Field(default_factory=dict)

    @field_validator("duplicate_subset")
    @classmethod
    def _subset_must_not_be_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) == 0:
            raise ValueError("duplicate_subset must list at least one column, or be None")
        return value