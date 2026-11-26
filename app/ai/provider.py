"""Provider interface and data contracts for AI-assisted column classification.

What flows through this boundary:
  IN   ColumnMetadata: sanitized, aggregate facts about ONE column (name, type, counts,
       percentages) plus summaries of the deterministic findings for that column.
       It has no field that can hold a cell value.
  OUT  A raw mapping from the provider, which the classifier validates against
       ClassificationSuggestion. Provider output is never trusted until it passes.

Nothing here touches the database, files or the network. A real provider (a later phase)
implements ClassificationProvider; this module does not depend on any specific model.
"""

import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.db.models import Severity
from app.schemas.findings import EvidenceValue, FindingCategory

MOCK_PREFIX = "[MOCK] "
MAX_COLUMN_NAME_LENGTH = 64
MAX_EXPLANATION_LENGTH = 300
MAX_REFS = 8
MAX_EVIDENCE_ITEMS = 20
MAX_EVIDENCE_KEY_LENGTH = 60
MAX_EVIDENCE_TEXT_LENGTH = 120

Percent = Annotated[float, Field(ge=0.0, le=100.0)]

# Supporting references have a fixed shape: "column_name" or "finding:potential_email".
_REF_PATTERN = re.compile(r"[a-z_]{1,30}(?::[a-z_]{1,30})?")


def has_control_characters(text: str) -> bool:
    """True if the text contains control, format (invisible) or similar characters."""
    return any(unicodedata.category(char).startswith("C") for char in text)


# --- Enumerations ------------------------------------------------------------


class ClassificationLabel(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PERSONAL = "personal"
    FINANCIAL = "financial"
    UNKNOWN = "unknown"


class ReviewRecommendation(StrEnum):
    """How strongly a human review is advised. Every suggestion is only a suggestion:
    nothing in this project applies a label without a human decision."""

    REQUIRED = "review_required"
    RECOMMENDED = "review_recommended"
    OPTIONAL = "review_optional"


RECOMMENDATION_RANK = {
    ReviewRecommendation.OPTIONAL: 0,
    ReviewRecommendation.RECOMMENDED: 1,
    ReviewRecommendation.REQUIRED: 2,
}


class ClassificationSource(StrEnum):
    MOCK = "mock"  # deterministic rules, not an LLM
    LLM = "llm"  # a real language model
    NONE = "none"  # no suggestion was produced (fallback result)


# --- Input to a provider -----------------------------------------------------


class FindingSummary(BaseModel):
    """One deterministic finding for the column. Titles and descriptions are deliberately
    not included: they embed the column name, which is untrusted text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: FindingCategory
    severity: Severity
    evidence: dict[str, EvidenceValue]  # counts, percentages and rule names only

    @field_validator("evidence")
    @classmethod
    def _evidence_is_small_and_clean(
        cls, value: dict[str, EvidenceValue]
    ) -> dict[str, EvidenceValue]:
        if len(value) > MAX_EVIDENCE_ITEMS:
            raise ValueError("too many evidence items")
        for key, item in value.items():
            if len(key) > MAX_EVIDENCE_KEY_LENGTH or has_control_characters(key):
                raise ValueError("invalid evidence key")
            if isinstance(item, str) and (
                len(item) > MAX_EVIDENCE_TEXT_LENGTH or has_control_characters(item)
            ):
                raise ValueError("invalid evidence text")
        return value


class ColumnMetadata(BaseModel):
    """Sanitized facts about one column. This is everything a provider is allowed to see."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column_ref: str = Field(pattern=r"^col_\d{1,4}$")  # opaque id, e.g. "col_3"
    column_name: str = Field(min_length=1, max_length=MAX_COLUMN_NAME_LENGTH)  # UNTRUSTED text
    name_was_modified: bool = False  # True if sanitizing changed or shortened the name
    inferred_type: str = Field(min_length=1, max_length=30)
    row_count: int = Field(ge=0)
    null_percent: Percent
    distinct_percent: Percent | None = None  # distinct values as a % of non-null values
    findings: tuple[FindingSummary, ...] = ()

    @field_validator("column_name")
    @classmethod
    def _name_has_no_control_characters(cls, value: str) -> str:
        if has_control_characters(value):
            raise ValueError("column name contains control characters")
        return value

    @property
    def available_refs(self) -> frozenset[str]:
        """The only references a provider may cite as supporting evidence."""
        refs = {"column_name", "inferred_type", "null_percent"}
        if self.distinct_percent is not None:
            refs.add("distinct_percent")
        refs.update(f"finding:{finding.category.value}" for finding in self.findings)
        return frozenset(refs)


# --- Output from a provider --------------------------------------------------


class ClassificationSuggestion(BaseModel):
    """The validated shape of a provider's answer. Unknown fields are rejected, so a provider
    cannot assert its own provenance (source, name, mock status) in its output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: ClassificationLabel
    explanation: str = Field(min_length=1, max_length=MAX_EXPLANATION_LENGTH)
    supporting_refs: tuple[str, ...] = Field(default=(), max_length=MAX_REFS)
    review_recommendation: ReviewRecommendation

    @field_validator("explanation")
    @classmethod
    def _normalize_explanation(cls, value: str) -> str:
        normalized = " ".join(value.split())  # collapses newlines, tabs and repeated spaces
        if not normalized:
            raise ValueError("explanation must not be blank")
        if has_control_characters(normalized):
            raise ValueError("explanation contains control characters")
        return normalized

    @field_validator("supporting_refs")
    @classmethod
    def _refs_have_a_safe_shape(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not all(_REF_PATTERN.fullmatch(ref) for ref in value):
            raise ValueError("supporting reference has an invalid format")
        return value


# --- Provider interface ------------------------------------------------------


class ProviderUnavailableError(Exception):
    """The provider cannot answer right now (offline, timeout, rate limit, missing credentials)."""


class ClassificationProvider(ABC):
    """Contract for anything that suggests a classification.

    Implementations must:
      - use only the ColumnMetadata they are given, and never log it;
      - enforce their own timeouts and bounded retries;
      - raise ProviderUnavailableError when they cannot answer;
      - return a plain mapping with exactly the ClassificationSuggestion fields. The
        classifier validates it, so a provider is not trusted to be well-formed.

    They must not touch the database, files or any tool that changes state.
    """

    name: ClassVar[str]
    source: ClassVar[ClassificationSource]

    @abstractmethod
    def classify(self, metadata: ColumnMetadata) -> Mapping[str, Any]:
        """Return a raw suggestion for one column."""