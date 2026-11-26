"""Validating, fail-safe classifier that sits between the application and a provider.

Responsibilities:
  1. Build ColumnMetadata from profiling and detection results: sanitized aggregate facts only.
     Raw cell values never reach this module, because ColumnProfileResult and DetectedFinding
     contain none.
  2. Call the provider and validate whatever it returns.
  3. Apply policy in ordinary code (a provider's answer is only a suggestion).
  4. Fail safely: any failure becomes a clearly marked "unknown, review required" result
     instead of an exception.

No database access happens here. Persisting results is a separate, later step.
Logs contain column references, the provider name, statuses, labels and exception class
names only. Never column names, explanations or values.
"""

import logging
import unicodedata
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, ValidationError

from app.ai.provider import (
    MAX_COLUMN_NAME_LENGTH,
    MAX_EVIDENCE_TEXT_LENGTH,
    MAX_EXPLANATION_LENGTH,
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
)
from app.schemas.findings import DetectedFinding, FindingCategory
from app.services.profiling import ColumnProfileResult, DatasetProfileResult

logger = logging.getLogger(__name__)

_ALWAYS_REVIEW_LABELS = frozenset(
    {ClassificationLabel.PERSONAL, ClassificationLabel.FINANCIAL, ClassificationLabel.UNKNOWN}
)
_PERSONAL_PATTERN_CATEGORIES = frozenset(
    {FindingCategory.POTENTIAL_EMAIL, FindingCategory.POTENTIAL_PHONE}
)

NOTE_SENSITIVE = "Review recommendation raised to 'review_required' for a sensitive or uncertain label."
NOTE_CONFLICT = (
    "The suggested label conflicts with deterministic email or phone findings for this column; "
    "human review is required."
)


class ResultStatus(StrEnum):
    OK = "ok"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_ERROR = "provider_error"
    INVALID_OUTPUT = "invalid_output"


_FALLBACK_EXPLANATIONS = {
    ResultStatus.PROVIDER_UNAVAILABLE: (
        "No suggestion available: the classification provider is unavailable. "
        "Classify this column manually."
    ),
    ResultStatus.PROVIDER_ERROR: (
        "No suggestion available: the classification provider failed. "
        "Classify this column manually."
    ),
    ResultStatus.INVALID_OUTPUT: (
        "No suggestion available: the provider returned an invalid response, which was "
        "discarded. Classify this column manually."
    ),
}

_DISPLAY_SOURCE = {
    ClassificationSource.MOCK: "Mock suggestion (rule-based, not an LLM)",
    ClassificationSource.LLM: "AI suggestion (LLM), unverified: a human must review it",
    ClassificationSource.NONE: "No suggestion",
}


class ClassificationResult(BaseModel):
    """A validated suggestion, or a safe fallback. Provenance is set here, never by the provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column_ref: str
    column_name: str  # sanitized
    label: ClassificationLabel
    explanation: str
    supporting_refs: tuple[str, ...]
    review_recommendation: ReviewRecommendation
    source: ClassificationSource
    provider_name: str
    status: ResultStatus
    policy_notes: tuple[str, ...] = ()

    @property
    def display_source(self) -> str:
        """Text the UI shows next to a suggestion."""
        return _DISPLAY_SOURCE[self.source]


# --- Sanitizing untrusted text and building provider input ---------------------


def sanitize_text(value: object, max_length: int) -> tuple[str, bool]:
    """Normalize untrusted text for use as data: NFKC, no control or invisible characters,
    single spaces, shortened to max_length. Returns (clean_text, was_changed).

    This limits length and character classes. It cannot make text "safe" to place inside an
    instruction, which is why names are always passed as structured data fields.
    """
    original = str(value)
    text = unicodedata.normalize("NFKC", original)
    kept = "".join(
        " " if char.isspace() else char
        for char in text
        if char.isspace() or not unicodedata.category(char).startswith("C")
    )
    cleaned = " ".join(kept.split())[:max_length].rstrip()
    return cleaned, cleaned != original


def build_column_metadata(
    profile: ColumnProfileResult,
    findings: Sequence[DetectedFinding],
    *,
    column_ref: str,
) -> ColumnMetadata:
    """Turn one column's profile and its findings into provider input.

    Only findings whose column matches are included, and only their category, severity and safe
    scalar evidence. Titles and descriptions are omitted because they embed the column name.
    """
    name, modified = sanitize_text(profile.column_name, MAX_COLUMN_NAME_LENGTH)
    if not name:  # a name made only of invisible characters
        name, modified = column_ref, True

    non_null_count = profile.row_count - profile.null_count
    distinct_percent = (
        round(100.0 * profile.distinct_count / non_null_count, 2)
        if profile.distinct_count is not None and non_null_count > 0
        else None
    )

    summaries = tuple(
        FindingSummary(
            category=finding.category,
            severity=finding.severity,
            evidence={
                key: (
                    sanitize_text(item, MAX_EVIDENCE_TEXT_LENGTH)[0]
                    if isinstance(item, str)
                    else item
                )
                for key, item in finding.safe_evidence.items()
            },
        )
        for finding in findings
        if finding.column_name == profile.column_name
    )

    return ColumnMetadata(
        column_ref=column_ref,
        column_name=name,
        name_was_modified=modified,
        inferred_type=profile.inferred_type.value,
        row_count=profile.row_count,
        null_percent=profile.null_percent,
        distinct_percent=distinct_percent,
        findings=summaries,
    )


# --- Policy ----------------------------------------------------------------------


def _apply_policy(
    metadata: ColumnMetadata, suggestion: ClassificationSuggestion
) -> tuple[ReviewRecommendation, tuple[str, ...]]:
    """Raise the review recommendation where policy demands. The label is never changed."""
    recommendation = suggestion.review_recommendation
    notes: list[str] = []

    if (
        suggestion.label in _ALWAYS_REVIEW_LABELS
        and RECOMMENDATION_RANK[recommendation] < RECOMMENDATION_RANK[ReviewRecommendation.REQUIRED]
    ):
        recommendation = ReviewRecommendation.REQUIRED
        notes.append(NOTE_SENSITIVE)

    pattern_found = any(f.category in _PERSONAL_PATTERN_CATEGORIES for f in metadata.findings)
    if pattern_found and suggestion.label is not ClassificationLabel.PERSONAL:
        recommendation = ReviewRecommendation.REQUIRED
        notes.append(NOTE_CONFLICT)

    return recommendation, tuple(notes)


# --- Classifier ------------------------------------------------------------------


class ColumnClassifier:
    def __init__(self, provider: ClassificationProvider) -> None:
        if not isinstance(provider, ClassificationProvider):
            raise TypeError("provider must be a ClassificationProvider.")
        source = getattr(provider, "source", None)
        if not isinstance(source, ClassificationSource) or source is ClassificationSource.NONE:
            raise TypeError("provider.source must be ClassificationSource.MOCK or .LLM.")
        self._provider = provider

    def classify_dataset(
        self, profile: DatasetProfileResult, findings: Sequence[DetectedFinding]
    ) -> list[ClassificationResult]:
        """One result per column, in column order. Never raises for provider problems."""
        return [
            self.classify_column(column, findings, column_ref=f"col_{position}")
            for position, column in enumerate(profile.columns, start=1)
        ]

    def classify_column(
        self,
        profile: ColumnProfileResult,
        findings: Sequence[DetectedFinding],
        *,
        column_ref: str = "col_1",
    ) -> ClassificationResult:
        metadata = build_column_metadata(profile, findings, column_ref=column_ref)
        return self.classify_metadata(metadata)

    def classify_metadata(self, metadata: ColumnMetadata) -> ClassificationResult:
        provider = self._provider
        try:
            raw = provider.classify(metadata)
        except ProviderUnavailableError:
            logger.warning(
                "Provider unavailable: column_ref=%s provider=%s", metadata.column_ref, provider.name
            )
            return self._fallback(metadata, ResultStatus.PROVIDER_UNAVAILABLE)
        except Exception as error:  # a provider must never be able to crash a scan
            logger.warning(
                "Provider failed: column_ref=%s provider=%s error_type=%s",
                metadata.column_ref,
                provider.name,
                type(error).__name__,
            )
            return self._fallback(metadata, ResultStatus.PROVIDER_ERROR)

        try:
            suggestion = ClassificationSuggestion.model_validate(raw)
        except (ValidationError, TypeError):
            logger.warning(
                "Invalid provider output: column_ref=%s provider=%s",
                metadata.column_ref,
                provider.name,
            )
            return self._fallback(metadata, ResultStatus.INVALID_OUTPUT)

        # Every cited reference must be one the request offered; otherwise it was invented.
        if not set(suggestion.supporting_refs) <= metadata.available_refs:
            logger.warning(
                "Provider cited unknown references: column_ref=%s provider=%s",
                metadata.column_ref,
                provider.name,
            )
            return self._fallback(metadata, ResultStatus.INVALID_OUTPUT)

        explanation = suggestion.explanation
        if provider.source is ClassificationSource.MOCK and not explanation.startswith(MOCK_PREFIX):
            explanation = (MOCK_PREFIX + explanation)[:MAX_EXPLANATION_LENGTH].rstrip()

        recommendation, notes = _apply_policy(metadata, suggestion)
        logger.info(
            "Classified: column_ref=%s provider=%s label=%s",
            metadata.column_ref,
            provider.name,
            suggestion.label.value,
        )
        return ClassificationResult(
            column_ref=metadata.column_ref,
            column_name=metadata.column_name,
            label=suggestion.label,
            explanation=explanation,
            supporting_refs=tuple(dict.fromkeys(suggestion.supporting_refs)),
            review_recommendation=recommendation,
            source=provider.source,
            provider_name=provider.name,
            status=ResultStatus.OK,
            policy_notes=notes,
        )

    def _fallback(self, metadata: ColumnMetadata, status: ResultStatus) -> ClassificationResult:
        return ClassificationResult(
            column_ref=metadata.column_ref,
            column_name=metadata.column_name,
            label=ClassificationLabel.UNKNOWN,
            explanation=_FALLBACK_EXPLANATIONS[status],
            supporting_refs=(),
            review_recommendation=ReviewRecommendation.REQUIRED,
            source=ClassificationSource.NONE,
            provider_name=self._provider.name,
            status=status,
        )