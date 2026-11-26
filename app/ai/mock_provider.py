"""Deterministic, offline mock provider.

It applies simple keyword rules to the sanitized column name and to the deterministic findings.
It is NOT a language model and does not infer anything. Its output is labelled as mock in three
places: the `source` it declares, a "[MOCK]" prefix in every explanation (the classifier
re-checks this), and the classifier's display text.

Priority, first match wins:
  1. email or phone findings            -> personal
  2. personal name keywords             -> personal
  3. financial name keywords            -> financial
  4. public (catalog) name keywords     -> public
  5. internal (operational) keywords    -> internal
  6. otherwise                          -> unknown
"""

import re
from collections.abc import Mapping
from typing import Any, ClassVar

from app.ai.provider import (
    MOCK_PREFIX,
    ClassificationLabel,
    ClassificationProvider,
    ClassificationSource,
    ColumnMetadata,
    ReviewRecommendation,
)
from app.schemas.findings import FindingCategory

PERSONAL_TOKENS = frozenset(
    {
        "email", "phone", "mobile", "telephone", "tel", "fax", "contact", "address",
        "dob", "birthdate", "birthday", "ssn", "passport", "firstname", "lastname",
        "fullname", "surname",
    }
)
FINANCIAL_TOKENS = frozenset(
    {
        "price", "cost", "amount", "total", "credit", "limit", "balance", "invoice",
        "payment", "salary", "revenue", "margin", "discount", "tax", "fee", "charge",
        "bank", "iban",
    }
)
PUBLIC_TOKENS = frozenset({"sku", "product", "category", "unit", "uom", "measure", "brand"})
INTERNAL_TOKENS = frozenset(
    {
        "warehouse", "status", "quantity", "qty", "stock", "reorder", "order", "delivery",
        "customer", "supplier", "notes", "date", "since", "count", "id",
    }
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_PATTERN_CATEGORIES = (FindingCategory.POTENTIAL_EMAIL, FindingCategory.POTENTIAL_PHONE)


def name_tokens(column_name: str) -> frozenset[str]:
    """Split a column name into lowercase words: "emailAddress" and "email_address" agree."""
    spaced = _CAMEL_BOUNDARY.sub("_", column_name)
    return frozenset(token for token in _NON_ALPHANUMERIC.split(spaced.lower()) if token)


class MockClassificationProvider(ClassificationProvider):
    name: ClassVar[str] = "mock-rules-v1"
    source: ClassVar[ClassificationSource] = ClassificationSource.MOCK

    def classify(self, metadata: ColumnMetadata) -> Mapping[str, Any]:
        categories = {finding.category for finding in metadata.findings}
        tokens = name_tokens(metadata.column_name)
        pattern_refs = [f"finding:{c.value}" for c in _PATTERN_CATEGORIES if c in categories]

        if pattern_refs:
            label, refs = ClassificationLabel.PERSONAL, pattern_refs
            reason = "Deterministic pattern checks found email-like or phone-like values in this column."
        elif tokens & PERSONAL_TOKENS:
            label, refs = ClassificationLabel.PERSONAL, ["column_name"]
            reason = "The column name suggests personal contact or identity data."
        elif tokens & FINANCIAL_TOKENS:
            label, refs = ClassificationLabel.FINANCIAL, ["column_name"]
            reason = "The column name suggests monetary or billing data."
        elif tokens & PUBLIC_TOKENS:
            label, refs = ClassificationLabel.PUBLIC, ["column_name"]
            reason = "The column name suggests general product or catalog information."
        elif tokens & INTERNAL_TOKENS:
            label, refs = ClassificationLabel.INTERNAL, ["column_name"]
            reason = "The column name suggests internal operational data."
        else:
            label, refs = ClassificationLabel.UNKNOWN, ["column_name"]
            reason = "No built-in rule matched this column's name or findings."

        if label in (
            ClassificationLabel.PERSONAL,
            ClassificationLabel.FINANCIAL,
            ClassificationLabel.UNKNOWN,
        ):
            recommendation = ReviewRecommendation.REQUIRED
        elif label is ClassificationLabel.INTERNAL:
            recommendation = ReviewRecommendation.RECOMMENDED
        else:  # public
            recommendation = (
                ReviewRecommendation.RECOMMENDED
                if metadata.findings
                else ReviewRecommendation.OPTIONAL
            )

        # Plain JSON-style values, like a real provider's parsed response.
        return {
            "label": label.value,
            "explanation": f"{MOCK_PREFIX}Rule-based placeholder, not LLM inference. {reason}",
            "supporting_refs": refs,
            "review_recommendation": recommendation.value,
        }