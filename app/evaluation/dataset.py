"""The fixed, labeled synthetic dataset used by the evaluation harness.

Everything here is invented and deterministic: no random numbers, no files, no network. Each
case is a small table with a ground-truth label for every column:

  1. classification label: public, internal, personal, financial or unknown (see ANNOTATION_GUIDELINES)
  2. detections: which finding categories SHOULD be reported for the column, and for the dataset

Honest limits (also printed in every report): the labels, the detectors and the mock provider
were written by the same author, so this measures how the implementation behaves on cases its
author could anticipate. It is not held out, has a single annotator and is not representative of
real data. Cases tagged "near_miss" and "known_limitation" are included on purpose so the report
shows where the implementation is weak, instead of only where it is strong.

validate_dataset() re-checks the ground truth with independent simple tests (null shares, emails,
phones, duplicates, rule violations), so a labeling mistake fails a test instead of silently
skewing a metric.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import pandas as pd

from app.schemas.findings import DateRule, DetectionConfig, NumericRule

CLASSIFICATION_LABELS = ("public", "internal", "personal", "financial", "unknown")
COLUMN_LEVEL_CATEGORIES = (
    "missing_values",
    "potential_email",
    "potential_phone",
    "numeric_rule_violation",
    "invalid_date",
)
DATASET_LEVEL_CATEGORIES = ("duplicate_rows",)
ALL_CATEGORIES = COLUMN_LEVEL_CATEGORIES + DATASET_LEVEL_CATEGORIES
TAGS = ("standard", "near_miss", "known_limitation")
DATASET_LEVEL_UNIT = "(whole dataset)"

# Ground truth for missing values avoids the detector's boundary on purpose: a column counts as
# "should be flagged" at 25% missing or more, and as "should be quiet" at 10% or less. Nothing sits
# between the two.
NULL_SHARE_FLAG = 0.25
NULL_SHARE_QUIET = 0.10

ANNOTATION_GUIDELINES = {
    "personal": "Values identify, describe or allow contacting a specific individual (names, emails, "
    "phone numbers, handles).",
    "financial": "Money amounts, prices, credit limits, balances or payment data.",
    "internal": "Operational or business data meant for internal use (ids, codes, quantities, "
    "internal dates, status flags).",
    "public": "Information intended for public disclosure (product catalogue names, public URLs, "
    "opening hours, published business contact lines).",
    "unknown": "A deliberately opaque column: no meaningful name and no meaningful content.",
    "detections": "missing_values: 25% or more of the values are missing. potential_email and "
    "potential_phone: at least one value is an email address or a North American-style or "
    "international phone number. numeric_rule_violation and invalid_date: only for columns that have "
    "a configured rule, and only when a value breaks it. duplicate_rows: two rows are identical in "
    "every column.",
}

_EMAIL_CHECK = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_CHECK = re.compile(r"(?<!\d)(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?!\d)")


@dataclass(frozen=True)
class LabeledColumn:
    name: str
    label: str
    detections: tuple[str, ...] = ()  # categories that SHOULD be reported for this column


@dataclass(frozen=True)
class LabeledCase:
    case_id: str
    tag: str
    description: str
    columns: tuple[LabeledColumn, ...]
    data: dict[str, tuple[Any, ...]]
    dataset_detections: tuple[str, ...] = ()
    numeric_min_rules: tuple[tuple[str, float], ...] = ()  # (column, minimum): values below it violate
    date_columns: tuple[str, ...] = ()  # columns checked as ISO dates (YYYY-MM-DD)


def _col(name: str, label: str, *detections: str) -> LabeledColumn:
    return LabeledColumn(name=name, label=label, detections=tuple(detections))


# --- Deterministic value helpers ---------------------------------------------------------------

_FIRST = ("ava", "ben", "carla", "dev", "elena", "farid", "gina", "hugo", "iris", "jonas")
_LAST = ("Moreno", "Nguyen", "Okafor", "Patel", "Quinn", "Rossi", "Silva", "Tanaka", "Ueda", "Varga")


def _names() -> tuple[str, ...]:
    return tuple(f"{first.title()} {last}" for first, last in zip(_FIRST, _LAST, strict=True))


def _emails() -> tuple[str, ...]:
    return tuple(f"{first}.{last.lower()}@example.com" for first, last in zip(_FIRST, _LAST, strict=True))


def _phones() -> tuple[str, ...]:
    formats = ("503-555-01{:02d}", "(503) 555-01{:02d}", "503.555.01{:02d}")
    return tuple(formats[i % 3].format(i) for i in range(10))


def _from_rows(names: Sequence[str], rows: Sequence[Sequence[Any]]) -> dict[str, tuple[Any, ...]]:
    return {name: tuple(row[i] for row in rows) for i, name in enumerate(names)}


def _cases() -> tuple[LabeledCase, ...]:
    names, emails, phones = _names(), _emails(), _phones()

    customers_clean = LabeledCase(
        case_id="customers_clean",
        tag="standard",
        description="Complete, duplicate-free customer table with clear personal, financial and operational columns.",
        columns=(
            _col("customer_id", "internal"),
            _col("full_name", "personal"),
            _col("email", "personal", "potential_email"),
            _col("phone", "personal", "potential_phone"),
            _col("credit_limit", "financial"),
            _col("signup_date", "internal"),
            _col("account_status", "internal"),
        ),
        data={
            "customer_id": tuple(range(1001, 1011)),
            "full_name": names,
            "email": emails,
            "phone": phones,
            "credit_limit": (5000, 7500, 3000, 10000, 2500, 8000, 6000, 4500, 12000, 3500),
            "signup_date": tuple(f"2026-01-{10 + i:02d}" for i in range(10)),
            "account_status": (
                "active", "active", "suspended", "active", "closed",
                "active", "active", "suspended", "active", "active",
            ),
        },
    )

    inventory_clean = LabeledCase(
        case_id="inventory_clean",
        tag="standard",
        description="Complete inventory table: operational codes, prices, and columns meant for the public.",
        columns=(
            _col("sku", "internal"),
            _col("product_name", "public"),
            _col("unit_price", "financial"),
            _col("quantity_on_hand", "internal"),
            _col("warehouse_code", "internal"),
            _col("public_url", "public"),
            _col("store_hours", "public"),
        ),
        data={
            "sku": tuple(f"SKU-{i + 1:04d}" for i in range(10)),
            "product_name": (
                "Olive Oil 1L", "Basmati Rice 5kg", "Whole Milk 2L", "Sea Salt 500g", "Black Tea 100 bags",
                "Tomato Paste 400g", "Oat Flakes 1kg", "Brown Sugar 1kg", "Spaghetti 500g", "Canned Chickpeas 400g",
            ),
            "unit_price": (4.5, 12.0, 2.1, 1.2, 3.4, 0.9, 2.8, 1.7, 1.1, 0.8),
            "quantity_on_hand": (120, 45, 300, 80, 60, 200, 90, 150, 400, 75),
            "warehouse_code": ("WH-A", "WH-A", "WH-B", "WH-B", "WH-C", "WH-A", "WH-C", "WH-B", "WH-A", "WH-C"),
            "public_url": tuple(f"https://www.example.com/products/{i + 1}" for i in range(10)),
            "store_hours": tuple(("Mon-Fri 8am-6pm", "Mon-Sat 9am-7pm", "Daily 7am-9pm")[i % 3] for i in range(10)),
        },
    )

    missing_values = LabeledCase(
        case_id="missing_values",
        tag="standard",
        description="Columns with 40%, 100% and 30% missing values, one with 10% (should stay quiet), and complete ones.",
        columns=(
            _col("order_id", "internal"),
            _col("ship_date", "internal", "missing_values"),
            _col("carrier_name", "internal"),
            _col("delivery_notes", "internal", "missing_values"),
            _col("tracking_code", "internal"),
            _col("discount_amount", "financial", "missing_values"),
        ),
        data={
            "order_id": tuple(range(5001, 5011)),
            "ship_date": (
                "2026-03-01", "2026-03-02", None, "2026-03-04", None,
                "2026-03-06", "2026-03-07", None, "2026-03-09", None,
            ),
            "carrier_name": (
                "FastFreight", "BlueLine", "FastFreight", "BlueLine", "FastFreight",
                None, "BlueLine", "FastFreight", "BlueLine", "FastFreight",
            ),
            "delivery_notes": (None,) * 10,
            "tracking_code": tuple(f"TRK-AA-{i + 1:03d}" for i in range(10)),
            "discount_amount": (5.0, None, 2.5, None, 0.0, 1.5, None, 3.0, 4.0, 2.0),
        },
    )

    support_phrases = (
        "Asked about the delivery window", "Requested a copy of the invoice", "Reported a damaged pallet",
        "Wants to change the delivery slot", "Asked about the return policy", "Confirmed the pickup time",
        "Needs a revised quote",
    )
    pii_patterns = LabeledCase(
        case_id="pii_patterns",
        tag="standard",
        description="Email and phone columns in several formats, and free text with emails in three of ten rows.",
        columns=(
            _col("contact_email", "personal", "potential_email"),
            _col("mobile_phone", "personal", "potential_phone"),
            _col("support_notes", "personal", "potential_email"),
            _col("employee_name", "personal"),
            _col("ticket_id", "internal"),
            _col("priority", "internal"),
        ),
        data={
            "contact_email": emails,
            "mobile_phone": phones,
            "support_notes": tuple(
                f"Customer asked us to reply to {_FIRST[i]}.k@example.org" if i in (2, 5, 8)
                else support_phrases[i % len(support_phrases)]
                for i in range(10)
            ),
            "employee_name": names,
            "ticket_id": tuple(range(9001, 9011)),
            "priority": tuple(("low", "medium", "high")[i % 3] for i in range(10)),
        },
    )

    near_miss_patterns = LabeledCase(
        case_id="near_miss_patterns",
        tag="near_miss",
        description="Values that look like emails or phone numbers at a glance but are not.",
        columns=(
            _col("social_handle", "personal"),
            _col("release_version", "internal"),
            _col("build_timestamp", "internal"),
            _col("meeting_note", "internal"),
            _col("batch_reference", "internal"),
        ),
        data={
            "social_handle": tuple(f"@{_FIRST[i]}_{10 + i}" for i in range(10)),
            "release_version": tuple(f"{i + 1}.{i + 2}.{i + 3}" for i in range(10)),
            "build_timestamp": tuple(f"2026-09-{i + 1:02d} 10:15:00" for i in range(10)),
            "meeting_note": tuple(
                ("see you @ 5pm", "call back @ noon", "lunch @ 1pm", "standup @ 9am", "review @ 3pm")[i % 5]
                for i in range(10)
            ),
            "batch_reference": tuple(f"A{i}B{i}-C{i}D{i}-E{i}F{i}" for i in range(10)),
        },
    )

    numeric_and_date_rules = LabeledCase(
        case_id="numeric_and_date_rules",
        tag="standard",
        description="Configured minimum-value and ISO-date rules, each with violating and clean columns.",
        columns=(
            _col("shipment_id", "internal"),
            _col("quantity", "internal", "numeric_rule_violation"),
            _col("weight_kg", "internal", "numeric_rule_violation"),
            _col("refund_amount", "financial"),
            _col("received_date", "internal", "invalid_date"),
            _col("delivery_date", "internal", "invalid_date"),
        ),
        data={
            "shipment_id": tuple(range(7001, 7011)),
            "quantity": (5, 3, -2, 7, 1, 4, 6, 2, 8, 9),
            "weight_kg": (12.5, 8.0, 15.2, -1.5, 9.9, 11.0, 7.3, 14.1, 6.8, 10.2),
            "refund_amount": (10.0, 0.5, 25.5, 3.0, 8.0, 12.0, 7.5, 1.0, 4.0, 6.0),
            "received_date": (
                "2026-09-01", "2026-02-30", "2026-09-03", "2026-09-04", "2026-09-05",
                "2026-09-06", "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10",
            ),
            "delivery_date": (
                "2026-09-11", "2026-09-12", "2026-13-01", "2026-09-14", "2026-09-15",
                "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-19", "2026-09-20",
            ),
        },
        numeric_min_rules=(("quantity", 0.0), ("weight_kg", 0.0), ("refund_amount", 0.0)),
        date_columns=("received_date", "delivery_date"),
    )

    unconfigured_rules = LabeledCase(
        case_id="unconfigured_rules",
        tag="near_miss",
        description="Negative numbers and an impossible date with NO rules configured: nothing should be reported.",
        columns=(
            _col("item_id", "internal"),
            _col("stock_delta", "internal"),
            _col("expiry_date", "internal"),
            _col("warehouse_code", "internal"),
        ),
        data={
            "item_id": tuple(range(300, 310)),
            "stock_delta": (-3, 4, -1, 2, 5, -2, 3, 1, -4, 6),
            "expiry_date": (
                "2027-01-31", "2027-02-30", "2027-03-15", "2027-04-01", "2027-05-20",
                "2027-06-11", "2027-07-04", "2027-08-19", "2027-09-23", "2027-10-30",
            ),
            "warehouse_code": ("WH-A", "WH-B", "WH-C", "WH-A", "WH-B", "WH-C", "WH-A", "WH-B", "WH-C", "WH-A"),
        },
    )

    duplicate_names = ("order_id", "customer_ref", "amount")
    duplicate_columns = (_col("order_id", "internal"), _col("customer_ref", "internal"), _col("amount", "financial"))
    base_rows = [
        (1, "C-10", 25.0), (2, "C-11", 30.0), (3, "C-12", 12.5), (4, "C-13", 40.0), (5, "C-14", 8.0),
        (6, "C-15", 19.0), (7, "C-16", 22.0), (8, "C-17", 15.0), (9, "C-18", 33.0),
    ]
    duplicates_present = LabeledCase(
        case_id="duplicates_present",
        tag="standard",
        description="Ten rows in which the last row repeats the fourth row exactly.",
        columns=duplicate_columns,
        data=_from_rows(duplicate_names, [*base_rows, base_rows[3]]),
        dataset_detections=("duplicate_rows",),
    )
    duplicates_absent = LabeledCase(
        case_id="duplicates_absent",
        tag="near_miss",
        description="Rows share the same customer and amount but differ in order_id, so no row repeats exactly.",
        columns=duplicate_columns,
        data={"order_id": tuple(range(1, 11)), "customer_ref": ("C-10",) * 10, "amount": (25.0,) * 10},
    )
    duplicates_repeated = LabeledCase(
        case_id="duplicates_repeated",
        tag="standard",
        description="Twelve rows in which two rows repeat earlier rows exactly.",
        columns=duplicate_columns,
        data=_from_rows(
            duplicate_names,
            [*base_rows, (10, "C-19", 27.0), base_rows[2], base_rows[5]],
        ),
        dataset_detections=("duplicate_rows",),
    )

    public_contacts = LabeledCase(
        case_id="public_contacts",
        tag="near_miss",
        description="Published branch names and a published hotline: phone-shaped values that are public, not personal.",
        columns=(
            _col("branch_name", "public"),
            _col("public_hotline", "public", "potential_phone"),
        ),
        data={
            "branch_name": (
                "Downtown Branch", "Riverside Branch", "Airport Branch", "Harbor Branch", "Hillcrest Branch",
                "Lakeside Branch", "Midtown Branch", "Northgate Branch", "Oakwood Branch", "Parkview Branch",
            ),
            "public_hotline": tuple(f"800-555-02{i:02d}" for i in range(10)),
        },
    )

    opaque_columns = LabeledCase(
        case_id="opaque_columns",
        tag="standard",
        description="Columns with meaningless names and content, plus one meaningless name that holds email addresses.",
        columns=(
            _col("field_a", "unknown"),
            _col("field_b", "unknown"),
            _col("column_3", "unknown"),
            _col("value", "unknown"),
            _col("misc", "unknown"),
            _col("ref_contact", "personal", "potential_email"),
        ),
        data={
            "field_a": ("k3x9", "q7m2", "z1p8", "m4w6", "t9c3", "b2n7", "y5r1", "h8d4", "v6j0", "g0s5"),
            "field_b": (17, 42, 8, 93, 61, 25, 70, 34, 56, 12),
            "column_3": (
                "alpha", "delta", "omega", "sigma", "theta", "kappa", "lambda", "gamma", "zeta", "beta",
            ),
            "value": (0.37, 0.82, 0.15, 0.64, 0.91, 0.28, 0.55, 0.73, 0.46, 0.09),
            "misc": ("x", "y", "x", "z", "y", "x", "z", "y", "x", "z"),
            "ref_contact": emails,
        },
    )

    known_limitations = LabeledCase(
        case_id="known_limitations",
        tag="known_limitation",
        description="Cases the documented detectors are not designed for: international and numeric phone numbers, "
        "a phone-like id, and placeholder or blank text meaning missing.",
        columns=(
            _col("intl_phone", "personal", "potential_phone"),
            _col("numeric_phone", "personal", "potential_phone"),
            _col("ten_digit_id", "internal"),
            _col("placeholder_missing", "internal", "missing_values"),
            _col("blank_strings", "internal", "missing_values"),
        ),
        data={
            "intl_phone": (
                "+44 20 7946 0958", "+33 1 42 68 53 00", "+49 30 9018 2034", "+81 3 1234 5678", "+61 2 9374 4000",
                "+34 91 123 45 67", "+39 06 6982 1234", "+31 20 794 0000", "+46 8 123 456 78", "+41 44 668 1800",
            ),
            "numeric_phone": tuple(5035550110 + i for i in range(10)),
            "ten_digit_id": tuple(str(4100000001 + i * 1111111) for i in range(10)),
            "placeholder_missing": ("N/A",) * 6 + ("ok-1", "ok-2", "ok-3", "ok-4"),
            "blank_strings": ("", "", "", "", "a", "b", "c", "d", "e", "f"),
        },
    )

    return (
        customers_clean, inventory_clean, missing_values, pii_patterns, near_miss_patterns,
        numeric_and_date_rules, unconfigured_rules, duplicates_present, duplicates_absent,
        duplicates_repeated, public_contacts, opaque_columns, known_limitations,
    )


CASES = _cases()


# --- Building inputs ---------------------------------------------------------------------------


def build_frame(case: LabeledCase) -> pd.DataFrame:
    """A fresh DataFrame for one case (never shared, so a caller cannot change the dataset)."""
    return pd.DataFrame({column.name: list(case.data[column.name]) for column in case.columns})


def detection_config(case: LabeledCase) -> DetectionConfig | None:
    """The detection rules the case configures, or None for the default configuration."""
    if not case.numeric_min_rules and not case.date_columns:
        return None
    return DetectionConfig(
        numeric_rules={name: NumericRule(min_value=minimum) for name, minimum in case.numeric_min_rules},
        date_rules={name: DateRule() for name in case.date_columns},
    )


# --- Summary and fingerprint ---------------------------------------------------------------------


def dataset_summary(cases: Sequence[LabeledCase] = CASES) -> dict[str, Any]:
    distribution = {label: 0 for label in CLASSIFICATION_LABELS}
    positives = {category: 0 for category in ALL_CATEGORIES}
    tags = {tag: 0 for tag in TAGS}
    columns = rows = 0
    for case in cases:
        tags[case.tag] += 1
        rows += len(next(iter(case.data.values())))
        for column in case.columns:
            columns += 1
            distribution[column.label] += 1
            for category in column.detections:
                positives[category] += 1
        for category in case.dataset_detections:
            positives[category] += 1
    return {
        "fingerprint": dataset_fingerprint(cases),
        "cases": len(cases),
        "columns": columns,
        "rows_total": rows,
        "dataset_level_units": len(cases),
        "label_distribution": distribution,
        "detection_positives": positives,
        "cases_by_tag": tags,
    }


def dataset_fingerprint(cases: Sequence[LabeledCase] = CASES) -> str:
    """SHA-256 of the complete dataset (data, labels, rules), so any change is visible in a report."""
    canonical = [
        {
            "case_id": case.case_id,
            "tag": case.tag,
            "description": case.description,
            "columns": [
                {"name": c.name, "label": c.label, "detections": sorted(c.detections)} for c in case.columns
            ],
            "data": {name: list(values) for name, values in case.data.items()},
            "dataset_detections": sorted(case.dataset_detections),
            "numeric_min_rules": [[name, minimum] for name, minimum in case.numeric_min_rules],
            "date_columns": list(case.date_columns),
        }
        for case in cases
    ]
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- Validation of the ground truth ------------------------------------------------------------------


def _is_valid_iso_date(value: Any) -> bool:
    try:
        datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError:
        return False
    return True


def validate_dataset(cases: Sequence[LabeledCase] = CASES) -> list[str]:
    """Return a list of problems with the ground truth (empty when it is consistent).

    The semantic checks use independent, deliberately simple tests. Cases tagged known_limitation
    skip the missing-value, email and phone checks, because their labels intentionally describe what
    the data means, not what a pattern would match.
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    for case in cases:
        cid = case.case_id
        if cid in seen_ids:
            problems.append(f"{cid}: duplicate case id")
        seen_ids.add(cid)
        if case.tag not in TAGS:
            problems.append(f"{cid}: unknown tag '{case.tag}'")
        names = [column.name for column in case.columns]
        if not names or len(set(names)) != len(names):
            problems.append(f"{cid}: columns must be non-empty and unique")
            continue
        if set(case.data) != set(names):
            problems.append(f"{cid}: data columns do not match the labeled columns")
            continue
        lengths = {len(values) for values in case.data.values()}
        if len(lengths) != 1 or 0 in lengths:
            problems.append(f"{cid}: all columns need the same, non-zero number of rows")
            continue
        rows = lengths.pop()
        numeric_rules = dict(case.numeric_min_rules)
        for name in (*numeric_rules, *case.date_columns):
            if name not in names:
                problems.append(f"{cid}.{name}: a configured rule names a column that does not exist")
        for category in case.dataset_detections:
            if category not in DATASET_LEVEL_CATEGORIES:
                problems.append(f"{cid}: '{category}' is not a dataset-level category")

        repeated = len({tuple(case.data[name][i] for name in names) for i in range(rows)}) < rows
        if repeated != ("duplicate_rows" in case.dataset_detections):
            problems.append(f"{cid}: repeated rows do not match the 'duplicate_rows' expectation")

        for column in case.columns:
            where = f"{cid}.{column.name}"
            values = case.data[column.name]
            present = [value for value in values if value is not None]
            expected = set(column.detections)
            if column.label not in CLASSIFICATION_LABELS:
                problems.append(f"{where}: unknown classification label '{column.label}'")
            for category in expected - set(COLUMN_LEVEL_CATEGORIES):
                problems.append(f"{where}: '{category}' is not a column-level category")

            if column.name in numeric_rules:
                violated = any(value < numeric_rules[column.name] for value in present)
            else:
                violated = False
            if violated != ("numeric_rule_violation" in expected):
                problems.append(f"{where}: values do not match the 'numeric_rule_violation' expectation")
            if column.name in case.date_columns:
                invalid = any(not _is_valid_iso_date(value) for value in present)
            else:
                invalid = False
            if invalid != ("invalid_date" in expected):
                problems.append(f"{where}: values do not match the 'invalid_date' expectation")

            if case.tag == "known_limitation":
                continue
            share = (rows - len(present)) / rows
            if not (share >= NULL_SHARE_FLAG or share <= NULL_SHARE_QUIET):
                problems.append(f"{where}: missing share {share:.2f} sits in the ambiguous zone")
            if (share >= NULL_SHARE_FLAG) != ("missing_values" in expected):
                problems.append(f"{where}: missing share does not match the 'missing_values' expectation")
            texts = [value for value in present if isinstance(value, str)]
            if any(_EMAIL_CHECK.search(text) for text in texts) != ("potential_email" in expected):
                problems.append(f"{where}: values do not match the 'potential_email' expectation")
            if any(_PHONE_CHECK.search(text) for text in texts) != ("potential_phone" in expected):
                problems.append(f"{where}: values do not match the 'potential_phone' expectation")
    return problems