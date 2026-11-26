import ast
import copy
import dataclasses
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.ai.classifier import ColumnClassifier
from app.ai.mock_provider import MockClassificationProvider
from app.ai.provider import (
    ClassificationLabel,
    ClassificationProvider,
    ClassificationSource,
    ProviderUnavailableError,
)
from app.core.config import Settings
from app.evaluation import dataset as ds
from app.evaluation import harness as hn
from app.evaluation import metrics as mx

ROOT = Path(__file__).resolve().parent.parent
FIXED_TIME = datetime(2026, 11, 14, 0, 0, tzinfo=timezone.utc)
GOOD: dict[str, Any] = {
    "label": "internal",
    "explanation": "Operational data.",
    "supporting_refs": ["column_name"],
    "review_recommendation": "review_recommended",
}


def fixed_clock() -> datetime:
    return FIXED_TIME


class ConstantProvider(ClassificationProvider):
    """A stand-in LLM that always answers 'internal'. Counts its calls; never touches a network."""

    name = "llm-stub"
    source = ClassificationSource.LLM

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, metadata: Any) -> Any:
        self.calls += 1
        return GOOD


class DownProvider(ClassificationProvider):
    name = "llm-down"
    source = ClassificationSource.LLM

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, metadata: Any) -> Any:
        self.calls += 1
        raise ProviderUnavailableError("offline")


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test in this file may reach a real provider, whatever the developer's .env says."""
    for name in (
        "AI_PROVIDER",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GOOGLE_API_KEY",
        "GEMINI_TIMEOUT_SECONDS",
        "GEMINI_MAX_ATTEMPTS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return hn.build_report(repeats=2, warmup=False, clock=fixed_clock)


def column_total() -> int:
    return sum(len(case.columns) for case in ds.CASES)


def replace_column(case: ds.LabeledCase, name: str, **changes: Any) -> ds.LabeledCase:
    columns = tuple(dataclasses.replace(c, **changes) if c.name == name else c for c in case.columns)
    return dataclasses.replace(case, columns=columns)


def case_by_id(case_id: str) -> ds.LabeledCase:
    return next(case for case in ds.CASES if case.case_id == case_id)


# =============================================================================================
# 1. Metric math
# =============================================================================================


def test_precision_recall_f1_values() -> None:
    result = mx.precision_recall_f1(tp=8, fp=2, fn=4)

    assert result["precision"] == pytest.approx(0.8)
    assert result["recall"] == pytest.approx(8 / 12)
    assert result["f1"] == pytest.approx(16 / 22)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        pytest.param((0, 0, 0), (None, None, None), id="nothing-at-all"),
        pytest.param((0, 3, 0), (0.0, None, None), id="no-actual-positives"),
        pytest.param((0, 2, 3), (0.0, 0.0, 0.0), id="all-wrong"),
        pytest.param((5, 0, 0), (1.0, 1.0, 1.0), id="perfect"),
    ],
)
def test_undefined_rates_are_none_not_zero(counts: tuple[int, int, int], expected: tuple[Any, Any, Any]) -> None:
    result = mx.precision_recall_f1(*counts)

    assert (result["precision"], result["recall"], result["f1"]) == expected


@pytest.mark.parametrize(
    ("successes", "total", "low", "high"),
    [
        pytest.param(5, 10, 0.2366, 0.7634, id="half-of-ten"),
        pytest.param(10, 10, 0.7225, 1.0, id="all-of-ten"),
        pytest.param(0, 10, 0.0, 0.2775, id="none-of-ten"),
    ],
)
def test_wilson_interval_known_values(successes: int, total: int, low: float, high: float) -> None:
    interval = mx.wilson_interval(successes, total)

    assert interval is not None
    assert interval[0] == pytest.approx(low, abs=1e-3)
    assert interval[1] == pytest.approx(high, abs=1e-3)


def test_wilson_interval_is_none_without_data() -> None:
    assert mx.wilson_interval(0, 0) is None


def test_wilson_interval_narrows_with_more_data() -> None:
    small = mx.wilson_interval(5, 10)
    large = mx.wilson_interval(50, 100)

    assert small is not None and large is not None
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        mx.wilson_interval(11, 10)


def test_binary_report_is_internally_consistent() -> None:
    result = mx.binary_report(tp=8, fp=2, fn=4, tn=86)

    assert (result["support"], result["predicted_positive"], result["units"]) == (12, 10, 100)
    assert result["precision"] == pytest.approx(0.8)
    low, high = result["recall_ci95"]
    assert 0.0 <= low < result["recall"] < high <= 1.0
    low, high = result["precision_ci95"]
    assert low < result["precision"] < high


def test_multiclass_report_matches_a_hand_computed_example() -> None:
    pairs = (
        [("a", "a")] * 3 + [("a", "b")] + [("b", "b")] * 2 + [("b", "a")] + [("b", mx.NO_PREDICTION)]
        + [("c", "c")] + [("c", "a")]
    )

    report = mx.multiclass_report(["a", "b", "c"], pairs)

    assert report["n"] == 10 and report["correct"] == 6
    assert report["accuracy"] == pytest.approx(0.6)
    assert report["matrix"]["a"] == {"a": 3, "b": 1, "c": 0, mx.NO_PREDICTION: 0}
    assert report["matrix"]["b"] == {"a": 1, "b": 2, "c": 0, mx.NO_PREDICTION: 1}
    assert report["no_prediction_count"] == 1
    for label, expected in {"a": (3, 2, 1, 4), "b": (2, 1, 2, 5), "c": (1, 0, 1, 8)}.items():
        got = report["per_class"][label]
        assert (got["tp"], got["fp"], got["fn"], got["tn"]) == expected
    assert report["per_class"]["a"]["precision"] == pytest.approx(0.6)
    assert report["per_class"]["a"]["recall"] == pytest.approx(0.75)
        # a and b tie at 4 actual examples each; a tie goes to the first label in the given order
    assert report["majority_class"] == "a"
    assert report["majority_baseline_accuracy"] == pytest.approx(0.4)


def test_multiclass_averages() -> None:
    pairs = (
        [("a", "a")] * 3 + [("a", "b")] + [("b", "b")] * 2 + [("b", "a")] + [("b", mx.NO_PREDICTION)]
        + [("c", "c")] + [("c", "a")]
    )

    report = mx.multiclass_report(["a", "b", "c"], pairs)

    assert report["macro"]["precision"] == pytest.approx((0.6 + 2 / 3 + 1.0) / 3)
    assert report["macro"]["recall"] == pytest.approx((0.75 + 0.5 + 0.5) / 3)
    assert report["micro"]["precision"] == pytest.approx(6 / 9)
    assert report["micro"]["recall"] == pytest.approx(6 / 10)
    assert report["micro"]["f1"] == pytest.approx(12 / 19)


def test_macro_average_excludes_classes_without_examples() -> None:
    report = mx.multiclass_report(["a", "b", "c"], [("a", "a"), ("a", "a"), ("b", "b"), ("b", "a")])

    assert report["macro"]["classes"] == ["a", "b"]
    assert report["per_class"]["c"]["precision"] is None
    assert report["per_class"]["c"]["recall"] is None
    assert report["macro"]["precision"] == pytest.approx((2 / 3 + 1.0) / 2)


def test_macro_average_counts_never_predicted_classes_as_zero() -> None:
    report = mx.multiclass_report(["a", "b"], [("a", "b"), ("a", "b"), ("b", "b")])

    assert report["per_class"]["a"]["precision"] is None
    assert report["per_class"]["a"]["recall"] == 0.0
    assert report["macro"]["zero_division_classes"] == ["a"]
    assert report["macro"]["precision"] == pytest.approx((0.0 + 1 / 3) / 2)
    assert report["macro"]["f1"] == pytest.approx((0.0 + 0.5) / 2)


@pytest.mark.parametrize(
    "pairs",
    [[("zzz", "a")], [("a", "zzz")]],
    ids=["unknown-actual", "unknown-predicted"],
)
def test_multiclass_rejects_unknown_labels(pairs: list[tuple[str, str]]) -> None:
    with pytest.raises(ValueError):
        mx.multiclass_report(["a", "b"], pairs)


def test_summarize_latencies() -> None:
    result = mx.summarize_latencies([3.0, 1.0, 2.0, 10.0])

    assert result == {"n": 4, "min_ms": 1.0, "median_ms": 2.5, "max_ms": 10.0, "mean_ms": 4.0}


def test_summarize_latencies_without_samples() -> None:
    result = mx.summarize_latencies([])

    assert result["n"] == 0
    assert result["median_ms"] is None


# =============================================================================================
# 2. The labeled dataset
# =============================================================================================


def test_dataset_passes_its_own_validation() -> None:
    assert ds.validate_dataset() == []


def test_case_ids_are_unique_and_tags_are_valid() -> None:
    ids = [case.case_id for case in ds.CASES]

    assert len(ids) == len(set(ids))
    assert {case.tag for case in ds.CASES} <= set(ds.TAGS)


def test_labels_match_the_classifier_label_set() -> None:
    assert set(ds.CLASSIFICATION_LABELS) == {label.value for label in ClassificationLabel}
    for case in ds.CASES:
        for column in case.columns:
            assert column.label in ds.CLASSIFICATION_LABELS


def test_every_class_has_enough_examples() -> None:
    distribution = ds.dataset_summary()["label_distribution"]

    assert all(count >= 5 for count in distribution.values()), distribution


def test_every_category_has_positive_and_negative_examples() -> None:
    summary = ds.dataset_summary()

    for category in ds.COLUMN_LEVEL_CATEGORIES:
        positives = summary["detection_positives"][category]
        assert positives >= 2, category
        assert summary["columns"] - positives >= 1, category
    duplicates = summary["detection_positives"]["duplicate_rows"]
    assert duplicates >= 2
    assert summary["cases"] - duplicates >= 1


def test_every_tag_is_used() -> None:
    assert {case.tag for case in ds.CASES} == set(ds.TAGS)


def test_dataset_is_deterministic() -> None:
    assert ds.dataset_fingerprint() == ds.dataset_fingerprint()
    for case in ds.CASES:
        pd.testing.assert_frame_equal(ds.build_frame(case), ds.build_frame(case))


def test_fingerprint_changes_with_the_data_and_the_labels() -> None:
    original = ds.dataset_fingerprint()
    case = ds.CASES[0]
    first = case.columns[0].name
    changed_data = dict(case.data)
    changed_data[first] = (*case.data[first][:-1], "changed")

    with_new_value = (dataclasses.replace(case, data=changed_data), *ds.CASES[1:])
    with_new_label = (replace_column(case, first, label="unknown"), *ds.CASES[1:])

    assert ds.dataset_fingerprint(with_new_value) != original
    assert ds.dataset_fingerprint(with_new_label) != original


def test_building_frames_does_not_mutate_the_dataset() -> None:
    case = case_by_id("customers_clean")
    snapshot = copy.deepcopy(case.data)

    frame = ds.build_frame(case)
    frame.iloc[0, 0] = -1
    frame["customer_id"] = 0

    assert case.data == snapshot
    assert ds.build_frame(case).iloc[0, 0] == snapshot["customer_id"][0]


@pytest.mark.parametrize(
    ("make_bad_case", "keyword"),
    [
        pytest.param(
            lambda: dataclasses.replace(case_by_id("duplicates_present"), dataset_detections=()),
            "duplicate_rows",
            id="missing-duplicate-expectation",
        ),
        pytest.param(
            lambda: replace_column(case_by_id("customers_clean"), "customer_id", label="secret"),
            "label",
            id="unknown-label",
        ),
        pytest.param(
            lambda: replace_column(case_by_id("customers_clean"), "email", detections=()),
            "potential_email",
            id="missing-email-expectation",
        ),
    ],
)
def test_validation_catches_wrong_ground_truth(make_bad_case: Callable[[], ds.LabeledCase], keyword: str) -> None:
    problems = ds.validate_dataset((make_bad_case(),))

    assert any(keyword in problem for problem in problems), problems


def test_detection_config_only_when_rules_exist() -> None:
    assert ds.detection_config(case_by_id("customers_clean")) is None
    config = ds.detection_config(case_by_id("numeric_and_date_rules"))

    assert config is not None
    assert set(config.numeric_rules) == {"quantity", "weight_kg", "refund_amount"}
    assert set(config.date_rules) == {"received_date", "delivery_date"}


# =============================================================================================
# 3. The report built from real workflow outputs
# =============================================================================================


def test_report_structure(report: dict[str, Any]) -> None:
    assert set(report) == {
        "report_version", "run", "dataset", "deterministic", "mock", "llm",
        "human_review", "limitations", "how_to_read", "fingerprints",
    }
    assert report["run"]["generated_at_utc"] == FIXED_TIME.isoformat()
    assert report["run"]["repeats"] == 2
    assert report["dataset"]["columns"] == column_total()
    assert report["dataset"]["cases"] == len(ds.CASES)


def test_report_is_json_serializable(report: dict[str, Any]) -> None:
    text = json.dumps(report, sort_keys=True)

    assert json.loads(text)["fingerprints"] == report["fingerprints"]


def test_detection_counts_are_consistent(report: dict[str, Any]) -> None:
    metrics = report["deterministic"]["metrics"]
    cases = len(ds.CASES)

    assert metrics["units"] == column_total() * len(ds.COLUMN_LEVEL_CATEGORIES) + cases
    for category, result in metrics["per_category"].items():
        expected_units = cases if category in ds.DATASET_LEVEL_CATEGORIES else column_total()
        assert result["tp"] + result["fp"] + result["fn"] + result["tn"] == expected_units
        if category in ds.DATASET_LEVEL_CATEGORIES:
            expected_support = sum(1 for case in ds.CASES if category in case.dataset_detections)
        else:
            expected_support = sum(1 for case in ds.CASES for c in case.columns if category in c.detections)
        assert result["support"] == expected_support
    assert metrics["positives"] == sum(r["support"] for r in metrics["per_category"].values())


def test_detection_errors_match_the_counts(report: dict[str, Any]) -> None:
    metrics = report["deterministic"]["metrics"]
    per_category = metrics["per_category"].values()

    assert metrics["false_positives"] == sum(r["fp"] for r in per_category)
    assert metrics["false_negatives"] == sum(r["fn"] for r in per_category)
    assert len(metrics["errors"]) == metrics["false_positives"] + metrics["false_negatives"]
    assert {e["kind"] for e in metrics["errors"]} <= {"false_positive", "false_negative"}


def test_detection_anchor_cases_are_found(report: dict[str, Any]) -> None:
    """Behaviors already verified in earlier phases: the harness must not report them as missed."""
    anchors = [
        ("customers_clean", "email", "potential_email"),
        ("duplicates_present", ds.DATASET_LEVEL_UNIT, "duplicate_rows"),
        ("numeric_and_date_rules", "quantity", "numeric_rule_violation"),
        ("numeric_and_date_rules", "received_date", "invalid_date"),
        ("missing_values", "delivery_notes", "missing_values"),
    ]
    missed = {
        (e["case"], e["column"], e["category"])
        for e in report["deterministic"]["metrics"]["errors"]
        if e["kind"] == "false_negative"
    }

    for case_id, column, category in anchors:  # make sure the anchors are real positives first
        case = case_by_id(case_id)
        if column == ds.DATASET_LEVEL_UNIT:
            assert category in case.dataset_detections
        else:
            assert category in next(c for c in case.columns if c.name == column).detections
    for anchor in anchors:
        assert anchor not in missed, anchor


def test_no_finding_falls_outside_the_labeled_units(report: dict[str, Any]) -> None:
    assert report["deterministic"]["metrics"]["unmatched_findings"] == []


def test_results_are_deterministic_across_repeats(report: dict[str, Any]) -> None:
    assert report["deterministic"]["consistent_across_repeats"] is True
    assert report["mock"]["consistent_across_repeats"] is True
    assert report["deterministic"]["workflow_statuses"] == {"completed": len(ds.CASES)}


def test_two_builds_have_identical_fingerprints(report: dict[str, Any]) -> None:
    again = hn.build_report(repeats=2, warmup=False, clock=fixed_clock)

    assert again["fingerprints"] == report["fingerprints"]
    assert again["deterministic"]["metrics"] == report["deterministic"]["metrics"]
    assert again["mock"]["metrics"] == report["mock"]["metrics"]


def test_mock_section_is_labelled_and_counted(report: dict[str, Any]) -> None:
    mock = report["mock"]

    assert mock["provider"] == MockClassificationProvider().name
    assert "not an LLM" in mock["kind"]
    assert mock["metrics"]["n"] == column_total()
    assert mock["outcomes"] == {"ok": column_total()}
    assert mock["metrics"]["labels"] == list(ds.CLASSIFICATION_LABELS)


def test_sections_are_kept_separate(report: dict[str, Any]) -> None:
    assert "per_class" not in report["deterministic"]["metrics"]
    assert "per_category" not in report["mock"]["metrics"]
    assert report["llm"]["ran"] is False
    assert "mock" not in json.dumps(report["llm"]).lower().replace("not requested", "")


def test_latency_and_provider_calls_are_measured(report: dict[str, Any]) -> None:
    cases = len(ds.CASES)
    calls = report["mock"]["provider_calls"]

    assert report["deterministic"]["latency"]["n"] == cases * 2
    assert report["mock"]["workflow_latency"]["n"] == cases * 2
    assert calls["count"] == column_total()
    assert calls["failures"] == 0
    assert calls["latency"]["n"] == column_total()
    assert report["deterministic"]["latency"]["median_ms"] > 0


def test_limitations_are_computed_from_the_data(report: dict[str, Any]) -> None:
    notes = " ".join(report["limitations"])

    assert f"{report['dataset']['columns']} labeled columns" in notes
    assert "Class imbalance" in notes
    assert "not evidence about a language model" in notes
    assert "same author" in notes


# =============================================================================================
# 4. The genuine LLM section
# =============================================================================================


def test_llm_is_not_requested_by_default(report: dict[str, Any]) -> None:
    assert report["llm"]["ran"] is False
    assert "--llm" in report["llm"]["reason"]


def test_mock_provider_is_never_accepted_as_an_llm() -> None:
    result = hn.build_report(repeats=1, warmup=False, llm_provider=MockClassificationProvider(), clock=fixed_clock)

    assert result["llm"]["ran"] is False
    assert "not an LLM" in result["llm"]["reason"]


def test_llm_section_with_a_stub_provider() -> None:
    provider = ConstantProvider()

    result = hn.build_report(repeats=1, warmup=False, llm_provider=provider, clock=fixed_clock)

    llm = result["llm"]
    assert llm["ran"] is True and llm["provider"] == "llm-stub"
    assert provider.calls == column_total() == llm["provider_calls"]["count"]
    assert llm["metrics"]["no_prediction_count"] == 0
    assert llm["metrics"]["per_class"]["internal"]["recall"] == 1.0
    assert llm["metrics"]["per_class"]["personal"]["recall"] == 0.0
    assert llm["cost"]["measured"] is False and llm["token_usage"]["measured"] is False
    assert "unverified" in llm["kind"] and llm["reproducible"] is False


def test_llm_output_does_not_change_the_reproducible_fingerprint(report: dict[str, Any]) -> None:
    with_llm = hn.build_report(repeats=2, warmup=False, llm_provider=ConstantProvider(), clock=fixed_clock)

    assert with_llm["fingerprints"]["results"] == report["fingerprints"]["results"]
    assert with_llm["mock"]["metrics"] == report["mock"]["metrics"]


def test_llm_failures_are_reported_and_calls_are_bounded() -> None:
    provider = DownProvider()
    expected_calls = sum(min(3, len(case.columns)) for case in ds.CASES)  # the workflow's circuit breaker

    result = hn.build_report(repeats=1, warmup=False, llm_provider=provider, clock=fixed_clock)

    llm = result["llm"]
    assert provider.calls == expected_calls == llm["provider_calls"]["count"]
    assert llm["provider_calls"]["failures"] == expected_calls
    assert llm["metrics"]["no_prediction_count"] == column_total()
    assert llm["metrics"]["accuracy"] == 0.0
    assert llm["outcomes"]["provider_unavailable"] == expected_calls
    assert llm["outcomes"]["not_attempted"] == column_total() - expected_calls
    assert "ok" not in llm["outcomes"]


def test_provenance_mismatch_is_an_error() -> None:
    results, _ = hn._one_pass(ds.CASES, ColumnClassifier(MockClassificationProvider()))

    with pytest.raises(hn.ProvenanceError):
        hn.evaluate_classification(ds.CASES, results, expected_source=ClassificationSource.LLM)
    hn.evaluate_classification(ds.CASES, results, expected_source=ClassificationSource.MOCK)


def test_resolve_llm_provider_refuses_the_mock() -> None:
    provider, note = hn.resolve_llm_provider(Settings(_env_file=None, ai_provider="mock"))

    assert provider is None
    assert note is not None and "not an LLM" in note


def test_resolve_llm_provider_with_a_misconfigured_gemini() -> None:
    """No key and no model: nothing may be sent anywhere, and the harness must stay graceful."""
    provider, note = hn.resolve_llm_provider(Settings(_env_file=None, ai_provider="gemini"))

    if provider is None:
        assert note is not None and "could not be created" in note
        return
    result = hn.build_report(repeats=1, warmup=False, llm_provider=provider, clock=fixed_clock)
    assert result["llm"]["ran"] is True
    assert "ok" not in result["llm"]["outcomes"]
    assert result["llm"]["metrics"]["no_prediction_count"] == column_total()


def test_metered_provider_counts_calls_and_failures() -> None:
    good = hn.make_metered_provider(ConstantProvider())
    down = hn.make_metered_provider(DownProvider())

    good.classify(object())
    good.classify(object())
    with pytest.raises(ProviderUnavailableError):
        down.classify(object())

    assert (good.name, good.source) == ("llm-stub", ClassificationSource.LLM)
    assert len(good.latencies_ms) == 2 and good.failures == 0
    assert len(down.latencies_ms) == 1 and down.failures == 1
    assert all(sample >= 0 for sample in good.latencies_ms)


# =============================================================================================
# 5. Human review counts (persisted events only)
# =============================================================================================


@pytest.fixture()
def reviewed(client: TestClient, upload: Callable[..., Any], mixed_csv: bytes) -> dict[str, list[int]]:
    """Persisted review events through the real API: 4 events on 3 findings, one of them flipped."""
    assert upload(mixed_csv, "mixed.csv").status_code == 201
    by_category: dict[str, list[int]] = {}
    for item in client.get("/findings").json()["items"]:
        by_category.setdefault(item["category"], []).append(item["id"])
    email_id = by_category["potential_email"][0]
    first_missing, second_missing = by_category["missing_values"]
    for finding_id, decision in (
        (first_missing, "approved"),
        (second_missing, "rejected"),
        (email_id, "rejected"),
        (email_id, "approved"),
    ):
        assert client.post(f"/findings/{finding_id}/review", json={"decision": decision}).status_code == 201
    return by_category


def test_review_stats_on_an_empty_database(session: Session) -> None:
    stats = hn.collect_review_stats(session)

    assert stats["available"] is True and stats["reproducible"] is False
    assert stats["decision_events"] == 0 and stats["findings_total"] == 0
    assert stats["acceptance_rate"] is None and stats["acceptance_rate_ci95"] is None
    assert stats["by_category"] == {}


def test_review_stats_from_persisted_events(session: Session, reviewed: dict[str, list[int]]) -> None:
    stats = hn.collect_review_stats(session)

    assert stats["decision_events"] == 4
    assert stats["decision_events_by_decision"] == {"approved": 2, "rejected": 2}
    assert (stats["findings_total"], stats["findings_with_decisions"], stats["findings_pending_status"]) == (3, 3, 0)
    assert stats["latest_decisions"] == {"approved": 2, "rejected": 1}
    assert stats["acceptance_rate"] == pytest.approx(2 / 3)
    assert stats["re_reviewed_findings"] == 1 and stats["decision_changed_findings"] == 1
    assert stats["status_mismatch_findings"] == 0 and stats["orphan_events"] == 0
    assert stats["by_category"]["potential_email"] == {"approved": 1, "rejected": 0, "acceptance_rate": 1.0}
    assert stats["by_category"]["missing_values"] == {"approved": 1, "rejected": 1, "acceptance_rate": 0.5}
    assert set(stats["by_detection_method"]) == {"null_percent_threshold", "regex_email_pattern"}


def test_review_section_is_labelled_and_outside_the_fingerprint(
    session: Session, reviewed: dict[str, list[int]], report: dict[str, Any]
) -> None:
    with_review = hn.build_report(repeats=2, warmup=False, review_session=session, clock=fixed_clock)

    assert with_review["human_review"]["available"] is True
    assert with_review["human_review"]["reproducible"] is False
    assert "not ground" in with_review["human_review"]["note"]
    assert with_review["fingerprints"]["results"] == report["fingerprints"]["results"]
    assert report["human_review"]["available"] is False


def test_unreadable_database_is_reported() -> None:
    with Session(create_engine("sqlite://")) as empty_session:  # no tables at all
        result = hn.build_report(repeats=1, warmup=False, review_session=empty_session, clock=fixed_clock)

    assert result["human_review"]["available"] is False
    assert "could not be read" in result["human_review"]["reason"]


# =============================================================================================
# 6. Markdown report and the command line
# =============================================================================================


def test_markdown_contains_every_section(report: dict[str, Any]) -> None:
    text = hn.render_markdown(report)

    for heading in (
        "# VerityGrid evaluation report",
        "## Dataset",
        "## Deterministic detectors",
        "## Mock classification provider",
        "## Genuine LLM provider",
        "## Human review events",
        "## Limitations",
        "## How to read this report",
    ):
        assert heading in text
    assert report["fingerprints"]["results"] in text
    assert "Confusion matrix" in text


def test_markdown_labels_mock_results(report: dict[str, Any]) -> None:
    text = hn.render_markdown(report)

    assert "MOCK results" in text and "not an LLM" in text
    assert "say nothing about how a language model performs" in text
    assert "Not run." in text  # the LLM section explains why it did not run


def test_cli_writes_both_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = hn.main(["--output-dir", str(tmp_path), "--repeats", "1", "--skip-review"])

    assert code == 0
    data = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "evaluation_report.md").read_text(encoding="utf-8")
    assert data["human_review"]["available"] is False
    assert data["llm"]["ran"] is False
    assert data["fingerprints"]["results"] in markdown
    output = capsys.readouterr().out
    assert data["fingerprints"]["results"] in output
    assert "genuine LLM: not run" in output


def test_cli_does_not_create_a_missing_database(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"

    code = hn.main(["--output-dir", str(tmp_path), "--repeats", "1", "--database-url", f"sqlite:///{missing}"])

    assert code == 0
    assert not missing.exists()
    data = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    assert data["human_review"]["available"] is False
    assert "not found" in data["human_review"]["reason"]


def test_cli_rejects_bad_repeat_counts() -> None:
    with pytest.raises(SystemExit) as excinfo:
        hn.main(["--repeats", "0"])

    assert excinfo.value.code == 2


# =============================================================================================
# 7. Hygiene
# =============================================================================================


def evaluation_sources() -> list[Path]:
    return sorted((ROOT / "app" / "evaluation").glob("*.py"))


def imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_evaluation_code_makes_no_network_calls() -> None:
    network = {"requests", "httpx", "httpx2", "urllib", "urllib3", "socket", "aiohttp", "google"}

    assert len(evaluation_sources()) == 4
    for path in evaluation_sources():
        assert not imported_modules(path) & network, path.name


def test_evaluation_code_does_not_use_the_ui() -> None:
    for path in evaluation_sources():
        assert not imported_modules(path) & {"ui", "streamlit"}, path.name