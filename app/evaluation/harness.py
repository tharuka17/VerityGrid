"""Reproducible evaluation harness for VerityGrid.

Run from the project root:
    python -m app.evaluation.harness                  # deterministic + mock, writes reports/
    python -m app.evaluation.harness --skip-review    # also ignore the local database
    python -m app.evaluation.harness --llm            # opt in to a genuine LLM run (paid calls)

Everything is computed from real workflow outputs on the fixed dataset in dataset.py:
  - "deterministic": findings from run_detectors, scored against the labeled expectations.
  - "mock": the keyword placeholder provider, scored against the labeled classes. NOT an LLM.
  - "llm": a genuine LLM provider, only when explicitly requested and only if the provider really
    is an LLM. A single, non-reproducible run.
  - "human_review": counts read from persisted ReviewDecision rows. Operational data, not ground
    truth, not reproducible, and kept out of the benchmark.

Nothing is hard-coded: no metric value appears in this file. Results are processed through the
real run_scan_workflow, so latency covers the actual production path.
"""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.classifier import ColumnClassifier, ResultStatus
from app.ai.factory import get_classification_provider
from app.ai.mock_provider import MockClassificationProvider
from app.ai.provider import ClassificationProvider, ClassificationSource
from app.core.config import Settings
from app.db.models import Finding, ReviewDecision
from app.db.session import build_engine
from app.evaluation import dataset as ds
from app.evaluation import metrics as mx
from app.workflows.scan_graph import WorkflowOptions, WorkflowResult, WorkflowStatus, run_scan_workflow

REPORT_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_DATABASE_URL = "sqlite:///./veritygrid.db"
REPORT_JSON = "evaluation_report.json"
REPORT_MARKDOWN = "evaluation_report.md"

MOCK_KIND = "MOCK: a rule-based placeholder, not an LLM"
LLM_KIND = "genuine LLM provider output (unverified; a single non-reproducible run)"
COST_REASON = (
    "Cost is not measured. The provider adapter does not record token usage and no price is "
    "configured here, so no figure is estimated."
)
TOKEN_REASON = "Token usage is not measured: the adapter does not expose usage metadata."

HOW_TO_READ = [
    "Support: how many actual positives (or examples of that class) exist in the fixed dataset. Small supports make every rate unstable.",
    "TP / FP / FN / TN: true positive = flagged and should be; false positive = flagged but should not be; false negative = should be flagged but was missed; true negative = correctly left alone.",
    "Precision = TP / (TP + FP): of everything flagged, the share that was right. n/a means nothing was flagged.",
    "Recall = TP / (TP + FN): of everything that should be flagged, the share that was found. n/a means there were no actual positives.",
    "F1 = 2*TP / (2*TP + FP + FN): one number balancing precision and recall. n/a when either is undefined.",
    "95% interval (Wilson): the range of true rates consistent with the observed counts. A wide interval means little evidence either way.",
    "Micro average: pools all counts before computing the rate, so large classes dominate. Macro average: the plain mean over classes that have examples, so small classes count equally; a class with examples that is never predicted counts as 0.",
    "Confusion matrix: rows are the true class, columns the predicted class. The diagonal is correct; the no_prediction column means no valid answer was produced (failure or not attempted).",
    "Accuracy and the majority baseline: accuracy is the share of correct columns; the baseline is what always answering the largest class would score. Accuracy only means something when it clearly beats the baseline.",
    "Latency: wall-clock milliseconds for one workflow run on one synthetic dataset, after a discarded warm-up pass. It is machine dependent and not part of the reproducible results.",
    "Provider calls: number of classify invocations. The Gemini adapter may retry internally, so HTTP requests can exceed this number, and per-call latency includes any retries.",
    "Cost and tokens: reported as not measured; nothing is estimated.",
    "Reviewer acceptance rate: the share of reviewed findings whose latest persisted decision is 'approved'. It counts what reviewers did, not what is correct, covers only flagged items (so it cannot see misses) and is not comparable to recall.",
    "Fingerprints: the dataset fingerprint changes when any value or label changes. The results fingerprint covers the dataset, the deterministic metrics and the mock metrics; two runs on the same code and dataset must print the same one.",
]


class ProvenanceError(RuntimeError):
    """A result's source does not match the section it was about to be scored in."""


# --- Provider metering -------------------------------------------------------------------------


def make_metered_provider(inner: ClassificationProvider) -> ClassificationProvider:
    """Wrap a provider so every classify() call is counted and timed.

    The class is created per provider so that `name` and `source` are class attributes, exactly
    like every other provider, whatever the base class requires.
    """

    class _Metered(ClassificationProvider):
        name = inner.name
        source = inner.source

        def __init__(self) -> None:
            self.inner = inner
            self.latencies_ms: list[float] = []
            self.failures = 0

        def classify(self, metadata: Any) -> Any:
            started = time.perf_counter()
            try:
                return inner.classify(metadata)
            except BaseException:
                self.failures += 1
                raise
            finally:
                self.latencies_ms.append((time.perf_counter() - started) * 1000.0)

    return _Metered()


# --- Running the workflow ---------------------------------------------------------------------


def _signature(result: WorkflowResult) -> tuple[Any, ...]:
    findings = tuple((f.column_name, f.category.value, f.severity.value) for f in result.findings)
    classes = tuple((c.column_ref, c.label.value, c.status.value) for c in result.classifications)
    return (result.status.value, findings, classes)


def _one_pass(
    cases: Sequence[ds.LabeledCase], classifier: ColumnClassifier | None
) -> tuple[dict[str, WorkflowResult], dict[str, float]]:
    results: dict[str, WorkflowResult] = {}
    run_ms: dict[str, float] = {}
    for case in cases:
        frame = ds.build_frame(case)
        options = WorkflowOptions(classify=classifier is not None, detection_config=ds.detection_config(case))
        started = time.perf_counter()
        result = run_scan_workflow(frame, dataset_name=case.case_id, options=options, classifier=classifier)
        run_ms[case.case_id] = (time.perf_counter() - started) * 1000.0
        results[case.case_id] = result
    failed = [case_id for case_id, result in results.items() if result.status is WorkflowStatus.FAILED]
    if failed:
        raise RuntimeError(f"The workflow failed for: {', '.join(failed)}. The evaluation is not valid.")
    return results, run_ms


def _evaluate_mode(
    cases: Sequence[ds.LabeledCase],
    provider: ClassificationProvider | None,
    *,
    repeats: int,
    warmup: bool,
) -> dict[str, Any]:
    """Score pass (metered) plus extra timing passes; every pass must give identical results."""

    def classifier_for(p: ClassificationProvider | None) -> ColumnClassifier | None:
        return ColumnClassifier(p) if p is not None else None

    if warmup:
        _one_pass(cases, classifier_for(provider))  # discarded: imports and first-call costs
    metered = make_metered_provider(provider) if provider is not None else None
    results, first_ms = _one_pass(cases, classifier_for(metered if metered is not None else None))
    signature = {case_id: _signature(result) for case_id, result in results.items()}
    run_ms = list(first_ms.values())
    consistent = True
    for _ in range(repeats - 1):
        other, other_ms = _one_pass(cases, classifier_for(provider))
        run_ms.extend(other_ms.values())
        if {case_id: _signature(result) for case_id, result in other.items()} != signature:
            consistent = False
    return {"results": results, "run_ms": run_ms, "consistent": consistent, "metered": metered}


# --- Scoring --------------------------------------------------------------------------------------


def _tally(actual: bool, predicted: bool, counts: dict[str, int]) -> str | None:
    if actual and predicted:
        counts["tp"] += 1
        return None
    if predicted:
        counts["fp"] += 1
        return "false_positive"
    if actual:
        counts["fn"] += 1
        return "false_negative"
    counts["tn"] += 1
    return None


def evaluate_detection(cases: Sequence[ds.LabeledCase], results: Mapping[str, WorkflowResult]) -> dict[str, Any]:
    """Score deterministic findings. A unit is one (column, category) pair, or one (dataset, duplicate_rows)."""
    zero = lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}  # noqa: E731
    per_category = {category: zero() for category in ds.ALL_CATEGORIES}
    by_tag = {tag: zero() for tag in ds.TAGS}
    errors: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for case in cases:
        predicted = {(f.column_name, f.category.value) for f in results[case.case_id].findings}
        known: set[tuple[str | None, str]] = set()

        def score(column: str | None, category: str, actual: bool) -> None:
            known.add((column, category))
            kind = _tally(actual, (column, category) in predicted, per_category[category])
            _tally(actual, (column, category) in predicted, by_tag[case.tag])
            if kind:
                errors.append(
                    {
                        "case": case.case_id,
                        "tag": case.tag,
                        "column": column if column is not None else ds.DATASET_LEVEL_UNIT,
                        "category": category,
                        "kind": kind,
                    }
                )

        for column in case.columns:
            for category in ds.COLUMN_LEVEL_CATEGORIES:
                score(column.name, category, category in column.detections)
        for category in ds.DATASET_LEVEL_CATEGORIES:
            score(None, category, category in case.dataset_detections)
        for column_name, category in sorted(predicted - known, key=lambda key: (key[0] or "", key[1])):
            unmatched.append({"case": case.case_id, "column": column_name, "category": category})

    reports = {category: mx.binary_report(**counts) for category, counts in per_category.items()}
    return {
        "units": sum(report["units"] for report in reports.values()),
        "positives": sum(report["support"] for report in reports.values()),
        "per_category": reports,
        "micro": mx.micro_average(reports.values()),
        "macro": mx.macro_average(reports),
        "by_tag": {tag: mx.binary_report(**counts) for tag, counts in by_tag.items()},
        "errors": errors,
        "false_positives": sum(1 for error in errors if error["kind"] == "false_positive"),
        "false_negatives": sum(1 for error in errors if error["kind"] == "false_negative"),
        "unmatched_findings": unmatched,
    }


def evaluate_classification(
    cases: Sequence[ds.LabeledCase],
    results: Mapping[str, WorkflowResult],
    *,
    expected_source: ClassificationSource,
) -> dict[str, Any]:
    """Score classification labels. Every successful result must carry `expected_source`."""
    pairs: list[tuple[str, str]] = []
    outcomes: Counter[str] = Counter()
    misclassified: list[dict[str, str]] = []
    for case in cases:
        by_ref = {c.column_ref: c for c in results[case.case_id].classifications}
        for position, column in enumerate(case.columns, start=1):
            result = by_ref.get(f"col_{position}")
            if result is None:
                predicted, outcome = mx.NO_PREDICTION, "not_attempted"
            elif result.status is ResultStatus.OK:
                if result.source is not expected_source:
                    raise ProvenanceError(
                        f"Expected '{expected_source.value}' results but found '{result.source.value}'."
                    )
                predicted, outcome = result.label.value, "ok"
            else:
                predicted, outcome = mx.NO_PREDICTION, result.status.value
            outcomes[outcome] += 1
            pairs.append((column.label, predicted))
            if predicted != column.label:
                misclassified.append(
                    {"case": case.case_id, "column": column.name, "actual": column.label, "predicted": predicted}
                )
    return {
        "metrics": mx.multiclass_report(ds.CLASSIFICATION_LABELS, pairs),
        "outcomes": dict(sorted(outcomes.items())),
        "misclassified": misclassified,
    }


# --- Human review (persisted events only) ----------------------------------------------------------


def _group_rates(groups: Mapping[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "approved": counts["approved"],
            "rejected": counts["rejected"],
            "acceptance_rate": mx.safe_ratio(counts["approved"], counts["approved"] + counts["rejected"]),
        }
        for key, counts in sorted(groups.items())
    }


def collect_review_stats(session: Session) -> dict[str, Any]:
    """Counts from persisted ReviewDecision rows. Nothing is inferred and nothing is written."""
    events = session.execute(
        select(ReviewDecision.id, ReviewDecision.finding_id, ReviewDecision.decision).order_by(ReviewDecision.id)
    ).all()
    findings = session.execute(
        select(Finding.id, Finding.category, Finding.detection_method, Finding.review_status)
    ).all()
    by_id = {row.id: row for row in findings}

    event_counts: Counter[str] = Counter()
    per_finding: Counter[int] = Counter()
    first: dict[int, str] = {}
    latest: dict[int, str] = {}
    for event in events:
        decision = str(event.decision)
        event_counts[decision] += 1
        per_finding[event.finding_id] += 1
        first.setdefault(event.finding_id, decision)
        latest[event.finding_id] = decision

    latest_counts = Counter(latest.values())
    decided = latest_counts["approved"] + latest_counts["rejected"]
    by_category: dict[str, Counter[str]] = {}
    by_method: dict[str, Counter[str]] = {}
    for finding_id, decision in latest.items():
        row = by_id.get(finding_id)
        if row is None:
            continue
        by_category.setdefault(str(row.category), Counter())[decision] += 1
        by_method.setdefault(str(row.detection_method), Counter())[decision] += 1

    return {
        "available": True,
        "reproducible": False,
        "decision_events": len(events),
        "decision_events_by_decision": dict(sorted(event_counts.items())),
        "findings_total": len(findings),
        "findings_with_decisions": len(latest),
        "findings_pending_status": sum(1 for row in findings if row.review_status == "pending"),
        "latest_decisions": {"approved": latest_counts["approved"], "rejected": latest_counts["rejected"]},
        "acceptance_rate": mx.safe_ratio(latest_counts["approved"], decided),
        "acceptance_rate_ci95": (
            None if decided == 0 else list(mx.wilson_interval(latest_counts["approved"], decided) or [])
        ),
        "re_reviewed_findings": sum(1 for count in per_finding.values() if count > 1),
        "decision_changed_findings": sum(1 for fid in latest if first[fid] != latest[fid]),
        "status_mismatch_findings": sum(
            1 for fid, decision in latest.items() if fid in by_id and by_id[fid].review_status != decision
        ),
        "orphan_events": sum(1 for fid in latest if fid not in by_id),
        "by_category": _group_rates(by_category),
        "by_detection_method": _group_rates(by_method),
        "note": "Operational data from the local database. It may include test decisions, is not ground "
        "truth, is not reproducible, and is not part of the benchmark above.",
    }


# --- Report assembly ---------------------------------------------------------------------------------


def _limitations(summary: Mapping[str, Any], detection: Mapping[str, Any], mock_metrics: Mapping[str, Any]) -> list[str]:
    columns, cases = summary["columns"], summary["cases"]
    notes = [
        f"Small sample: {columns} labeled columns in {cases} synthetic cases, plus {cases} dataset-level "
        "duplicate checks. One more or one fewer correct answer moves the percentages a lot; the 95% "
        "intervals show how wide the uncertainty is."
    ]
    small = [f"{label} ({count})" for label, count in summary["label_distribution"].items() if count < 10]
    if small:
        notes.append(
            "Classes with fewer than 10 labeled columns: " + ", ".join(small)
            + ". Their per-class precision and recall are especially unstable."
        )
    thin = [f"{category} ({count})" for category, count in summary["detection_positives"].items() if count < 5]
    if thin:
        notes.append(
            "Detection categories with fewer than 5 positive examples: " + ", ".join(thin)
            + ". Their recall rests on a handful of cases."
        )
    baseline = mock_metrics["majority_baseline_accuracy"]
    if baseline is not None:
        majority = mock_metrics["majority_class"]
        notes.append(
            f"Class imbalance: '{majority}' is the largest class ({baseline:.1%} of columns), so a classifier "
            f"that always answered '{majority}' would score {baseline:.1%} accuracy. Use per-class recall and "
            "the macro averages, not accuracy alone."
        )
    notes.append(
        f"Detection is also imbalanced: {detection['positives']} of {detection['units']} (column or dataset, "
        "category) units are positives and the rest are negatives. Accuracy would look high even for a detector "
        "that flagged nothing, so this report does not use it."
    )
    notes += [
        "The labels, the detectors and the mock provider were written by the same author: there is no "
        "independent annotator, no inter-annotator agreement, and the dataset was not held out from development. "
        "The results describe how the implementation behaves on cases its author could anticipate; they do not "
        "estimate real-world performance.",
        "The data is synthetic: values, formats and the class mix are invented and do not represent any real organisation.",
        "Mock results measure a keyword matcher against these labels. They are not evidence about a language model.",
        "Genuine LLM results, when present, come from a single run of one model on this small dataset with no "
        "temperature set, so repeating the run may give different answers.",
        "Timings are wall-clock measurements on one machine and vary between runs.",
        "Human review counts come from the local database, may include test decisions, and are not ground truth.",
    ]
    return notes


def _classification_section(
    cases: Sequence[ds.LabeledCase],
    run: Mapping[str, Any],
    *,
    provider: ClassificationProvider,
    expected_source: ClassificationSource,
    kind: str,
    repeats: int,
) -> dict[str, Any]:
    evaluation = evaluate_classification(cases, run["results"], expected_source=expected_source)
    metered = run["metered"]
    return {
        "ran": True,
        "provider": provider.name,
        "kind": kind,
        "metrics": evaluation["metrics"],
        "outcomes": evaluation["outcomes"],
        "misclassified": evaluation["misclassified"],
        "provider_calls": {
            "count": len(metered.latencies_ms),
            "failures": metered.failures,
            "latency": mx.summarize_latencies(metered.latencies_ms),
            "note": "Counts classify() invocations in the scoring pass. The adapter may retry internally, so "
            "HTTP requests can exceed this number; per-call latency includes any retries.",
        },
        "workflow_latency": mx.summarize_latencies(run["run_ms"]),
        "repeats": repeats,
        "consistent_across_repeats": run["consistent"],
    }


def build_report(
    *,
    repeats: int = 5,
    warmup: bool = True,
    llm_provider: ClassificationProvider | None = None,
    llm_note: str | None = None,
    review_session: Session | None = None,
    review_note: str | None = None,
    cases: Sequence[ds.LabeledCase] = ds.CASES,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    problems = ds.validate_dataset(cases)
    if problems:
        raise ValueError("The evaluation dataset is inconsistent: " + "; ".join(problems))
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    summary = ds.dataset_summary(cases)
    deterministic = _evaluate_mode(cases, None, repeats=repeats, warmup=warmup)
    detection = evaluate_detection(cases, deterministic["results"])
    statuses = Counter(result.status.value for result in deterministic["results"].values())

    mock_provider = MockClassificationProvider()
    mock_run = _evaluate_mode(cases, mock_provider, repeats=repeats, warmup=warmup)
    mock = _classification_section(
        cases, mock_run, provider=mock_provider, expected_source=ClassificationSource.MOCK,
        kind=MOCK_KIND, repeats=repeats,
    )

    if llm_provider is None:
        llm: dict[str, Any] = {
            "ran": False,
            "reason": llm_note or "Not requested. Run with --llm to measure a real model; that makes paid provider calls.",
        }
    elif llm_provider.source is not ClassificationSource.LLM:
        llm = {
            "ran": False,
            "reason": f"The provider '{llm_provider.name}' is not an LLM (source: {llm_provider.source.value}), so it was not run.",
        }
    else:
        llm_run = _evaluate_mode(cases, llm_provider, repeats=1, warmup=False)
        llm = _classification_section(
            cases, llm_run, provider=llm_provider, expected_source=ClassificationSource.LLM,
            kind=LLM_KIND, repeats=1,
        )
        llm["reproducible"] = False
        llm["cost"] = {"measured": False, "reason": COST_REASON}
        llm["token_usage"] = {"measured": False, "reason": TOKEN_REASON}

    if review_session is None:
        review: dict[str, Any] = {
            "available": False,
            "reproducible": False,
            "reason": review_note or "Not collected.",
        }
    else:
        try:
            review = collect_review_stats(review_session)
        except SQLAlchemyError as error:
            review = {
                "available": False,
                "reproducible": False,
                "reason": f"The database could not be read ({type(error).__name__}).",
            }

    results_payload = {
        "dataset": summary["fingerprint"],
        "deterministic": detection,
        "mock": {"metrics": mock["metrics"], "outcomes": mock["outcomes"]},
    }
    results_fingerprint = hashlib.sha256(
        json.dumps(results_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    return {
        "report_version": REPORT_VERSION,
        "run": {
            "generated_at_utc": clock().isoformat(),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "repeats": repeats,
            "warmup_pass": warmup,
            "command": "python -m app.evaluation.harness",
        },
        "dataset": {**summary, "annotation_guidelines": ds.ANNOTATION_GUIDELINES},
        "deterministic": {
            "metrics": detection,
            "latency": mx.summarize_latencies(deterministic["run_ms"]),
            "latency_scope": "One workflow run on one synthetic dataset (profile, detectors, policy).",
            "workflow_statuses": dict(sorted(statuses.items())),
            "repeats": repeats,
            "consistent_across_repeats": deterministic["consistent"],
        },
        "mock": mock,
        "llm": llm,
        "human_review": review,
        "limitations": _limitations(summary, detection, mock["metrics"]),
        "how_to_read": HOW_TO_READ,
        "fingerprints": {
            "dataset": summary["fingerprint"],
            "results": results_fingerprint,
            "scope": "dataset + deterministic metrics + mock metrics; excludes timings, timestamps, "
            "genuine LLM output and human review data",
        },
    }


# --- Markdown rendering --------------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_ci(interval: Sequence[float] | None) -> str:
    return "n/a" if not interval else f"[{interval[0]:.2f}, {interval[1]:.2f}]"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(cell if isinstance(cell, str) else _fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def _rate_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    return [
        [name, r["support"], r["tp"], r["fp"], r["fn"], r["tn"], r["precision"], r["recall"], r["f1"], _fmt_ci(r["recall_ci95"])]
        for name, r in reports.items()
    ]


def _latency_table(latency: Mapping[str, Any]) -> str:
    return _table(
        ["Runs", "Min ms", "Median ms", "Max ms", "Mean ms"],
        [[latency["n"], latency["min_ms"], latency["median_ms"], latency["max_ms"], latency["mean_ms"]]],
    )


def _classification_markdown(section: Mapping[str, Any], banner: str) -> list[str]:
    metrics = section["metrics"]
    lines = [banner, "", f"Provider: `{section['provider']}`. Labeled columns scored: {metrics['n']}.", ""]
    lines += [
        _table(
            ["Class", "Support", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Recall 95% CI"],
            _rate_rows(metrics["per_class"]),
        ),
        "",
        f"Accuracy: {_fmt(metrics['accuracy'])} (95% CI {_fmt_ci(metrics['accuracy_ci95'])}). "
        f"Majority-class baseline ('{metrics['majority_class']}'): {_fmt(metrics['majority_baseline_accuracy'])}. "
        f"No valid prediction: {metrics['no_prediction_count']}.",
        f"Macro: precision {_fmt(metrics['macro']['precision'])}, recall {_fmt(metrics['macro']['recall'])}, "
        f"F1 {_fmt(metrics['macro']['f1'])} (classes with examples: {', '.join(metrics['macro']['classes']) or 'none'}; "
        f"counted as 0 because never predicted: {', '.join(metrics['macro']['zero_division_classes']) or 'none'}).",
        f"Micro: precision {_fmt(metrics['micro']['precision'])}, recall {_fmt(metrics['micro']['recall'])}, "
        f"F1 {_fmt(metrics['micro']['f1'])}.",
        "",
        "Confusion matrix (rows: true class, columns: predicted):",
        "",
    ]
    headers = ["Actual \\ Predicted", *metrics["predicted_columns"], "Total"]
    rows = [[actual, *[row[p] for p in metrics["predicted_columns"]], sum(row.values())] for actual, row in metrics["matrix"].items()]
    lines += [_table(headers, rows), "", f"Outcomes: {json.dumps(section['outcomes'], sort_keys=True)}", ""]
    wrong = section["misclassified"]
    lines.append(f"Misclassified or unanswered columns: {len(wrong)}")
    if wrong:
        lines += ["", _table(["Case", "Column", "True", "Predicted"], [[w["case"], w["column"], w["actual"], w["predicted"]] for w in wrong])]
    calls = section["provider_calls"]
    lines += [
        "",
        f"Provider calls: {calls['count']} (failures: {calls['failures']}). {calls['note']}",
        "",
        "Provider call latency:",
        "",
        _latency_table(calls["latency"]),
        "",
        "Workflow latency:",
        "",
        _latency_table(section["workflow_latency"]),
        "",
        f"Identical results across {section['repeats']} pass(es): {_fmt(section['consistent_across_repeats'])}.",
        "",
    ]
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    run, data, det = report["run"], report["dataset"], report["deterministic"]
    metrics = det["metrics"]
    lines = [
        "# VerityGrid evaluation report",
        "",
        f"Generated: {run['generated_at_utc']} | Python {run['python']} | pandas {run['pandas']} | passes: {run['repeats']}",
        f"Dataset fingerprint: `{report['fingerprints']['dataset']}`",
        f"Results fingerprint: `{report['fingerprints']['results']}`",
        f"Fingerprint scope: {report['fingerprints']['scope']}.",
        "",
        "Three sources of evidence follow and are never combined: deterministic detectors, the mock provider "
        "(a placeholder, not an LLM) and, only if it was run, a genuine LLM provider.",
        "",
        "## Dataset",
        "",
        f"{data['cases']} synthetic cases, {data['columns']} labeled columns, {data['rows_total']} rows in total.",
        f"Cases by tag: {json.dumps(data['cases_by_tag'], sort_keys=True)}.",
        "",
        "Labeled columns per class:",
        "",
        _table(["Class", "Columns", "Definition"], [[label, count, data["annotation_guidelines"].get(label, "")] for label, count in data["label_distribution"].items()]),
        "",
        "Expected detections (actual positives) per category:",
        "",
        _table(["Category", "Positive units"], [[category, count] for category, count in data["detection_positives"].items()]),
        "",
        f"Detection rule for ground truth: {data['annotation_guidelines']['detections']}",
        "",
        "## Deterministic detectors",
        "",
        f"Units scored: {metrics['units']} ({metrics['positives']} actual positives). "
        f"False positives: {metrics['false_positives']}. False negatives: {metrics['false_negatives']}.",
        "",
        _table(
            ["Category", "Support", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Recall 95% CI"],
            _rate_rows(metrics["per_category"]),
        ),
        "",
        f"Micro: precision {_fmt(metrics['micro']['precision'])}, recall {_fmt(metrics['micro']['recall'])}, F1 {_fmt(metrics['micro']['f1'])}. "
        f"Macro: precision {_fmt(metrics['macro']['precision'])}, recall {_fmt(metrics['macro']['recall'])}, F1 {_fmt(metrics['macro']['f1'])}.",
        "",
        "By case tag:",
        "",
        _table(
            ["Tag", "Units", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1"],
            [[tag, r["units"], r["support"], r["tp"], r["fp"], r["fn"], r["precision"], r["recall"], r["f1"]] for tag, r in metrics["by_tag"].items()],
        ),
        "",
        f"Workflow statuses: {json.dumps(det['workflow_statuses'], sort_keys=True)}. "
        f"Identical results across {det['repeats']} pass(es): {_fmt(det['consistent_across_repeats'])}.",
        f"Findings outside the labeled units: {len(metrics['unmatched_findings'])}.",
        "",
        "Errors (every false positive and false negative):",
        "",
    ]
    if metrics["errors"]:
        lines.append(_table(["Case", "Tag", "Column", "Category", "Kind"], [[e["case"], e["tag"], e["column"], e["category"], e["kind"]] for e in metrics["errors"]]))
    else:
        lines.append("None.")
    lines += ["", f"Latency. {det['latency_scope']}", "", _latency_table(det["latency"]), ""]

    lines += ["## Mock classification provider", ""]
    lines += _classification_markdown(
        report["mock"],
        "**These are MOCK results (a keyword placeholder, not an LLM). They say nothing about how a language model performs.**",
    )

    lines += ["## Genuine LLM provider", ""]
    llm = report["llm"]
    if not llm["ran"]:
        lines += [f"Not run. {llm['reason']}", ""]
    else:
        lines += _classification_markdown(
            llm, "**Genuine LLM output from a single run. It is unverified, not reproducible and says nothing beyond this small dataset.**"
        )
        lines += [f"Cost: {llm['cost']['reason']}", f"Tokens: {llm['token_usage']['reason']}", ""]

    lines += ["## Human review events", ""]
    review = report["human_review"]
    if not review["available"]:
        lines += [f"Not available. {review['reason']}", ""]
    else:
        latest = review["latest_decisions"]
        lines += [
            review["note"],
            "",
            f"Decision events: {review['decision_events']} ({json.dumps(review['decision_events_by_decision'], sort_keys=True)}). "
            f"Findings stored: {review['findings_total']}; with decisions: {review['findings_with_decisions']}; "
            f"still pending: {review['findings_pending_status']}.",
            f"Latest decision per finding: approved {latest['approved']}, rejected {latest['rejected']}. "
            f"Acceptance rate: {_fmt(review['acceptance_rate'])} (95% CI {_fmt_ci(review['acceptance_rate_ci95'])}).",
            f"Re-reviewed findings: {review['re_reviewed_findings']}; decision changed: {review['decision_changed_findings']}; "
            f"status mismatches: {review['status_mismatch_findings']}; orphan events: {review['orphan_events']}.",
            "",
            "By finding category (latest decisions):",
            "",
            _table(["Category", "Approved", "Rejected", "Acceptance"], [[k, v["approved"], v["rejected"], v["acceptance_rate"]] for k, v in review["by_category"].items()]),
            "",
            "By detection method (latest decisions):",
            "",
            _table(["Method", "Approved", "Rejected", "Acceptance"], [[k, v["approved"], v["rejected"], v["acceptance_rate"]] for k, v in review["by_detection_method"].items()]),
            "",
        ]

    lines += ["## Limitations", ""] + [f"- {note}" for note in report["limitations"]]
    lines += ["", "## How to read this report", ""] + [f"- {item}" for item in report["how_to_read"]]
    return "\n".join(lines) + "\n"


# --- Command line ---------------------------------------------------------------------------------------


def resolve_llm_provider(settings: Settings) -> tuple[ClassificationProvider | None, str | None]:
    """The configured provider if (and only if) it is a genuine LLM; otherwise None and the reason."""
    try:
        provider = get_classification_provider(settings)
    except Exception as error:
        return None, f"The configured provider could not be created ({type(error).__name__})."
    if provider.source is not ClassificationSource.LLM:
        return None, (
            f"The configured provider ('{provider.name}') is not an LLM (source: {provider.source.value}). "
            "Set AI_PROVIDER=gemini and the Gemini settings in .env to run a genuine LLM evaluation."
        )
    return provider, None


def open_review_session(database_url: str) -> tuple[Session | None, str | None]:
    """Open the database read-only in practice (only SELECTs run); never create a missing SQLite file."""
    try:
        url = make_url(database_url)
        if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
            if not Path(url.database).exists():
                return None, "The database file was not found, so no review events could be read."
        return Session(build_engine(database_url)), None
    except Exception as error:
        return None, f"The database could not be opened ({type(error).__name__})."


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.harness",
        description="Generate the reproducible VerityGrid evaluation report from the fixed labeled dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where to write the report files")
    parser.add_argument("--repeats", type=int, default=5, help="workflow passes used for timing and consistency (1-50)")
    parser.add_argument("--skip-review", action="store_true", help="do not read human review events from the database")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL, help="database to read review events from")
    parser.add_argument("--llm", action="store_true", help="also evaluate the configured genuine LLM (paid calls)")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt for --llm")
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 50:
        parser.error("--repeats must be between 1 and 50")
    return args


def _confirm_llm(provider: ClassificationProvider, assume_yes: bool) -> bool:
    calls = sum(len(case.columns) for case in ds.CASES)
    print(
        f"About to make up to {calls} paid calls through '{provider.name}'. Sanitized column metadata of the "
        "synthetic dataset (never cell values) is sent to the provider."
    )
    if assume_yes:
        return True
    try:
        return input("Type 'yes' to continue: ").strip().lower() == "yes"
    except EOFError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    llm_provider: ClassificationProvider | None = None
    llm_note: str | None = None
    if args.llm:
        try:
            llm_provider, llm_note = resolve_llm_provider(Settings())
        except Exception as error:
            llm_note = f"The settings could not be loaded ({type(error).__name__})."
        if llm_provider is not None and not _confirm_llm(llm_provider, args.yes):
            llm_provider, llm_note = None, "Cancelled at the confirmation prompt; no provider calls were made."

    review_session: Session | None = None
    review_note: str | None = "Skipped (--skip-review)." if args.skip_review else None
    if not args.skip_review:
        review_session, review_note = open_review_session(args.database_url)
    try:
        report = build_report(
            repeats=args.repeats,
            llm_provider=llm_provider,
            llm_note=llm_note,
            review_session=review_session,
            review_note=review_note,
        )
    finally:
        if review_session is not None:
            review_session.close()
            review_session.get_bind().dispose()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / REPORT_JSON
    markdown_path = args.output_dir / REPORT_MARKDOWN
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    summary = report["dataset"]
    micro = report["deterministic"]["metrics"]["micro"]
    mock = report["mock"]["metrics"]
    llm = report["llm"]
    review = report["human_review"]
    print("VerityGrid evaluation report")
    print(f"  dataset: {summary['cases']} cases, {summary['columns']} labeled columns, fingerprint {summary['fingerprint'][:12]}")
    print(
        f"  deterministic detection: micro precision {_fmt(micro['precision'])}, recall {_fmt(micro['recall'])}, "
        f"F1 {_fmt(micro['f1'])} (FP {report['deterministic']['metrics']['false_positives']}, "
        f"FN {report['deterministic']['metrics']['false_negatives']})"
    )
    print(f"  mock classification (not an LLM): accuracy {_fmt(mock['accuracy'])} on {mock['n']} columns")
    print(f"  genuine LLM: {'ran' if llm['ran'] else 'not run (' + llm['reason'] + ')'}")
    print(f"  human review: {'decision events ' + str(review['decision_events']) if review['available'] else 'not available'}")
    print(f"  results fingerprint: {report['fingerprints']['results']}")
    print(f"  wrote {json_path}")
    print(f"  wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())