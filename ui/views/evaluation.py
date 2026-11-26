"""Evaluation view: the latest generated evaluation report, read over HTTP.

The dashboard calculates nothing here. Every number comes from the report that the evaluation command
(`python -m app.evaluation.harness`) wrote and the API served unchanged. A missing or unusable report is
explained instead of replaced by placeholder values. Three sources of evidence stay separate: the
deterministic detectors, the mock provider (a placeholder, not an LLM) and, only if it was run, a genuine
LLM provider. Report text (case, column and class names) is untrusted: it is shown with st.text or inside
tables, never as markup.
"""

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import streamlit as st

from ui import common
from ui.api_client import ApiClient, ApiError

REGENERATE_COMMAND = "python -m app.evaluation.harness --skip-review"
SHAPE_WARNING = (
    "This section could not be displayed because the report does not have the expected shape. "
    "Regenerate the report with the current evaluation command."
)
MOCK_BANNER = (
    "These are MOCK results (a keyword placeholder, not an LLM). They say nothing about how a "
    "language model performs."
)
LLM_BANNER = (
    "Genuine LLM output from a single, non-reproducible run. It is unverified and says nothing "
    "beyond this small dataset."
)
RATE_HEADERS = ["Support", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Recall 95% CI"]
ERROR_COLUMNS = {
    "case": "Case",
    "tag": "Tag",
    "column": "Column",
    "category": "Category",
    "kind": "Kind",
}
MISCLASSIFIED_COLUMNS = {"case": "Case", "column": "Column", "actual": "True", "predicted": "Predicted"}


# --- Formatting (pure functions; tested without a browser) ---------------------------------------


def fmt_rate(value: object) -> str:
    """A rate with no denominator arrives as None: show n/a, never 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.3f}"


def fmt_interval(interval: object) -> str:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return "n/a"
    low, high = interval
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (low, high)):
        return "n/a"
    return f"[{low:.2f}, {high:.2f}]"


def fmt_count(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "n/a"
    return str(value)


def fmt_ms(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.1f}"


def fmt_bool(value: object) -> str:
    return "yes" if value is True else "no" if value is False else "n/a"


def counts_text(counts: object) -> str:
    if not isinstance(counts, Mapping) or not counts:
        return "none"
    return ", ".join(f"{key}: {fmt_count(value)}" for key, value in sorted(counts.items()))


def rate_table(reports: Mapping[str, Mapping[str, Any]], first_header: str) -> pd.DataFrame:
    rows = []
    for name, report in reports.items():
        rows.append(
            {
                first_header: str(name),
                "Support": fmt_count(report.get("support")),
                "TP": fmt_count(report.get("tp")),
                "FP": fmt_count(report.get("fp")),
                "FN": fmt_count(report.get("fn")),
                "TN": fmt_count(report.get("tn")),
                "Precision": fmt_rate(report.get("precision")),
                "Recall": fmt_rate(report.get("recall")),
                "F1": fmt_rate(report.get("f1")),
                "Recall 95% CI": fmt_interval(report.get("recall_ci95")),
            }
        )
    return pd.DataFrame(rows, columns=[first_header, *RATE_HEADERS])


def confusion_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    columns = list(metrics["predicted_columns"])
    rows = []
    for actual, row in metrics["matrix"].items():
        rows.append(
            {
                "Actual \\ Predicted": str(actual),
                **{column: fmt_count(row.get(column, 0)) for column in columns},
                "Total": fmt_count(sum(row.values())),
            }
        )
    return pd.DataFrame(rows, columns=["Actual \\ Predicted", *columns, "Total"])


def latency_table(latency: object) -> pd.DataFrame:
    data = latency if isinstance(latency, Mapping) else {}
    row = {
        "Runs": fmt_count(data.get("n")),
        "Min ms": fmt_ms(data.get("min_ms")),
        "Median ms": fmt_ms(data.get("median_ms")),
        "Max ms": fmt_ms(data.get("max_ms")),
        "Mean ms": fmt_ms(data.get("mean_ms")),
    }
    return pd.DataFrame([row])


def review_table(groups: object) -> pd.DataFrame:
    rows = []
    if isinstance(groups, Mapping):
        for name, item in sorted(groups.items()):
            rows.append(
                {
                    "Name": str(name),
                    "Approved": fmt_count(item.get("approved")),
                    "Rejected": fmt_count(item.get("rejected")),
                    "Acceptance": fmt_rate(item.get("acceptance_rate")),
                }
            )
    return pd.DataFrame(rows, columns=["Name", "Approved", "Rejected", "Acceptance"])


# --- Loading ---------------------------------------------------------------------------------------


@st.cache_data(ttl=10, show_spinner=False)
def _load_report(base_url: str, _client: ApiClient) -> dict[str, Any]:
    return _client.evaluation_report()


# --- Sections ----------------------------------------------------------------------------------------


def _guarded(title: str, draw: Callable[[], None]) -> None:
    """One section that does not match the expected shape must not take the whole page down."""
    st.subheader(title)
    try:
        draw()
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        st.warning(SHAPE_WARNING)


def _draw_text_list(items: object) -> None:
    if isinstance(items, list):
        for item in items:
            st.text(str(item))


def _draw_summary(report: Mapping[str, Any], modified_at: object) -> None:
    run, fingerprints = report["run"], report["fingerprints"]
    st.text(f"Report file written: {common.format_utc(modified_at)}")
    st.text(
        f"Generated: {run.get('generated_at_utc', 'n/a')} | Python {run.get('python', 'n/a')} | "
        f"pandas {run.get('pandas', 'n/a')} | passes {fmt_count(run.get('repeats'))}"
    )
    st.text(f"Dataset fingerprint: {fingerprints['dataset']}")
    st.text(f"Results fingerprint: {fingerprints['results']}")
    st.text(f"Fingerprint scope: {fingerprints.get('scope', 'n/a')}")
    if run.get("is_subset"):
        st.warning("PARTIAL RUN: only some cases were evaluated. Do not compare these numbers with a full report.")
        st.text("Cases: " + ", ".join(str(case) for case in run.get("cases_selected", [])))


def _draw_dataset(report: Mapping[str, Any]) -> None:
    data = report["dataset"]
    tiles = st.columns(3)
    tiles[0].metric("Synthetic cases", fmt_count(data["cases"]))
    tiles[1].metric("Labeled columns", fmt_count(data["columns"]))
    tiles[2].metric("Rows in total", fmt_count(data["rows_total"]))
    st.text("Cases by tag: " + counts_text(data.get("cases_by_tag")))
    st.text("Labeled columns per class: " + counts_text(data.get("label_distribution")))
    st.text("Actual positives per detection category: " + counts_text(data.get("detection_positives")))
    guidelines = data.get("annotation_guidelines")
    if isinstance(guidelines, Mapping):
        with st.expander("How the labels were defined"):
            for name, text in guidelines.items():
                st.text(f"{name}: {text}")


def _draw_averages(label: str, block: Mapping[str, Any]) -> None:
    st.text(
        f"{label}: precision {fmt_rate(block.get('precision'))}, recall {fmt_rate(block.get('recall'))}, "
        f"F1 {fmt_rate(block.get('f1'))}"
    )
    classes = block.get("classes")
    if isinstance(classes, list):
        never = block.get("zero_division_classes") or []
        st.text(
            "Averaged over: " + (", ".join(map(str, classes)) or "none")
            + ". Counted as 0 because never predicted: " + (", ".join(map(str, never)) or "none") + "."
        )


def _draw_detection(report: Mapping[str, Any]) -> None:
    det = report["deterministic"]
    metrics = det["metrics"]
    tiles = st.columns(4)
    tiles[0].metric("Units scored", fmt_count(metrics["units"]))
    tiles[1].metric("Actual positives", fmt_count(metrics["positives"]))
    tiles[2].metric("False positives", fmt_count(metrics["false_positives"]))
    tiles[3].metric("False negatives", fmt_count(metrics["false_negatives"]))
    st.dataframe(rate_table(metrics["per_category"], "Category"), hide_index=True)
    _draw_averages("Micro (all counts pooled)", metrics["micro"])
    _draw_averages("Macro (mean over categories with examples)", metrics["macro"])
    st.markdown("**By case tag**")
    st.dataframe(rate_table(metrics["by_tag"], "Tag"), hide_index=True)
    errors = metrics.get("errors") or []
    with st.expander(f"Every false positive and false negative ({len(errors)})"):
        if errors:
            st.dataframe(common.to_frame(errors, ERROR_COLUMNS), hide_index=True)
        else:
            st.text("None were recorded.")
    st.text(
        f"Findings outside the labeled units: {fmt_count(len(metrics.get('unmatched_findings') or []))}. "
        f"Workflow statuses: {counts_text(det.get('workflow_statuses'))}. "
        f"Identical results across {fmt_count(det.get('repeats'))} pass(es): "
        f"{fmt_bool(det.get('consistent_across_repeats'))}."
    )
    st.text(str(det.get("latency_scope") or "Workflow latency:"))
    st.dataframe(latency_table(det.get("latency")), hide_index=True)


def _draw_classification(section: Mapping[str, Any], banner: str) -> None:
    if not section["ran"]:
        st.info("This source was not run, so there are no results.")
        st.text(str(section.get("reason") or ""))
        return
    st.warning(banner)
    metrics = section["metrics"]
    st.text(f"Provider: {section.get('provider', 'n/a')}")
    tiles = st.columns(4)
    tiles[0].metric("Columns scored", fmt_count(metrics["n"]))
    tiles[1].metric(
        "Accuracy", fmt_rate(metrics["accuracy"]), help="Only meaningful if it clearly beats the baseline."
    )
    tiles[2].metric("Majority-class baseline", fmt_rate(metrics["majority_baseline_accuracy"]))
    tiles[3].metric("No valid prediction", fmt_count(metrics["no_prediction_count"]))
    st.text(f"Accuracy 95% interval: {fmt_interval(metrics.get('accuracy_ci95'))}")
    coverage = section.get("coverage")
    if isinstance(coverage, Mapping):
        st.text(
            f"Coverage: {fmt_count(coverage.get('answered'))} of {fmt_count(coverage.get('columns'))} "
            f"columns received a valid answer ({fmt_rate(coverage.get('answered_share'))})."
        )
    st.dataframe(rate_table(metrics["per_class"], "Class"), hide_index=True)
    _draw_averages("Micro (all counts pooled)", metrics["micro"])
    _draw_averages("Macro (mean over classes with examples)", metrics["macro"])
    st.markdown("**Confusion matrix** (rows: true class, columns: predicted)")
    st.dataframe(confusion_table(metrics), hide_index=True)
    st.text("Outcomes: " + counts_text(section.get("outcomes")))

    answered = section.get("answered_only")
    if isinstance(answered, Mapping) and isinstance(coverage, Mapping) and coverage.get("unanswered"):
        st.markdown("**Answered-only results**")
        st.text(
            "Columns without a valid answer are left out here instead of counted as wrong. If the failures "
            "are not random this can flatter the model, and fewer columns mean wider uncertainty."
        )
        st.text(
            f"Answered columns: {fmt_count(answered.get('n'))}. Accuracy: {fmt_rate(answered.get('accuracy'))} "
            f"(95% interval {fmt_interval(answered.get('accuracy_ci95'))})."
        )
        st.dataframe(rate_table(answered["per_class"], "Class"), hide_index=True)

    wrong = section.get("misclassified") or []
    with st.expander(f"Misclassified or unanswered columns ({len(wrong)})"):
        if wrong:
            st.dataframe(common.to_frame(wrong, MISCLASSIFIED_COLUMNS), hide_index=True)
        else:
            st.text("None.")

    calls = section.get("provider_calls")
    if isinstance(calls, Mapping):
        st.text(f"Provider calls: {fmt_count(calls.get('count'))} (failures: {fmt_count(calls.get('failures'))}).")
        st.text(str(calls.get("note") or ""))
        st.dataframe(latency_table(calls.get("latency")), hide_index=True)
    guard = section.get("guard")
    if isinstance(guard, Mapping):
        st.text(
            f"Pacing and stop rule: {fmt_count(guard.get('calls_sent'))} calls sent, "
            f"{fmt_ms(guard.get('seconds_waited'))} s waited, stopped early: {fmt_bool(guard.get('stopped_early'))}, "
            f"calls skipped after the stop: {fmt_count(guard.get('calls_skipped_after_stop'))}."
        )
    st.text("Workflow latency (one workflow run on one dataset):")
    st.dataframe(latency_table(section.get("workflow_latency")), hide_index=True)
    st.text(
        f"Identical results across {fmt_count(section.get('repeats'))} pass(es): "
        f"{fmt_bool(section.get('consistent_across_repeats'))}."
    )
    for key, label in (("cost", "Cost"), ("token_usage", "Token usage")):
        item = section.get(key)
        if isinstance(item, Mapping):
            st.text(f"{label}: {item.get('reason') or 'see the report'}")


def _draw_review(review: Mapping[str, Any]) -> None:
    if not review["available"]:
        st.info("Human review counts are not part of this report.")
        st.text(str(review.get("reason") or ""))
        return
    st.warning(
        "These counts come from the local database. They may include test decisions, are not ground truth "
        "and are not reproducible."
    )
    latest = review.get("latest_decisions") or {}
    tiles = st.columns(3)
    tiles[0].metric("Decision events", fmt_count(review.get("decision_events")))
    tiles[1].metric("Findings with decisions", fmt_count(review.get("findings_with_decisions")))
    tiles[2].metric(
        "Acceptance rate",
        fmt_rate(review.get("acceptance_rate")),
        help="Share of reviewed findings whose latest decision is approve. Not comparable to recall.",
    )
    st.text(
        f"Latest decision per finding: approved {fmt_count(latest.get('approved'))}, "
        f"rejected {fmt_count(latest.get('rejected'))}. "
        f"Acceptance 95% interval: {fmt_interval(review.get('acceptance_rate_ci95'))}."
    )
    st.text(
        f"Re-reviewed findings: {fmt_count(review.get('re_reviewed_findings'))}. "
        f"Decision changed: {fmt_count(review.get('decision_changed_findings'))}. "
        f"Status mismatches: {fmt_count(review.get('status_mismatch_findings'))}. "
        f"Orphan events: {fmt_count(review.get('orphan_events'))}."
    )
    for title, key in (("By finding category", "by_category"), ("By detection method", "by_detection_method")):
        table = review_table(review.get(key))
        if len(table):
            st.markdown("**Latest decisions by group**")
            st.text(title)
            st.dataframe(table, hide_index=True)
    st.text(str(review.get("note") or ""))


# --- Entry point -------------------------------------------------------------------------------------


def render(client: ApiClient) -> None:
    st.header("Evaluation")
    st.caption(
        "The latest report written by the evaluation command and served by the API. "
        "The dashboard calculates nothing on this page."
    )
    try:
        answer = _load_report(client.base_url, client)
    except ApiError as error:
        common.show_api_error(error)
        return

    if not answer["available"]:
        st.info("No usable evaluation report is available, so there is nothing to show.")
        st.text(str(answer.get("reason") or ""))
        st.text(f"Generate it from the project folder with: {REGENERATE_COMMAND}")
        st.caption("Then click Refresh data in the sidebar.")
        return

    report = answer["report"]
    st.info(
        "Three sources of evidence are shown separately and are never combined: the deterministic detectors, "
        "the mock provider (a placeholder, not an LLM) and, only if it was run, a genuine LLM provider. "
        "Read the limitations first: the dataset is small, synthetic and was written by the same author as the "
        "code it measures."
    )
    with st.expander("Limitations (read these first)", expanded=True):
        try:
            _draw_text_list(report["limitations"])
        except (KeyError, TypeError):
            st.warning(SHAPE_WARNING)
    _guarded("Report", lambda: _draw_summary(report, answer.get("file_modified_at")))
    _guarded("Dataset", lambda: _draw_dataset(report))
    _guarded("Deterministic detectors", lambda: _draw_detection(report))
    _guarded("Mock classification provider", lambda: _draw_classification(report["mock"], MOCK_BANNER))
    _guarded("Genuine LLM provider", lambda: _draw_classification(report["llm"], LLM_BANNER))
    _guarded("Human review events", lambda: _draw_review(report["human_review"]))
    with st.expander("How to read this report"):
        try:
            _draw_text_list(report["how_to_read"])
        except (KeyError, TypeError):
            st.warning(SHAPE_WARNING)