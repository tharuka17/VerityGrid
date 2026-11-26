"""Data catalog view: stored datasets, their latest scan, column profiles and classifications.

Everything shown comes from the API. Names (dataset, source file, column, explanation) are
untrusted text: they are shown with st.text or inside tables, never as markdown.
"""

from typing import Any

import pandas as pd
import streamlit as st

from ui import common
from ui.api_client import ApiClient, ApiError

PAGE_SIZE = 20
DATASET_COLUMNS = {
    "id": "Dataset",
    "name": "Name",
    "source_filename": "Source file",
    "row_count": "Rows",
    "column_count": "Columns",
    "created": "Created",
}
PROFILE_COLUMNS = {
    "column_name": "Column",
    "inferred_type": "Type",
    "null_count": "Nulls",
    "null_percent": "Null %",
    "distinct_count": "Distinct",
    "source_text": "Classification source",
    "label_text": "Label",
    "explanation_text": "Explanation",
}
WORKFLOW_STATUS_TEXT = {
    "completed": "Completed: every requested step finished.",
    "partially_completed": (
        "Partially completed: the deterministic checks finished and were saved, "
        "but AI classification did not fully complete."
    ),
    "failed": "Failed: the scan did not complete.",
}


@st.cache_data(ttl=10, show_spinner=False)
def _load_page(base_url: str, page: int, _client: ApiClient) -> dict[str, Any]:
    return _client.list_datasets(limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)


@st.cache_data(ttl=10, show_spinner=False)
def _load_detail(base_url: str, dataset_id: int, _client: ApiClient) -> dict[str, Any]:
    dataset = _client.get_dataset(dataset_id)
    columns = _client.get_columns(dataset_id)
    scan_id = dataset.get("latest_scan_id")
    scan = _client.get_scan(scan_id) if isinstance(scan_id, int) else None
    return {"dataset": dataset, "columns": columns, "scan": scan}


def render(client: ApiClient) -> None:
    st.header("Data catalog")
    st.caption(
        "Datasets stored by the API, with their latest scan and column profiles. "
        "No cell values are stored anywhere: only names, types, counts and findings."
    )
    try:
        first_page = _load_page(client.base_url, 1, client)
        total = first_page["total"]
        if total == 0:
            st.info("No datasets yet. Open **Upload and scan** to scan a CSV file.")
            return
        pages = max(1, -(-total // PAGE_SIZE))
        page_number = 1
        if pages > 1:
            page_number = int(st.number_input("Page", min_value=1, max_value=pages, value=1, step=1))
        page = first_page if page_number == 1 else _load_page(client.base_url, page_number, client)
    except ApiError as error:
        common.show_api_error(error)
        return

    rows = [{**item, "created": common.format_utc(item.get("created_at"))} for item in page["items"]]
    st.dataframe(common.to_frame(rows, DATASET_COLUMNS), hide_index=True)
    st.caption(f"Showing {len(rows)} of {total} datasets, newest first.")

    ids = [item["id"] for item in page["items"] if isinstance(item.get("id"), int)]
    if not ids:
        return
    # Option labels show only the id: names are untrusted text and stay in the table above.
    dataset_id = st.selectbox("Dataset to inspect", ids, format_func=lambda value: f"Dataset #{value}")
    _render_detail(client, int(dataset_id))


def _render_detail(client: ApiClient, dataset_id: int) -> None:
    try:
        detail = _load_detail(client.base_url, dataset_id, client)
    except ApiError as error:
        common.show_api_error(error)
        return
    dataset, columns, scan = detail["dataset"], detail["columns"], detail["scan"]

    st.subheader("Dataset")
    st.text(f"Name: {dataset['name']}")
    st.text(f"Source file: {dataset.get('source_filename') or '-'}")
    st.text(f"Created: {common.format_utc(dataset.get('created_at'))}")
    tiles = st.columns(5)
    tiles[0].metric("Rows", dataset["row_count"])
    tiles[1].metric("Columns", dataset["column_count"])
    tiles[2].metric("Scans", dataset["scan_count"])
    tiles[3].metric("Findings", dataset["finding_count"])
    tiles[4].metric("Pending review", dataset["pending_review_count"])
    if dataset["finding_count"]:
        st.button("Open Findings explorer", on_click=common.go_to_view, args=("Findings explorer",))

    if scan is None:
        st.info("This dataset has no scan record.")
    else:
        _render_scan(scan)
    _render_profiles(columns["columns"])


def _render_scan(scan: dict[str, Any]) -> None:
    st.subheader("Latest scan")
    st.text(f"Scan #{scan['id']}: {scan['status']}")
    st.text(
        f"Started: {common.format_utc(scan.get('started_at'))}   "
        f"Finished: {common.format_utc(scan.get('finished_at'))}"
    )
    if scan.get("error_details"):
        st.error("This scan failed. No results were saved for it.")
        st.text(str(scan["error_details"]))
    summary = scan.get("summary")
    if not isinstance(summary, dict):
        return
    severity = summary.get("findings_by_severity") or {}
    st.text(
        f"Findings: {summary.get('finding_count', 0)} "
        f"(high {severity.get('high', 0)}, medium {severity.get('medium', 0)}, "
        f"low {severity.get('low', 0)})"
    )
    workflow = summary.get("workflow")
    if isinstance(workflow, dict):
        _render_workflow(workflow)
    else:
        st.caption("This scan ran the standard deterministic checks; no AI classification was requested.")


def _render_workflow(workflow: dict[str, Any]) -> None:
    status = str(workflow.get("status", "unknown"))
    st.text(f"Workflow status: {status}")
    st.caption(WORKFLOW_STATUS_TEXT.get(status, ""))
    classification = workflow.get("classification")
    if isinstance(classification, dict) and classification.get("requested"):
        st.text(
            f"Classification: {classification.get('attempted', 0)} attempted, "
            f"{classification.get('succeeded', 0)} succeeded, {classification.get('failed', 0)} failed, "
            f"{classification.get('skipped_by_cap', 0)} skipped by the column cap, "
            f"{classification.get('skipped_by_breaker', 0)} skipped after repeated failures"
        )
        st.text(f"Provider: {classification.get('provider') or '-'}")
        if classification.get("circuit_opened"):
            st.warning("The circuit breaker stopped further provider calls after repeated failures.")
    for issue in workflow.get("issues") or []:
        if isinstance(issue, dict):
            suffix = f" ({issue['error_type']})" if issue.get("error_type") else ""
            st.text(f"Issue in {issue.get('step', '?')}: {issue.get('code', '?')}{suffix}")
    durations = workflow.get("durations_ms")
    if isinstance(durations, dict) and durations:
        with st.expander("Step timings"):
            st.dataframe(
                pd.DataFrame({"Step": list(durations), "Milliseconds": list(durations.values())}),
                hide_index=True,
            )


def _render_profiles(columns: list[dict[str, Any]]) -> None:
    st.subheader("Column profiles")
    if not columns:
        st.info("No column profiles are stored for this dataset.")
        return
    _render_classification_notices(columns)
    rows = [
        {
            **column,
            "source_text": common.describe_classification(column.get("classification_source")),
            "label_text": column.get("classification_label") or "-",
            "explanation_text": column.get("classification_explanation") or "-",
        }
        for column in columns
    ]
    st.dataframe(common.to_frame(rows, PROFILE_COLUMNS), hide_index=True)


def _render_classification_notices(columns: list[dict[str, Any]]) -> None:
    """Say, in words, where every label came from. Mock output must never pass as model output."""
    sources = [column.get("classification_source") for column in columns]
    mock, model, failed, unclassified = (
        sources.count("mock"),
        sources.count("llm"),
        sources.count("none"),
        sources.count(None),
    )
    st.text(
        f"Classification sources: {mock} mock, {model} AI model, "
        f"{failed} no suggestion, {unclassified} not classified"
    )
    if mock:
        st.warning(
            "Labels marked Mock come from a keyword-based placeholder, not from a language model. "
            "Do not treat them as model output."
        )
    if model:
        st.info("Labels marked AI model are suggestions from a language model. They are unverified; a human must review them.")
    if failed:
        st.info("Columns marked No suggestion were attempted, but no valid suggestion was produced.")
    if unclassified == len(columns):
        st.caption(
            "Classification is optional and off by default. Uploads through the API do not request it, "
            "so datasets created through the dashboard show Not classified."
        )