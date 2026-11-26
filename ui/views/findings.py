"""Findings explorer: filter stored findings, page through them and inspect one.

Everything comes from the API. Finding evidence holds counts, percentages and rule names only,
and the UI shows no raw values (there are none to show). Column names and review comments are
untrusted text, so they appear only through st.text or inside tables, never as markdown.
"""

from typing import Any

import pandas as pd
import streamlit as st

from ui import common
from ui.api_client import ApiClient, ApiError

PAGE_SIZE = 25
ALL = "All"
ALL_DATASETS = 0  # dataset ids start at 1, so 0 can stand for "no dataset filter"
DATASET_LIST_LIMIT = 200  # the API's maximum page size
SEVERITIES = ("high", "medium", "low")
REVIEW_STATUSES = ("pending", "approved", "rejected")
# These must match the API's DetectionMethod values; a test compares the two lists.
DETECTION_METHODS = {
    "null_percent_threshold": "Missing-value threshold",
    "pandas_duplicated": "Duplicate rows",
    "regex_email_pattern": "Email pattern",
    "regex_phone_pattern": "Phone-number pattern",
    "configured_numeric_bounds": "Numeric range rule",
    "strptime_format_check": "Date validity rule",
}
TABLE_COLUMNS = {
    "id": "Finding",
    "severity": "Severity",
    "category": "Category",
    "method_text": "Detection method",
    "column_name": "Column",
    "dataset_id": "Dataset",
    "scan_run_id": "Scan",
    "review_status": "Review status",
    "title": "Title",
}
HISTORY_COLUMNS = {
    "created": "When (UTC)",
    "decision": "Decision",
    "reviewer": "Reviewer",
    "comment": "Comment",
}
REVIEW_SELECTION_KEY = "review_selected_finding"  # read by the Review queue view


@st.cache_data(ttl=10, show_spinner=False)
def _load_datasets(base_url: str, _client: ApiClient) -> dict[str, Any]:
    page = _client.list_datasets(limit=DATASET_LIST_LIMIT, offset=0)
    ids = [item["id"] for item in page["items"] if isinstance(item, dict) and isinstance(item.get("id"), int)]
    return {"ids": ids, "total": page["total"]}


@st.cache_data(ttl=10, show_spinner=False)
def _load_findings(
    base_url: str,
    page: int,
    dataset_id: int | None,
    severity: str | None,
    review_status: str | None,
    detection_method: str | None,
    _client: ApiClient,
) -> dict[str, Any]:
    return _client.list_findings(
        limit=PAGE_SIZE,
        offset=(page - 1) * PAGE_SIZE,
        dataset_id=dataset_id,
        severity=severity,
        review_status=review_status,
        detection_method=detection_method,
    )


@st.cache_data(ttl=10, show_spinner=False)
def _load_finding(base_url: str, finding_id: int, _client: ApiClient) -> dict[str, Any]:
    return _client.get_finding(finding_id)


def _open_in_review(finding_id: int) -> None:
    """Button callback: remember the finding and switch to the Review queue."""
    st.session_state[REVIEW_SELECTION_KEY] = finding_id
    common.go_to_view("Review queue")


def _method_text(method: object) -> str:
    return DETECTION_METHODS.get(str(method), str(method)) if method else "-"


def render(client: ApiClient) -> None:
    st.header("Findings explorer")
    st.caption(
        "Stored detection results, most severe first. Evidence holds counts and rule names only; "
        "no cell values are stored."
    )
    try:
        datasets = _load_datasets(client.base_url, client)
    except ApiError as error:
        common.show_api_error(error)
        return
    if datasets["total"] == 0:
        st.info("No datasets yet. Open **Upload and scan** to scan a CSV file.")
        return

    dataset_id, severity, review_status, method = _render_filters(datasets)
    filters = (dataset_id, severity, review_status, method)

    try:
        page = _load_findings(client.base_url, 1, *filters, client)
        total = page["total"]
        pages = max(1, -(-total // PAGE_SIZE))
        page_number = 1
        if total and pages > 1:
            page_number = _page_selector(pages, filters)
            if page_number != 1:
                page = _load_findings(client.base_url, page_number, *filters, client)
    except ApiError as error:
        common.show_api_error(error)
        return

    if total == 0:
        if any(value is not None for value in filters):
            st.info("No findings match these filters.")
        else:
            st.info("No findings are stored yet. Scan a dataset to create some.")
        st.caption(_rules_caption())
        return

    rows = [
        {**common.with_column_label(item), "method_text": _method_text(item.get("detection_method"))}
        for item in page["items"]
    ]
    st.dataframe(common.to_frame(rows, TABLE_COLUMNS), hide_index=True)
    st.caption(f"Showing {len(rows)} of {total} findings (page {page_number} of {pages}).")
    st.caption(_rules_caption())

    ids = [item["id"] for item in page["items"] if isinstance(item.get("id"), int)]
    if ids:
        finding_id = st.selectbox("Inspect a finding", ids, format_func=lambda value: f"Finding #{value}")
        _render_detail(client, int(finding_id))


def _rules_caption() -> str:
    return (
        "Numeric range and date validity checks run only when the upload included detection rules "
        "(the rules box in Upload and scan). Without rules, those detection methods have no findings."
    )


def _render_filters(datasets: dict[str, Any]) -> tuple[int | None, str | None, str | None, str | None]:
    columns = st.columns(4)
    dataset_choice = columns[0].selectbox(
        "Dataset",
        [ALL_DATASETS, *datasets["ids"]],
        format_func=lambda value: "All datasets" if value == ALL_DATASETS else f"Dataset #{value}",
    )
    severity = columns[1].selectbox("Severity", [ALL, *SEVERITIES])
    review_status = columns[2].selectbox("Review status", [ALL, *REVIEW_STATUSES])
    method = columns[3].selectbox(
        "Detection method",
        [ALL, *DETECTION_METHODS],
        format_func=lambda value: "All methods" if value == ALL else DETECTION_METHODS[value],
    )
    if datasets["total"] > len(datasets["ids"]):
        st.caption(f"The dataset filter lists the {len(datasets['ids'])} newest of {datasets['total']} datasets.")
    return (
        None if dataset_choice == ALL_DATASETS else int(dataset_choice),
        None if severity == ALL else severity,
        None if review_status == ALL else review_status,
        None if method == ALL else method,
    )


def _page_selector(pages: int, filters: tuple[Any, ...]) -> int:
    """A page number that starts again at 1 whenever the filters change (the key includes them)."""
    key = f"findings_page_{'|'.join(str(value) for value in filters)}"
    if st.session_state.get(key, 1) > pages:  # data shrank since the last visit
        st.session_state[key] = pages
    return int(st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key=key))


def _render_detail(client: ApiClient, finding_id: int) -> None:
    try:
        finding = _load_finding(client.base_url, finding_id, client)
    except ApiError as error:
        common.show_api_error(error)
        return

    st.subheader(f"Finding #{finding['id']}")
    st.text(finding.get("title") or "-")
    st.text(
        f"Severity: {finding.get('severity', '-')}   Category: {finding.get('category', '-')}   "
        f"Review status: {finding.get('review_status', '-')}"
    )
    st.text(
        f"Detection method: {_method_text(finding.get('detection_method'))}   "
        f"Column: {finding.get('column_name') or common.WHOLE_DATASET_TEXT}"
    )
    st.text(
        f"Dataset #{finding.get('dataset_id', '-')}   Scan #{finding.get('scan_run_id', '-')}   "
        f"Created: {common.format_utc(finding.get('created_at'))}"
    )

    st.markdown("**What was found**")
    st.text(finding.get("description") or "-")
    st.markdown("**Recommended next step**")
    st.text(finding.get("recommendation") or "-")

    evidence = finding.get("safe_evidence")
    if isinstance(evidence, dict) and evidence:
        st.markdown("**Evidence** (counts, percentages and rule names only)")
        st.dataframe(
            pd.DataFrame({"Evidence": list(evidence), "Value": [str(value) for value in evidence.values()]}),
            hide_index=True,
        )

    st.markdown("**Review history**")
    history = [
        {**item, "created": common.format_utc(item.get("created_at"))}
        for item in finding["review_decisions"]
        if isinstance(item, dict)
    ]
    if history:
        st.dataframe(common.to_frame(history, HISTORY_COLUMNS), hide_index=True)
    else:
        st.caption("No review decisions yet.")
    st.button("Review this finding", on_click=_open_in_review, args=(int(finding["id"]),))