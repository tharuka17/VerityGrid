"""Upload and scan view: send a CSV to the API, run a scan and show the result.

The file is read into memory and sent to the API as is. It is never previewed or displayed, the
dashboard keeps no copy, and the API stores only column profiles, counts and findings. The API
validates everything (encoding, headers, size, rows, columns and the optional detection rules); the
dashboard only explains the API's answer.
"""

from typing import Any

import streamlit as st

from ui import common
from ui.api_client import ApiClient, ApiError

# A memory guard for the dashboard process only. The API enforces the real limits.
UI_UPLOAD_CAP_BYTES = 25 * 1024 * 1024
RULES_MAX_CHARS = 20_000  # must equal the API's limit on the rules text; a test compares them
RESULT_KEY = "last_scan_result"

EXAMPLE_RULES = """{
  "numeric_rules": {"quantity_on_hand": {"min_value": 0}},
  "date_rules": {"last_counted_date": {"date_format": "%Y-%m-%d"}}
}"""
RULES_NOTES = (
    "Without rules, the checks that run are missing values, duplicate rows, and email and phone patterns.",
    "Numeric ranges and date formats are checked only for the columns you name here. Every column name must exist in the file.",
    "Other fields: missing_value_threshold_percent, missing_value_overrides, missing_value_exempt_columns, duplicate_subset, pattern_columns, pattern_min_match_count.",
    "A numeric rule needs min_value, max_value or both. A date format may use %Y %m %d %y %H %M %S %b %B (each once) and the separators - / . : , space and T.",
    "At most 20 numeric rules and 20 date rules. The API validates the rules; they apply to this scan only and are not stored.",
    "Replace the column names in the example with columns from your own file.",
)

ERROR_HINTS = {
    "file_too_large": "The file is larger than the API accepts. Use a smaller file.",
    "empty_file": "The file is empty, or it has a header but no data rows.",
    "invalid_encoding": "The file must be UTF-8 text. Save it as CSV UTF-8 and try again.",
    "malformed_csv": "A row has a different number of fields than the header, or a quotation mark is not closed.",
    "invalid_header": (
        "Column names must be non-empty, unique (ignoring upper and lower case), "
        "at most 255 characters and free of control characters."
    ),
    "too_many_rows": "The file has more rows than the API allows.",
    "too_many_columns": "The file has more columns than the API allows.",
    "validation_error": (
        "The API rejected the request. Check that the dataset name is at most 255 characters and, if you "
        "entered detection rules, that they are valid JSON that names columns the file has (see the example "
        "under the rules box)."
    ),
    "scan_failed": "The file was valid but the scan failed. No partial results were saved; the failure is recorded on the scan.",
}
FINDING_COLUMNS = {
    "id": "Finding",
    "severity": "Severity",
    "category": "Category",
    "column_name": "Column",
    "title": "Title",
    "review_status": "Review status",
}


def clean_rules_text(text: str | None) -> str | None:
    """Blank or missing rules mean "no rules". Anything else is sent as typed (the API validates it)."""
    cleaned = (text or "").strip()
    return cleaned or None


def render(client: ApiClient) -> None:
    st.header("Upload and scan")
    st.warning("Use synthetic or non-sensitive data only. The API has no authentication yet.")
    st.caption(
        "The file is validated by the API, profiled and checked, then discarded: only column "
        "profiles, counts and findings are stored. Each scan creates a new dataset."
    )

    uploaded = st.file_uploader("CSV file (UTF-8)", type=["csv"])
    name = st.text_input("Dataset name (optional)", max_chars=255, help="Defaults to the file name.")
    rules = st.text_area(
        "Detection rules (optional, JSON)",
        height=120,
        max_chars=RULES_MAX_CHARS,
        help="Configures the checks for this scan only. Leave it empty to use the default rules.",
    )
    with st.expander("About the rules, with an example"):
        for note in RULES_NOTES:
            st.text(note)
        st.code(EXAMPLE_RULES, language="json")
    if st.button("Scan file", type="primary", disabled=uploaded is None):
        _run_scan(client, uploaded, name, rules)

    result = st.session_state.get(RESULT_KEY)
    if result:
        _render_result(result)


def _run_scan(client: ApiClient, uploaded: Any, name: str, rules: str | None = None) -> None:
    st.session_state.pop(RESULT_KEY, None)
    if uploaded is None:
        return
    if uploaded.size > UI_UPLOAD_CAP_BYTES:
        st.error("The file is larger than the dashboard will send (25 MB). The API's own limit is usually smaller.")
        return
    content = uploaded.getvalue()
    try:
        with st.spinner("Uploading and scanning..."):
            result = client.upload_scan(
                uploaded.name,
                content,
                dataset_name=name.strip() or None,
                rules=clean_rules_text(rules),
            )
    except ApiError as error:
        common.show_api_error(error)
        hint = ERROR_HINTS.get(error.code or "")
        if hint:
            st.info(hint)
        return
    common.clear_cache()  # the Overview and the catalog must show the new dataset
    st.session_state[RESULT_KEY] = result


def _render_result(result: dict[str, Any]) -> None:
    st.success(
        f"Scan #{result['scan_run_id']} {result['status']}. "
        f"Dataset #{result['dataset_id']} was created."
    )
    severity = result.get("findings_by_severity") or {}
    tiles = st.columns(4)
    tiles[0].metric("Rows", result.get("row_count", "-"))
    tiles[1].metric("Columns", result.get("column_count", "-"))
    tiles[2].metric("Findings", result["finding_count"])
    tiles[3].metric("High severity", severity.get("high", 0))

    st.caption(
        "Uploads through the API are not classified. Numeric-range and date-validity checks run only for the "
        "columns named in the rules box; without rules, the checks that run are missing values, duplicate rows, "
        "and email and phone patterns."
    )
    findings = result["findings"]
    if not findings:
        st.info("No findings were detected under the rules that were applied.")
    else:
        rows = [common.with_column_label(item) for item in findings]
        st.dataframe(common.to_frame(rows, FINDING_COLUMNS), hide_index=True)

    left, middle, right = st.columns(3)
    left.button("Open Data catalog", on_click=common.go_to_view, args=("Data catalog",))
    middle.button("Open Findings explorer", on_click=common.go_to_view, args=("Findings explorer",))
    right.button("Open Review queue", on_click=common.go_to_view, args=("Review queue",))