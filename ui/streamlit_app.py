"""VerityGrid dashboard: entry point, navigation and the Overview view.

Start (from the project root, with the API already running):
    streamlit run ui/streamlit_app.py

The dashboard is a pure HTTP client: it never imports the backend, opens the database or reads
the Gemini key. Other views live in ui/views/ and are loaded on demand; a view that is not
installed yet says so instead of failing.
"""

import importlib
import sys
from pathlib import Path

# Streamlit puts only this file's folder on the import path; the views import `ui.*`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ui import common  # noqa: E402
from ui.api_client import ApiClient, ApiError  # noqa: E402

# View name -> module in ui/views (None: rendered in this file).
VIEWS: dict[str, str | None] = {
    "Overview": None,
    "Data catalog": "catalog",
    "Upload and scan": "upload",
    "Findings explorer": "findings",
    "Review queue": "review",
    "Evaluation": "evaluation",
}

PENDING_COLUMNS = {
    "id": "Finding",
    "severity": "Severity",
    "category": "Category",
    "column_name": "Column",
    "dataset_id": "Dataset",
    "title": "Title",
}
RECENT_COLUMNS = {
    "name": "Dataset",
    "latest_scan_id": "Scan",
    "latest_scan_status": "Scan status",
    "row_count": "Rows",
    "column_count": "Columns",
    "finding_count": "Findings",
    "pending_review_count": "Pending review",
    "created": "Created",
}


def render_sidebar(client: ApiClient) -> str:
    st.sidebar.title("VerityGrid")
    st.sidebar.caption(
        "AI-assisted data governance workbench. An independent portfolio prototype using "
        "synthetic data; not affiliated with Sysco or Sysco LABS."
    )
    view = st.sidebar.radio("View", list(VIEWS), key="view")
    st.sidebar.divider()

    if st.sidebar.button("Refresh data"):
        common.clear_cache()
    try:
        health = common.cached_health(client.base_url, client)
        st.sidebar.success("API reachable")
        st.sidebar.text(f"{health.get('app', 'API')} ({health.get('environment', 'unknown')})")
    except ApiError:
        st.sidebar.error("API unreachable")
    st.sidebar.text(f"API: {client.base_url}")
    return view


def render_overview(client: ApiClient) -> None:
    st.header("Overview")
    st.caption("Read from the API. Every number is calculated from stored scan results.")
    try:
        data = common.load_overview(client.base_url, client)
    except ApiError as error:
        common.show_api_error(error)
        return

    metrics = data["metrics"]
    if metrics["dataset_count"] == 0:
        st.info("No datasets yet. Open **Upload and scan** to scan a CSV file.")
        return

    scans = metrics["scans_by_status"]
    pending = metrics["findings_by_review_status"].get("pending", 0)
    percent = metrics["reviewed_finding_percent"]

    top = st.columns(3)
    top[0].metric("Datasets", metrics["dataset_count"])
    top[1].metric(
        "Scans",
        metrics["scan_count"],
        help=f"Completed: {scans.get('completed', 0)}. Failed: {scans.get('failed', 0)}.",
    )
    top[2].metric("Findings", metrics["finding_count"])
    bottom = st.columns(3)
    bottom[0].metric("Pending review", pending)
    bottom[1].metric("Pending high severity", metrics["pending_high_severity_count"])
    bottom[2].metric(
        "Findings reviewed",
        "n/a" if percent is None else f"{percent}%",
        help="Share of findings with an approve or reject decision. n/a means there are no findings yet.",
    )

    if scans.get("failed", 0):
        st.warning(
            f"{scans['failed']} scan(s) failed. A failed scan keeps only a generic reason; "
            "no results were saved for it."
        )

    if metrics["finding_count"] > 0:
        left, right = st.columns(2)
        severity = metrics["findings_by_severity"]
        review = metrics["findings_by_review_status"]
        left.subheader("Findings by severity")
        left.bar_chart(pd.Series({name.title(): severity.get(name, 0) for name in common.SEVERITY_ORDER}))
        right.subheader("Findings by review status")
        right.bar_chart(
            pd.Series({name.title(): review.get(name, 0) for name in ("pending", "approved", "rejected")})
        )

    st.subheader("Unresolved findings")
    page = data["pending"]
    if page["total"] == 0:
        st.success("No findings are waiting for review.")
    else:
        pending_rows = [common.with_column_label(item) for item in page["items"]]
        st.dataframe(common.to_frame(pending_rows, PENDING_COLUMNS), hide_index=True)
        st.caption(
            f"Showing {len(page['items'])} of {page['total']} pending findings, most severe first. "
            "Use **Review queue** to approve or reject them."
        )

    st.subheader("Recent scans")
    st.caption("The newest datasets and their latest scan. Each upload creates one dataset and one scan.")
    rows = [{**item, "created": common.format_utc(item.get("created_at"))} for item in data["recent"]]
    st.dataframe(common.to_frame(rows, RECENT_COLUMNS), hide_index=True)


def render_view(client: ApiClient, name: str) -> None:
    module_name = VIEWS[name]
    if module_name is None:
        render_overview(client)
        return
    full_name = f"ui.views.{module_name}"
    try:
        module = importlib.import_module(full_name)
    except ModuleNotFoundError as error:
        if error.name not in {full_name, "ui.views"}:
            raise  # a missing dependency inside a view is a real error, not "not installed yet"
        st.header(name)
        st.info("This view is added in a later build step and is not installed yet.")
        return
    module.render(client)


def main() -> None:
    st.set_page_config(page_title="VerityGrid", layout="wide")
    try:
        client = common.get_client()
    except Exception:
        # The settings error text could contain the configured URL, so it is not shown.
        st.error("The dashboard configuration is invalid. Check VERITYGRID_API_URL and the timeouts.")
        st.stop()
    render_view(client, render_sidebar(client))


main()