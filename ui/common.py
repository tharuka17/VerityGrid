"""Shared Streamlit helpers: the cached API client, error display, formatting and caching.

Rules for every view:
  - Show only what the API returned. On failure show an explanation, never placeholder numbers.
  - Show untrusted text (column names, dataset names, comments) with st.text or inside a
    dataframe, never through st.markdown or st.write, which would interpret it as markup.
"""

from datetime import timezone
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient, ApiError, ApiErrorKind, UiSettings

CLASSIFICATION_SOURCE_TEXT = {
    "mock": "Mock: rule-based placeholder, not an LLM",
    "llm": "AI model suggestion, unverified (a human must review it)",
    "none": "No suggestion: classification failed or was unavailable",
}
NOT_CLASSIFIED_TEXT = "Not classified"
SEVERITY_ORDER = ("high", "medium", "low")
READ_ONLY_CODE = "read_only_mode"  # must equal app.api.access.READ_ONLY_CODE; a test compares them


@st.cache_resource
def get_client() -> ApiClient:
    """One HTTP client per Streamlit server process (restart Streamlit to pick up new settings)."""
    settings = UiSettings()
    return ApiClient(
        settings.api_url,
        connect_timeout=settings.connect_timeout_seconds,
        read_timeout=settings.read_timeout_seconds,
    )


@st.cache_data(ttl=5, show_spinner=False)
def cached_health(base_url: str, _client: ApiClient) -> dict[str, Any]:
    """Arguments that start with an underscore are not hashed; base_url keys the cache."""
    return _client.health()


@st.cache_data(ttl=10, show_spinner=False)
def load_overview(base_url: str, _client: ApiClient) -> dict[str, Any]:
    """Everything the Overview shows, fetched from the API.

    Recent scans: the API has no list-scans endpoint, and every upload creates one dataset and
    one scan, so the newest datasets with their latest scan are used. That costs one call per
    dataset (five at most). Errors are not cached, so a failed load is retried on the next run.
    """
    metrics = _client.metrics_summary()
    pending = _client.list_findings(limit=10, offset=0, review_status="pending")
    datasets = _client.list_datasets(limit=5, offset=0)
    recent = [
        _client.get_dataset(item["id"])
        for item in datasets["items"]
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    ]
    return {"metrics": metrics, "pending": pending, "recent": recent}


def clear_cache() -> None:
    """Drop cached API data. Call after any write (upload, review) and on manual refresh."""
    st.cache_data.clear()


def show_api_error(error: ApiError) -> None:
    """Explain a failed API call in plain words, with a hint about what to check."""
    if error.kind is ApiErrorKind.UNREACHABLE:
        st.error("The dashboard cannot reach the API.")
        st.caption(
            "Check that the API is running (for example `uvicorn app.main:app --reload`) and "
            "that VERITYGRID_API_URL points to it."
        )
    elif error.kind is ApiErrorKind.TIMEOUT:
        st.error("The API took too long to respond.")
        st.caption("It may be busy. Try again, or raise VERITYGRID_READ_TIMEOUT_SECONDS.")
    elif error.kind is ApiErrorKind.INVALID_RESPONSE:
        st.error("The API sent a response the dashboard does not understand.")
        st.caption(
            f"{error.message} The API and the dashboard may be out of sync; restart both."
        )
    elif error.code == READ_ONLY_CODE:
        st.info(
            "This deployment is read-only, so uploads, scans and reviews are disabled. "
            "You can browse the existing synthetic data in the other views."
        )
    else:
        st.error(f"The API reported a problem (HTTP {error.status}, code {error.code}).")
        st.text(error.message)  # plain text: never interpret API text as markup
        for location, rule in error.issues:
            st.text(f"  {location}: {rule}")


def format_utc(value: object) -> str:
    """Show an ISO timestamp from the API as UTC text, or a dash when it is missing."""
    if not isinstance(value, str):
        return "-"
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return "-"
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def describe_classification(source: str | None) -> str:
    """Say where a classification came from. Mock output must never look like model output."""
    if source is None:
        return NOT_CLASSIFIED_TEXT
    return CLASSIFICATION_SOURCE_TEXT.get(source, "Unknown source")


def to_frame(rows: list[dict[str, Any]], columns: dict[str, str]) -> pd.DataFrame:
    """Rows from the API -> a table with readable headers. Missing keys become empty cells."""
    return pd.DataFrame(
        [{label: row.get(key) for key, label in columns.items()} for row in rows],
        columns=list(columns.values()),
    )
    
WHOLE_DATASET_TEXT = "(whole dataset)"


def with_column_label(row: dict[str, Any]) -> dict[str, Any]:
    """Findings about the whole dataset (for example duplicate rows) have no column."""
    return {**row, "column_name": row.get("column_name") or WHOLE_DATASET_TEXT}


def go_to_view(view: str) -> None:
    """Button callback: switch the sidebar navigation to another view."""
    st.session_state["view"] = view  # "view" is the key of the sidebar radio in streamlit_app.py