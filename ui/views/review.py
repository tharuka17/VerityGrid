"""Review queue: inspect a finding and record an approve or reject decision.

A review only records a decision and an audit event through the API. It never changes, deletes
or remediates any data. The reviewer name is self-reported (the API has no authentication) and
comments are stored as typed. Findings are shown with counts and rule names only; column names
and comments are untrusted text and are never rendered as markdown.
"""

from typing import Any

import pandas as pd
import streamlit as st

from ui import common
from ui.api_client import ApiClient, ApiError

QUEUE_LIMIT = 50
# Written by the Findings explorer ("Review this finding"); keep in step with ui/views/findings.py.
SELECTION_KEY = "review_selected_finding"
CHOICE_KEY = "review_finding_choice"
FLASH_KEY = "review_flash"
DECISIONS = {"approved": "Approve this recommendation", "rejected": "Reject this recommendation"}
HISTORY_COLUMNS = {
    "created": "When (UTC)",
    "decision": "Decision",
    "reviewer": "Reviewer",
    "comment": "Comment",
}


@st.cache_data(ttl=10, show_spinner=False)
def _load_queue(base_url: str, _client: ApiClient) -> dict[str, Any]:
    return _client.list_findings(limit=QUEUE_LIMIT, offset=0, review_status="pending")


@st.cache_data(ttl=10, show_spinner=False)
def _load_finding(base_url: str, finding_id: int, _client: ApiClient) -> dict[str, Any]:
    return _client.get_finding(finding_id)


def render(client: ApiClient) -> None:
    st.header("Review queue")
    st.caption(
        "Pending findings, most severe first. A decision is recorded with an audit event; "
        "it never changes, deletes or fixes any data."
    )
    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        st.success(flash)

    # Hand-off from the Findings explorer: use it once, so it cannot override later choices.
    preselected = st.session_state.pop(SELECTION_KEY, None)
    if isinstance(preselected, int):
        st.session_state[CHOICE_KEY] = preselected

    try:
        queue = _load_queue(client.base_url, client)
    except ApiError as error:
        common.show_api_error(error)
        return

    items = [item for item in queue["items"] if isinstance(item, dict) and isinstance(item.get("id"), int)]
    # Labels use only the id and two fixed values (severity, category), never untrusted text.
    labels = {
        item["id"]: f"#{item['id']} · {item.get('severity', '?')} · {item.get('category', '?')}"
        for item in items
    }
    current = st.session_state.get(CHOICE_KEY)
    if isinstance(current, int) and current not in labels:
        labels = {current: f"#{current} (opened directly)", **labels}  # reviewed or chosen elsewhere
    if not labels:
        st.success("No findings are waiting for review.")
        st.caption("To change an earlier decision, open it in the Findings explorer and use Review this finding.")
        return

    st.text(f"{queue['total']} finding(s) pending review.")
    if queue["total"] > len(items):
        st.caption(f"Showing the {len(items)} most severe. Review some to see the rest.")
    finding_id = st.selectbox("Finding", list(labels), format_func=lambda value: labels[value], key=CHOICE_KEY)
    _render_finding(client, int(finding_id))


def _render_finding(client: ApiClient, finding_id: int) -> None:
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
        f"Detection method: {finding.get('detection_method', '-')}   "
        f"Column: {finding.get('column_name') or common.WHOLE_DATASET_TEXT}"
    )
    st.text(
        f"Dataset #{finding.get('dataset_id', '-')}   Scan #{finding.get('scan_run_id', '-')}   "
        f"Created: {common.format_utc(finding.get('created_at'))}"
    )
    status = finding.get("review_status")
    if status and status != "pending":
        st.info(
            f"This finding is already {status}. A new decision is added to its history "
            "and becomes its current status."
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

    _render_form(client, int(finding["id"]))


def _render_form(client: ApiClient, finding_id: int) -> None:
    st.subheader("Your decision")
    st.warning(
        "The reviewer name and comment are stored exactly as typed, and the API has no "
        "authentication, so the name is self-reported. Do not enter sensitive values."
    )
    with st.form(f"review_form_{finding_id}", clear_on_submit=True):
        decision = st.radio("Decision", list(DECISIONS), format_func=DECISIONS.__getitem__, index=None)
        comment = st.text_area("Comment (optional)", max_chars=1000)
        reviewer = st.text_input("Your name (optional)", max_chars=100)
        submitted = st.form_submit_button("Record decision", type="primary")
    if not submitted:
        return
    if decision is None:
        st.warning("Choose Approve or Reject first. Nothing was recorded.")
        return

    try:
        with st.spinner("Recording the decision..."):
            result = client.review_finding(
                finding_id,
                decision,
                comment=comment.strip() or None,
                reviewer=reviewer.strip() or None,
            )
    except ApiError as error:
        common.show_api_error(error)
        return

    common.clear_cache()  # every view must show the new status
    new_status = (result.get("finding") or {}).get("review_status", decision)
    st.session_state[FLASH_KEY] = f"Recorded: finding #{finding_id} {decision}. Its review status is now {new_status}."
    st.rerun()