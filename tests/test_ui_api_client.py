"""Tests for the dashboard's HTTP client and for the rules the dashboard is built on.

Four groups:
  1. Client unit tests with a fake HTTP session (no network, no server).
  2. Contract tests: the client talks to the REAL FastAPI app in-process, against a temporary
     database, so any drift between what the UI expects and what the API returns fails here.
  3. Static tests that read the UI source with `ast`: the UI must not import backend code or
     mention backend secrets, must not render untrusted text as markup, and its copies of API
     facts (detection methods, error codes, view names) must match the API.
  4. Tests for the pure helpers in ui/common.py.

Nothing here renders a Streamlit page: page behavior is checked by hand.
"""

import ast
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.ai.provider import ClassificationProvider, ClassificationSource
from app.api.routes.evaluation import get_report_path
from app.api.routes.findings import DetectionMethod
from app.evaluation import harness
from app.main import app
from app.services.ingestion import IngestionError
from app.workflows import scan_graph
from ui.api_client import ApiClient, ApiError, ApiErrorKind, UiSettings

BASE_URL = "http://api.test"
ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"
SETTINGS_KEYS = {"api_url", "connect_timeout_seconds", "read_timeout_seconds"}

# --- Minimal valid payloads (what the client requires, nothing more) ------------------------

HEALTH_PAYLOAD = {"status": "ok"}
METRICS_PAYLOAD = {
    "dataset_count": 0,
    "scan_count": 0,
    "scans_by_status": {},
    "finding_count": 0,
    "findings_by_severity": {},
    "findings_by_category": {},
    "findings_by_review_status": {},
    "pending_high_severity_count": 0,
    "review_decision_count": 0,
    "reviewed_finding_percent": None,
}
PAGE_PAYLOAD = {"items": [], "total": 0}
DATASET_PAYLOAD = {
    "id": 7,
    "name": "d",
    "row_count": 1,
    "column_count": 1,
    "created_at": "2026-01-01T00:00:00Z",
    "scan_count": 1,
    "finding_count": 0,
    "pending_review_count": 0,
    "latest_scan_id": 3,
    "latest_scan_status": "completed",
}
COLUMNS_PAYLOAD = {"dataset_id": 7, "row_count": 1, "columns": []}
SCAN_PAYLOAD = {
    "id": 3,
    "dataset_id": 7,
    "status": "completed",
    "started_at": "2026-01-01T00:00:00Z",
    "summary": None,
}
UPLOAD_PAYLOAD = {
    "dataset_id": 7,
    "scan_run_id": 3,
    "status": "completed",
    "finding_count": 0,
    "findings": [],
}
FINDING_PAYLOAD = {"id": 9, "review_status": "pending", "review_decisions": []}
REVIEW_PAYLOAD = {"decision": {}, "finding": {}}


# --- Test doubles -------------------------------------------------------------------------------


class FakeResponse:
    """The two members ApiClient uses from a requests response."""

    def __init__(self, status_code: int = 200, payload: Any = None, *, invalid_json: bool = False) -> None:
        self.status_code = status_code
        self._payload = payload
        self._invalid_json = invalid_json

    def json(self) -> Any:
        if self._invalid_json:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Records every request and returns (or raises) one fixed outcome."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.calls.append({"method": method, "url": url, **kwargs})
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def make_client(outcome: Any, **kwargs: Any) -> tuple[ApiClient, FakeSession]:
    session = FakeSession(outcome)
    return ApiClient(BASE_URL, session=session, **kwargs), session  # type: ignore[arg-type]


def api_error(call: Callable[[], Any]) -> ApiError:
    with pytest.raises(ApiError) as excinfo:
        call()
    return excinfo.value


class AppSession:
    """Adapts the FastAPI TestClient to the one method ApiClient calls on a requests session."""

    def __init__(self, client: TestClient) -> None:
        self._client = client

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Any = None,
        json: Any = None,
        files: Any = None,
        data: Any = None,
        timeout: Any = None,
    ) -> Any:
        return self._client.request(
            method, url.removeprefix(BASE_URL), params=params, json=json, files=files, data=data
        )


@pytest.fixture(autouse=True)
def clean_ui_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VERITYGRID_API_URL",
        "VERITYGRID_CONNECT_TIMEOUT_SECONDS",
        "VERITYGRID_READ_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def api(client: TestClient) -> ApiClient:
    """The dashboard client wired to the real API (tests/conftest.py gives a temporary database)."""
    return ApiClient(BASE_URL, session=AppSession(client))  # type: ignore[arg-type]


# =============================================================================================
# 1. Settings
# =============================================================================================


def test_settings_defaults() -> None:
    settings = UiSettings(_env_file=None)

    assert settings.api_url == "http://127.0.0.1:8000"
    assert settings.connect_timeout_seconds == 3.0
    assert settings.read_timeout_seconds == 30.0


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITYGRID_API_URL", "http://example.test:9000/")
    monkeypatch.setenv("VERITYGRID_CONNECT_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("VERITYGRID_READ_TIMEOUT_SECONDS", "45")

    settings = UiSettings(_env_file=None)

    assert settings.api_url == "http://example.test:9000"
    assert settings.connect_timeout_seconds == 5.0
    assert settings.read_timeout_seconds == 45.0


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        pytest.param("  http://localhost:8000/  ", "http://localhost:8000", id="spaces-and-slash"),
        pytest.param("https://api.example.test/v1/", "https://api.example.test/v1", id="path-prefix"),
        pytest.param("http://localhost:8000///", "http://localhost:8000", id="many-slashes"),
    ],
)
def test_settings_normalize_the_url(given: str, expected: str) -> None:
    assert UiSettings(_env_file=None, api_url=given).api_url == expected


@pytest.mark.parametrize(
    "value",
    [
        "ftp://example.test",
        "not a url",
        "",
        "http://",
        "http://user:pass@example.test",
        "http://user@example.test",
        "http://example.test?x=1",
        "http://example.test#frag",
    ],
    ids=["ftp", "not-a-url", "empty", "no-host", "credentials", "user-only", "query", "fragment"],
)
def test_settings_reject_unusable_urls(value: str) -> None:
    with pytest.raises(ValidationError):
        UiSettings(_env_file=None, api_url=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"connect_timeout_seconds": 0.1},
        {"connect_timeout_seconds": 31},
        {"read_timeout_seconds": 0.5},
        {"read_timeout_seconds": 301},
    ],
    ids=["connect-too-small", "connect-too-large", "read-too-small", "read-too-large"],
)
def test_settings_reject_out_of_range_timeouts(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        UiSettings(_env_file=None, **kwargs)


def test_settings_from_an_env_file_ignore_backend_secrets(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=SECRET-KEY-VALUE\n"
        "AI_PROVIDER=gemini\n"
        "DATABASE_URL=sqlite:///x.db\n"
        "VERITYGRID_API_URL=http://127.0.0.1:8123\n",
        encoding="utf-8",
    )

    settings = UiSettings(_env_file=str(env_file))

    assert settings.api_url == "http://127.0.0.1:8123"
    assert set(settings.model_dump()) == SETTINGS_KEYS
    assert "SECRET" not in repr(settings)


# =============================================================================================
# 2. Transport failures
# =============================================================================================


@pytest.mark.parametrize(
    ("error", "kind", "message"),
    [
        pytest.param(
            requests.ConnectTimeout("SECRET-TRANSPORT"),
            ApiErrorKind.UNREACHABLE,
            "Could not connect to the API in time.",
            id="connect-timeout",
        ),
        pytest.param(
            requests.ReadTimeout("SECRET-TRANSPORT"),
            ApiErrorKind.TIMEOUT,
            "The API took too long to respond.",
            id="read-timeout",
        ),
        pytest.param(
            requests.Timeout("SECRET-TRANSPORT"),
            ApiErrorKind.TIMEOUT,
            "The API took too long to respond.",
            id="timeout",
        ),
        pytest.param(
            requests.ConnectionError("SECRET-TRANSPORT"),
            ApiErrorKind.UNREACHABLE,
            "Could not connect to the API.",
            id="connection-error",
        ),
        pytest.param(
            requests.TooManyRedirects("SECRET-TRANSPORT"),
            ApiErrorKind.UNREACHABLE,
            "The request to the API failed.",
            id="other-request-error",
        ),
    ],
)
def test_transport_errors_become_api_errors(error: Exception, kind: ApiErrorKind, message: str) -> None:
    client, _ = make_client(error)

    raised = api_error(client.health)

    assert raised.kind is kind
    assert raised.message == message
    assert raised.status is None
    assert "SECRET" not in str(raised)


def test_unexpected_exceptions_are_not_swallowed() -> None:
    client, _ = make_client(RuntimeError("a bug, not a network problem"))

    with pytest.raises(RuntimeError):
        client.health()


# =============================================================================================
# 3. Response handling
# =============================================================================================


def test_a_json_object_is_returned() -> None:
    client, _ = make_client(FakeResponse(200, HEALTH_PAYLOAD))

    assert client.health() == HEALTH_PAYLOAD


def test_a_success_response_that_is_not_json_is_invalid() -> None:
    client, _ = make_client(FakeResponse(200, invalid_json=True))

    raised = api_error(client.health)

    assert raised.kind is ApiErrorKind.INVALID_RESPONSE
    assert raised.status == 200
    assert raised.message == "The API returned a response that is not valid JSON."


@pytest.mark.parametrize("payload", [None, 42, "text", True], ids=["null", "number", "string", "boolean"])
def test_scalar_json_is_invalid(payload: Any) -> None:
    client, _ = make_client(FakeResponse(200, payload))

    assert api_error(client.health).kind is ApiErrorKind.INVALID_RESPONSE


def test_a_list_where_an_object_is_expected_is_invalid() -> None:
    client, _ = make_client(FakeResponse(200, []))

    raised = api_error(client.health)

    assert raised.kind is ApiErrorKind.INVALID_RESPONSE
    assert "the health check" in raised.message


def test_error_responses_use_the_api_error_shape() -> None:
    body = {"error": {"code": "not_found", "message": "Dataset 999 was not found."}}
    client, _ = make_client(FakeResponse(404, body))

    raised = api_error(lambda: client.get_dataset(999))

    assert raised.kind is ApiErrorKind.HTTP_ERROR
    assert (raised.status, raised.code) == (404, "not_found")
    assert raised.message == "Dataset 999 was not found."
    assert raised.issues == []


def test_validation_issues_are_kept() -> None:
    body = {
        "error": {
            "code": "validation_error",
            "message": "The request failed validation.",
            "issues": [{"location": "body.decision", "message": "Input should be 'approved' or 'rejected'"}],
        }
    }
    client, _ = make_client(FakeResponse(422, body))

    raised = api_error(lambda: client.review_finding(1, "pending"))

    assert (raised.status, raised.code) == (422, "validation_error")
    assert raised.issues == [("body.decision", "Input should be 'approved' or 'rejected'")]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(FakeResponse(500, {"detail": "SECRET-BODY"}), id="detail-object"),
        pytest.param(FakeResponse(500, "SECRET-BODY"), id="json-string"),
        pytest.param(FakeResponse(500, ["SECRET-BODY"]), id="json-list"),
        pytest.param(FakeResponse(500, {"error": "SECRET-BODY"}), id="error-is-a-string"),
        pytest.param(FakeResponse(500, {"error": {"code": 5, "message": "SECRET-BODY"}}), id="code-not-a-string"),
        pytest.param(FakeResponse(500, {"error": {"message": "SECRET-BODY"}}), id="no-code"),
        pytest.param(FakeResponse(500, invalid_json=True), id="not-json"),
    ],
)
def test_unrecognised_error_bodies_are_not_echoed(response: FakeResponse) -> None:
    client, _ = make_client(response)

    raised = api_error(client.health)

    assert raised.kind is ApiErrorKind.HTTP_ERROR
    assert raised.code == "unexpected_error_body"
    assert raised.message == "The API returned HTTP 500."
    assert "SECRET" not in str(raised)


def test_malformed_issue_entries_are_ignored() -> None:
    body = {
        "error": {
            "code": "validation_error",
            "message": "m",
            "issues": ["bad", {"location": "query.limit", "message": "too small"}, 5, {"message": "x"}],
        }
    }
    client, _ = make_client(FakeResponse(422, body))

    assert api_error(client.health).issues == [("query.limit", "too small"), ("", "x")]


def test_missing_issues_are_an_empty_list() -> None:
    body = {"error": {"code": "validation_error", "message": "m", "issues": None}}
    client, _ = make_client(FakeResponse(422, body))

    assert api_error(client.health).issues == []


@pytest.mark.parametrize("status", [400, 404, 422, 500, 503])
def test_every_error_status_is_an_http_error(status: int) -> None:
    body = {"error": {"code": "some_code", "message": "Something happened."}}
    client, _ = make_client(FakeResponse(status, body))

    raised = api_error(client.health)

    assert (raised.kind, raised.status, raised.code) == (ApiErrorKind.HTTP_ERROR, status, "some_code")


# =============================================================================================
# 4. Requests the client sends
# =============================================================================================


def test_urls_are_joined_and_timeouts_are_passed() -> None:
    session = FakeSession(FakeResponse(200, HEALTH_PAYLOAD))
    client = ApiClient("http://api.test/", connect_timeout=1.5, read_timeout=9, session=session)  # type: ignore[arg-type]

    client.health()

    call = session.calls[0]
    assert client.base_url == "http://api.test"
    assert (call["method"], call["url"]) == ("GET", "http://api.test/health")
    assert call["timeout"] == (1.5, 9)
    assert call["params"] == {}


def test_default_timeouts() -> None:
    client, session = make_client(FakeResponse(200, HEALTH_PAYLOAD))

    client.health()

    assert session.calls[0]["timeout"] == (3.0, 30.0)


def test_list_datasets_sends_paging() -> None:
    client, session = make_client(FakeResponse(200, PAGE_PAYLOAD))

    client.list_datasets(limit=5, offset=10)

    assert session.calls[0]["params"] == {"limit": 5, "offset": 10}


@pytest.mark.parametrize(
    ("call", "path", "payload"),
    [
        pytest.param(lambda c: c.get_dataset(7), "/datasets/7", DATASET_PAYLOAD, id="dataset"),
        pytest.param(lambda c: c.get_columns(7), "/datasets/7/columns", COLUMNS_PAYLOAD, id="columns"),
        pytest.param(lambda c: c.get_scan(3), "/scans/3", SCAN_PAYLOAD, id="scan"),
        pytest.param(lambda c: c.get_finding(9), "/findings/9", FINDING_PAYLOAD, id="finding"),
        pytest.param(lambda c: c.metrics_summary(), "/metrics/summary", METRICS_PAYLOAD, id="metrics"),
        pytest.param(lambda c: c.list_datasets(), "/datasets", PAGE_PAYLOAD, id="datasets"),
    ],
)
def test_read_endpoints_use_the_expected_paths(call: Callable[[ApiClient], Any], path: str, payload: Any) -> None:
    client, session = make_client(FakeResponse(200, payload))

    call(client)

    assert (session.calls[0]["method"], session.calls[0]["url"]) == ("GET", f"{BASE_URL}{path}")


@pytest.mark.parametrize("value", ["1/../x", "abc", "1;2", ""], ids=["traversal", "letters", "semicolon", "empty"])
def test_ids_must_be_integers(value: str) -> None:
    client, session = make_client(FakeResponse(200, DATASET_PAYLOAD))

    with pytest.raises(ValueError):
        client.get_dataset(value)  # type: ignore[arg-type]

    assert session.calls == []  # nothing was sent, so a path cannot be injected through an id


def test_list_findings_sends_only_the_filters_given() -> None:
    client, session = make_client(FakeResponse(200, PAGE_PAYLOAD))

    client.list_findings(limit=25, offset=0, severity="high", detection_method="regex_email_pattern")

    assert session.calls[0]["params"] == {
        "limit": 25,
        "offset": 0,
        "severity": "high",
        "detection_method": "regex_email_pattern",
    }


def test_list_findings_sends_every_filter() -> None:
    client, session = make_client(FakeResponse(200, PAGE_PAYLOAD))

    client.list_findings(
        limit=10,
        offset=20,
        dataset_id=1,
        scan_id=2,
        severity="low",
        review_status="pending",
        category="missing_values",
        detection_method="null_percent_threshold",
    )

    assert session.calls[0]["params"] == {
        "limit": 10,
        "offset": 20,
        "dataset_id": 1,
        "scan_id": 2,
        "severity": "low",
        "review_status": "pending",
        "category": "missing_values",
        "detection_method": "null_percent_threshold",
    }


@pytest.mark.parametrize(
    ("name", "expected_data"),
    [
        pytest.param("scan name", {"dataset_name": "scan name"}, id="with-name"),
        pytest.param(None, None, id="without-name"),
    ],
)
def test_upload_sends_the_file_and_the_optional_name(name: str | None, expected_data: Any) -> None:
    client, session = make_client(FakeResponse(201, UPLOAD_PAYLOAD))

    client.upload_scan("a.csv", b"a\n1\n", dataset_name=name)

    call = session.calls[0]
    assert (call["method"], call["url"]) == ("POST", f"{BASE_URL}/scans")
    assert call["files"] == {"file": ("a.csv", b"a\n1\n", "text/csv")}
    assert call["data"] == expected_data


@pytest.mark.parametrize(
    ("kwargs", "expected_body"),
    [
        pytest.param(
            {"comment": "checked", "reviewer": "tester"},
            {"decision": "approved", "comment": "checked", "reviewer": "tester"},
            id="everything",
        ),
        pytest.param({}, {"decision": "approved"}, id="decision-only"),
        pytest.param({"comment": "", "reviewer": None}, {"decision": "approved"}, id="blank-values-left-out"),
    ],
)
def test_review_sends_only_what_was_given(kwargs: dict[str, Any], expected_body: dict[str, Any]) -> None:
    client, session = make_client(FakeResponse(201, REVIEW_PAYLOAD))

    client.review_finding(5, "approved", **kwargs)

    call = session.calls[0]
    assert (call["method"], call["url"]) == ("POST", f"{BASE_URL}/findings/5/review")
    assert call["json"] == expected_body


# =============================================================================================
# 5. Response shapes the client requires
# =============================================================================================

INVALID_SHAPES = [
    pytest.param(lambda c: c.health(), {}, id="health-missing-status"),
    pytest.param(lambda c: c.health(), [], id="health-wrong-type"),
    pytest.param(lambda c: c.metrics_summary(), {"dataset_count": 1}, id="metrics-missing-keys"),
    pytest.param(lambda c: c.list_datasets(), {"items": []}, id="datasets-missing-total"),
    pytest.param(lambda c: c.list_datasets(), {"items": "x", "total": 0}, id="datasets-items-not-a-list"),
    pytest.param(lambda c: c.list_datasets(), {"items": [], "total": "0"}, id="datasets-total-not-an-int"),
    pytest.param(lambda c: c.get_dataset(1), {"id": 1}, id="dataset-missing-keys"),
    pytest.param(lambda c: c.get_columns(1), {"dataset_id": 1}, id="columns-missing-keys"),
    pytest.param(
        lambda c: c.get_columns(1), {"dataset_id": 1, "row_count": 1, "columns": {}}, id="columns-not-a-list"
    ),
    pytest.param(
        lambda c: c.get_scan(1),
        {"id": 1, "dataset_id": 1, "status": "completed", "started_at": "x"},
        id="scan-missing-summary",
    ),
    pytest.param(
        lambda c: c.upload_scan("a.csv", b"x"),
        {"dataset_id": 1, "scan_run_id": 1, "status": "completed", "finding_count": 0},
        id="upload-missing-findings",
    ),
    pytest.param(lambda c: c.list_findings(), {"total": 0}, id="findings-missing-items"),
    pytest.param(
        lambda c: c.get_finding(1), {"id": 1, "review_status": "pending"}, id="finding-missing-history"
    ),
    pytest.param(lambda c: c.review_finding(1, "approved"), {"decision": {}}, id="review-missing-finding"),
]
VALID_SHAPES = [
    pytest.param(lambda c: c.health(), HEALTH_PAYLOAD, id="health"),
    pytest.param(lambda c: c.metrics_summary(), METRICS_PAYLOAD, id="metrics"),
    pytest.param(lambda c: c.list_datasets(), PAGE_PAYLOAD, id="datasets"),
    pytest.param(lambda c: c.get_dataset(7), DATASET_PAYLOAD, id="dataset"),
    pytest.param(lambda c: c.get_columns(7), COLUMNS_PAYLOAD, id="columns"),
    pytest.param(lambda c: c.get_scan(3), SCAN_PAYLOAD, id="scan"),
    pytest.param(lambda c: c.upload_scan("a.csv", b"x"), UPLOAD_PAYLOAD, id="upload"),
    pytest.param(lambda c: c.list_findings(), PAGE_PAYLOAD, id="findings"),
    pytest.param(lambda c: c.get_finding(9), FINDING_PAYLOAD, id="finding"),
    pytest.param(lambda c: c.review_finding(9, "approved"), REVIEW_PAYLOAD, id="review"),
]


@pytest.mark.parametrize(("call", "payload"), INVALID_SHAPES)
def test_invalid_shapes_are_rejected(call: Callable[[ApiClient], Any], payload: Any) -> None:
    client, _ = make_client(FakeResponse(200, payload))

    raised = api_error(lambda: call(client))

    assert raised.kind is ApiErrorKind.INVALID_RESPONSE
    assert "unexpected response" in raised.message
    assert raised.status is None


@pytest.mark.parametrize(("call", "payload"), VALID_SHAPES)
def test_valid_shapes_are_accepted(call: Callable[[ApiClient], Any], payload: Any) -> None:
    client, _ = make_client(FakeResponse(200, payload))

    assert call(client) == payload


# =============================================================================================
# 6. Contract tests against the real API
# =============================================================================================


def test_contract_empty_database(api: ApiClient) -> None:
    assert api.health()["status"] == "ok"
    metrics = api.metrics_summary()
    assert metrics["dataset_count"] == 0
    assert metrics["reviewed_finding_percent"] is None
    assert api.list_datasets()["total"] == 0
    assert api.list_findings()["total"] == 0


def test_contract_full_flow(api: ApiClient, mixed_csv: bytes) -> None:
    result = api.upload_scan("mixed.csv", mixed_csv, dataset_name="mixed_sample")
    assert (result["status"], result["finding_count"]) == ("completed", 3)
    dataset_id, scan_id = result["dataset_id"], result["scan_run_id"]

    datasets = api.list_datasets()
    assert datasets["total"] == 1
    assert datasets["items"][0]["name"] == "mixed_sample"

    dataset = api.get_dataset(dataset_id)
    assert (dataset["finding_count"], dataset["pending_review_count"]) == (3, 3)
    assert dataset["latest_scan_status"] == "completed"
    assert len(api.get_columns(dataset_id)["columns"]) == 5
    assert api.get_scan(scan_id)["status"] == "completed"

    assert api.list_findings(dataset_id=dataset_id)["total"] == 3
    assert api.list_findings(severity="high")["total"] == 1
    assert api.list_findings(review_status="pending")["total"] == 3
    assert api.list_findings(detection_method="null_percent_threshold")["total"] == 2
    email = api.list_findings(detection_method="regex_email_pattern")
    assert email["total"] == 1
    finding_id = email["items"][0]["id"]
    assert api.get_finding(finding_id)["review_decisions"] == []

    review = api.review_finding(finding_id, "approved", comment="checked", reviewer="tester")
    assert review["finding"]["review_status"] == "approved"
    assert review["decision"]["reviewer"] == "tester"
    assert api.get_finding(finding_id)["review_decisions"][0]["comment"] == "checked"

    metrics = api.metrics_summary()
    assert metrics["findings_by_review_status"] == {"pending": 2, "approved": 1, "rejected": 0}
    assert metrics["review_decision_count"] == 1
    assert metrics["reviewed_finding_percent"] == pytest.approx(33.33)


ERROR_CASES = [
    pytest.param(lambda a: a.get_dataset(999), 404, "not_found", None, id="dataset-not-found"),
    pytest.param(lambda a: a.get_scan(999), 404, "not_found", None, id="scan-not-found"),
    pytest.param(lambda a: a.get_finding(999), 404, "not_found", None, id="finding-not-found"),
    pytest.param(lambda a: a.get_columns(999), 404, "not_found", None, id="columns-not-found"),
    pytest.param(lambda a: a.upload_scan("bad.csv", b"a,a\n1,2\n"), 422, "invalid_header", None, id="duplicate-header"),
    pytest.param(lambda a: a.upload_scan("empty.csv", b""), 422, "empty_file", None, id="empty-file"),
    pytest.param(lambda a: a.upload_scan("x.csv", b"a,b\n1\n"), 422, "malformed_csv", None, id="ragged-row"),
    pytest.param(lambda a: a.upload_scan("x.csv", b"name\ncaf\xe9\n"), 422, "invalid_encoding", None, id="latin-1"),
    pytest.param(
        lambda a: a.list_findings(detection_method="bogus"),
        422,
        "validation_error",
        "query.detection_method",
        id="bad-method",
    ),
    pytest.param(
        lambda a: a.list_findings(severity="critical"),
        422,
        "validation_error",
        "query.severity",
        id="bad-severity",
    ),
    pytest.param(
        lambda a: a.review_finding(1, "pending"),
        422,
        "validation_error",
        "body.decision",
        id="bad-decision",
    ),
    pytest.param(lambda a: a.review_finding(999, "approved"), 404, "not_found", None, id="review-missing-finding"),
]


@pytest.mark.parametrize(("call", "status", "code", "issue_location"), ERROR_CASES)
def test_contract_errors(
    api: ApiClient,
    call: Callable[[ApiClient], Any],
    status: int,
    code: str,
    issue_location: str | None,
) -> None:
    raised = api_error(lambda: call(api))

    assert raised.kind is ApiErrorKind.HTTP_ERROR
    assert (raised.status, raised.code) == (status, code)
    assert raised.message
    if issue_location is not None:
        assert [location for location, _ in raised.issues] == [issue_location]


def test_contract_file_too_large(api: ApiClient, set_limits: Callable[..., None]) -> None:
    set_limits(max_bytes=3)

    raised = api_error(lambda: api.upload_scan("a.csv", b"a\n1\n"))

    assert (raised.status, raised.code) == (413, "file_too_large")
    assert raised.message == "The file exceeds the limit of 3 bytes."


def test_contract_scan_failed(api: ApiClient, mixed_csv: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("SECRET-DETAIL")

    monkeypatch.setattr(scan_graph, "run_detectors", explode)

    raised = api_error(lambda: api.upload_scan("mixed.csv", mixed_csv))

    assert (raised.status, raised.code) == (500, "scan_failed")
    assert "RuntimeError" in raised.message
    assert "SECRET" not in raised.message


def test_upload_error_hints_cover_the_api_codes() -> None:
    hints = constant(UI_DIR / "views" / "upload.py", "ERROR_HINTS")
    ingestion_codes = {cls.code for cls in IngestionError.__subclasses__()}

    assert ingestion_codes <= set(hints)  # every ingestion failure has a hint
    assert set(hints) <= ingestion_codes | {"validation_error", "scan_failed"}  # and no hint is for a code the API never sends
    assert all(isinstance(text, str) and text for text in hints.values())


# =============================================================================================
# 7. Static rules about the UI source
# =============================================================================================


def ui_files() -> list[Path]:
    return sorted(UI_DIR.rglob("*.py"))


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def constant(path: Path, name: str) -> Any:
    """The literal value of a module-level constant, read from source without importing it."""
    for node in parse(path).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} defines no constant {name}")


def imported_modules(tree: ast.Module) -> set[str]:
    """Top-level names of everything a module imports."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def navigation_targets(tree: ast.Module) -> list[str]:
    """View names passed to go_to_view(...), directly or through a button's on_click and args."""

    def is_go_to_view(node: ast.AST) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr == "go_to_view") or (
            isinstance(node, ast.Name) and node.id == "go_to_view"
        )

    targets: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if is_go_to_view(node.func) and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                targets.append(first.value)
        if any(keyword.arg == "on_click" and is_go_to_view(keyword.value) for keyword in node.keywords):
            for keyword in node.keywords:
                if keyword.arg == "args" and isinstance(keyword.value, ast.Tuple):
                    targets += [
                        item.value
                        for item in keyword.value.elts
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    ]
    return targets


def test_expected_ui_files_exist_and_parse() -> None:
    expected = {
        "api_client.py",
        "common.py",
        "streamlit_app.py",
        "views/catalog.py",
        "views/upload.py",
        "views/findings.py",
        "views/review.py",
        "views/evaluation.py",
    }

    found = {path.relative_to(UI_DIR).as_posix() for path in ui_files()}

    assert expected <= found
    for path in ui_files():
        parse(path)  # a syntax error fails here with the file name


def test_views_registry() -> None:
    views = constant(UI_DIR / "streamlit_app.py", "VIEWS")

    assert list(views) == [
        "Overview",
        "Data catalog",
        "Upload and scan",
        "Findings explorer",
        "Review queue",
        "Evaluation",
    ]
    assert views["Overview"] is None
    for name, module in views.items():
        if module is None:
            continue
        path = UI_DIR / "views" / f"{module}.py"
        assert path.exists(), f"{name} points to a missing file"
        render = [n for n in parse(path).body if isinstance(n, ast.FunctionDef) and n.name == "render"]
        assert len(render) == 1, f"{path.name} must define exactly one render function"
        assert [arg.arg for arg in render[0].args.args] == ["client"]


def test_detection_methods_match_the_api() -> None:
    methods = constant(UI_DIR / "views" / "findings.py", "DETECTION_METHODS")

    assert set(methods) == {method.value for method in DetectionMethod}
    assert all(isinstance(label, str) and label for label in methods.values())
    assert len(set(methods.values())) == len(methods)  # labels must be distinguishable


def test_review_selection_key_matches() -> None:
    writer = constant(UI_DIR / "views" / "findings.py", "REVIEW_SELECTION_KEY")
    reader = constant(UI_DIR / "views" / "review.py", "SELECTION_KEY")

    assert writer == reader == "review_selected_finding"


def test_navigation_targets_exist() -> None:
    views = constant(UI_DIR / "streamlit_app.py", "VIEWS")
    targets: list[str] = []
    for path in ui_files():
        targets += navigation_targets(parse(path))

    assert {"Review queue", "Findings explorer", "Data catalog"} <= set(targets)
    assert set(targets) <= set(views), f"unknown view names: {set(targets) - set(views)}"


def test_ui_does_not_import_backend_code() -> None:
    forbidden = {
        "app",
        "sqlalchemy",
        "google",
        "langgraph",
        "langchain",
        "langsmith",
        "fastapi",
        "uvicorn",
        "starlette",
    }

    for path in ui_files():
        used = imported_modules(parse(path)) & forbidden
        assert not used, f"{path.relative_to(ROOT).as_posix()} imports backend code: {sorted(used)}"


def test_ui_never_mentions_backend_secrets() -> None:
    forbidden = (
        "gemini_api_key",
        "ai_provider",
        "database_url",
        "google_api_key",
        "langsmith_",
        "langchain_",
        "os.environ",
        "getenv",
        "import os",
    )

    for path in ui_files():
        text = path.read_text(encoding="utf-8").lower()
        found = [item for item in forbidden if item in text]
        assert not found, f"{path.relative_to(ROOT).as_posix()} mentions {found}"


def test_ui_contains_no_hardcoded_secrets() -> None:
    patterns = (
        r"AIza[0-9A-Za-z_\-]{20,}",
        r"sk-[A-Za-z0-9]{20,}",
        r"(?i)bearer\s+[a-z0-9._\-]{20,}",
        r"(?i)api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
    )

    for path in ui_files():
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            assert not re.search(pattern, text), f"{path.relative_to(ROOT).as_posix()} matches {pattern}"


def test_untrusted_text_is_never_rendered_as_markup() -> None:
    for path in ui_files():
        text = path.read_text(encoding="utf-8")
        assert "unsafe_allow_html" not in text, f"{path.name} enables raw HTML"
        for node in ast.walk(parse(path)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            name = node.func.attr
            assert name not in {"write", "html", "write_stream"}, f"{path.name} calls {name}"
            if name == "markdown":
                first = node.args[0] if node.args else None
                assert isinstance(first, ast.Constant) and isinstance(first.value, str), (
                    f"{path.name} line {node.lineno}: markdown must receive a fixed string, not built text"
                )


def test_only_the_client_module_touches_the_network() -> None:
    network = {"requests", "httpx", "httpx2", "urllib", "urllib3", "http", "socket", "aiohttp"}

    for path in ui_files():
        if path.name == "api_client.py":
            continue
        used = imported_modules(parse(path)) & network
        assert not used, f"{path.relative_to(ROOT).as_posix()} imports {sorted(used)}"


# =============================================================================================
# 8. Pure helpers in ui/common.py
# =============================================================================================


@pytest.fixture(scope="module")
def common() -> Any:
    pytest.importorskip("streamlit")
    from ui import common as module

    return module


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("2026-10-24T07:17:29.839689Z", "2026-10-24 07:17 UTC", id="zulu"),
        pytest.param("2026-10-24T12:47:00+05:30", "2026-10-24 07:17 UTC", id="offset-converted-to-utc"),
        pytest.param(None, "-", id="none"),
        pytest.param("garbage", "-", id="garbage"),
        pytest.param(123, "-", id="number"),
        pytest.param("", "-", id="empty"),
    ],
)
def test_format_utc(common: Any, value: object, expected: str) -> None:
    assert common.format_utc(value) == expected


def test_describe_classification(common: Any) -> None:
    mock = common.describe_classification("mock")
    llm = common.describe_classification("llm")

    assert common.describe_classification(None) == "Not classified"
    assert mock.startswith("Mock") and "not an LLM" in mock
    assert "unverified" in llm and "human" in llm
    assert "No suggestion" in common.describe_classification("none")
    assert common.describe_classification("something-else") == "Unknown source"
    assert mock != llm  # mock output must never read like model output


def test_to_frame(common: Any) -> None:
    rows = [{"id": 1, "name": "a"}, {"id": 2}]

    frame = common.to_frame(rows, {"id": "ID", "name": "Name", "other": "Other"})

    assert list(frame.columns) == ["ID", "Name", "Other"]
    assert frame.shape == (2, 3)
    assert frame["Name"].isna().tolist() == [False, True]  # a missing key becomes an empty cell
    assert frame["Other"].isna().all()


def test_to_frame_with_no_rows(common: Any) -> None:
    frame = common.to_frame([], {"id": "ID", "name": "Name"})

    assert frame.shape == (0, 2)
    assert list(frame.columns) == ["ID", "Name"]


def test_with_column_label(common: Any) -> None:
    original = {"id": 1, "column_name": None}

    labelled = common.with_column_label(original)

    assert labelled == {"id": 1, "column_name": "(whole dataset)"}
    assert original == {"id": 1, "column_name": None}  # the input is not modified
    assert common.with_column_label({"id": 1})["column_name"] == "(whole dataset)"
    assert common.with_column_label({"column_name": ""})["column_name"] == "(whole dataset)"
    assert common.with_column_label({"column_name": "email"})["column_name"] == "email"


def test_common_constants(common: Any) -> None:
    assert common.WHOLE_DATASET_TEXT == "(whole dataset)"
    assert common.NOT_CLASSIFIED_TEXT == "Not classified"
    assert common.SEVERITY_ORDER == ("high", "medium", "low")
    assert set(common.CLASSIFICATION_SOURCE_TEXT) == {"mock", "llm", "none"}
    
    
# =============================================================================================
# 9. The evaluation endpoint and the Evaluation view (9B-2)
# =============================================================================================

EVAL_AVAILABLE = {
    "available": True,
    "reason": None,
    "file_modified_at": "2026-11-14T09:56:40Z",
    "report": {"report_version": 1},
}
EVAL_MISSING = {
    "available": False,
    "reason": "No evaluation report has been generated yet.",
    "file_modified_at": None,
    "report": None,
}
GOOD_ANSWER: dict[str, Any] = {
    "label": "internal",
    "explanation": "Operational data.",
    "supporting_refs": ["column_name"],
    "review_recommendation": "review_recommended",
}


class StubLlm(ClassificationProvider):
    """A stand-in LLM that always answers 'internal'. Never touches a network."""

    name = "llm-stub"
    source = ClassificationSource.LLM

    def classify(self, metadata: Any) -> Any:
        return GOOD_ANSWER


class DummyClient:
    base_url = "http://api.test"


class _Context:
    def __init__(self, owner: "Recorder") -> None:
        self.owner = owner

    def __enter__(self) -> "Recorder":
        return self.owner

    def __exit__(self, *exc_info: Any) -> bool:
        return False


class Recorder:
    """Stands in for the `streamlit` module: records every call, so a view can be 'drawn' without a browser."""

    def __init__(self, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] | None = None) -> None:
        self.calls = calls if calls is not None else []

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            if name == "columns":
                spec = args[0]
                return [Recorder(self.calls) for _ in range(spec if isinstance(spec, int) else len(spec))]
            if name in {"expander", "spinner", "container"}:
                return _Context(self)
            return None

        return call


def rendered_text(calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> str:
    parts: list[str] = []
    for _name, args, kwargs in calls:
        for value in (*args, *kwargs.values()):
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, pd.DataFrame):
                parts.append(value.to_csv(index=False))
    return "\n".join(parts)


def draw_view(module: Any, monkeypatch: pytest.MonkeyPatch, answer: Any) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    recorder = Recorder()
    monkeypatch.setattr(module, "st", recorder)

    def load(base_url: str, client: Any) -> Any:
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(module, "_load_report", load)
    module.render(DummyClient())
    return recorder.calls


def available(report: dict[str, Any]) -> dict[str, Any]:
    return {"available": True, "reason": None, "file_modified_at": "2026-11-14T09:56:40Z", "report": report}


@pytest.fixture(scope="module")
def eval_report() -> dict[str, Any]:
    """A real report built by the harness (deterministic and mock parts; no provider, no network)."""
    return harness.build_report(repeats=1, warmup=False)


@pytest.fixture(scope="module")
def evaluation_view() -> Any:
    pytest.importorskip("streamlit")
    from ui.views import evaluation

    return evaluation


@pytest.fixture()
def eval_report_path(tmp_path: Path, client: TestClient) -> Iterator[Path]:
    """Point the real API at a file inside tmp_path for one test."""
    path = tmp_path / "evaluation_report.json"
    app.dependency_overrides[get_report_path] = lambda: path
    yield path
    app.dependency_overrides.pop(get_report_path, None)


# --- Client ----------------------------------------------------------------------------------------


def test_evaluation_report_uses_the_expected_path_and_returns_the_answer() -> None:
    client, session = make_client(FakeResponse(200, EVAL_AVAILABLE))

    answer = client.evaluation_report()

    assert (session.calls[0]["method"], session.calls[0]["url"]) == ("GET", f"{BASE_URL}/evaluation/report")
    assert answer == EVAL_AVAILABLE


def test_evaluation_report_without_a_report_is_valid() -> None:
    client, _ = make_client(FakeResponse(200, EVAL_MISSING))

    assert client.evaluation_report() == EVAL_MISSING


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param([], id="list"),
        pytest.param({"available": True, "reason": None, "report": {}}, id="missing-key"),
        pytest.param({**EVAL_AVAILABLE, "available": "yes"}, id="available-not-a-boolean"),
        pytest.param({**EVAL_AVAILABLE, "report": None}, id="available-without-a-report"),
        pytest.param({**EVAL_MISSING, "report": {}}, id="unavailable-with-a-report"),
    ],
)
def test_evaluation_report_rejects_unexpected_shapes(payload: Any) -> None:
    client, _ = make_client(FakeResponse(200, payload))

    assert api_error(client.evaluation_report).kind is ApiErrorKind.INVALID_RESPONSE


# --- Contract tests against the real API -----------------------------------------------------------


def test_contract_evaluation_report_unavailable(api: ApiClient, eval_report_path: Path) -> None:
    answer = api.evaluation_report()

    assert answer["available"] is False and answer["report"] is None
    assert "No evaluation report has been generated yet" in answer["reason"]


def test_contract_evaluation_report_available(
    api: ApiClient, eval_report_path: Path, eval_report: dict[str, Any]
) -> None:
    eval_report_path.write_text(json.dumps(eval_report), encoding="utf-8")

    answer = api.evaluation_report()

    assert answer["available"] is True
    assert answer["report"]["fingerprints"] == eval_report["fingerprints"]
    assert answer["file_modified_at"] is not None


# --- Formatting helpers ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, "n/a", id="none-is-not-zero"),
        pytest.param(0, "0.000", id="zero-is-zero"),
        pytest.param(0.5, "0.500", id="half"),
        pytest.param(1, "1.000", id="one"),
        pytest.param(True, "n/a", id="boolean"),
        pytest.param("x", "n/a", id="text"),
    ],
)
def test_fmt_rate(evaluation_view: Any, value: object, expected: str) -> None:
    assert evaluation_view.fmt_rate(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, "n/a", id="none"),
        pytest.param([0.1, 0.9], "[0.10, 0.90]", id="list"),
        pytest.param((0.05, 1.0), "[0.05, 1.00]", id="tuple"),
        pytest.param([1], "n/a", id="wrong-length"),
        pytest.param(["a", "b"], "n/a", id="text"),
    ],
)
def test_fmt_interval(evaluation_view: Any, value: object, expected: str) -> None:
    assert evaluation_view.fmt_interval(value) == expected


def test_fmt_count(evaluation_view: Any) -> None:
    assert evaluation_view.fmt_count(3) == "3"
    assert evaluation_view.fmt_count(True) == "n/a"
    assert evaluation_view.fmt_count(2.5) == "n/a"
    assert evaluation_view.fmt_count(None) == "n/a"


def test_rate_table_shows_undefined_rates_as_n_a(evaluation_view: Any) -> None:
    report = {"support": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 5, "precision": None, "recall": None, "f1": None, "recall_ci95": None}

    table = evaluation_view.rate_table({"x": report}, "Name")

    assert table.to_dict("records") == [
        {
            "Name": "x", "Support": "0", "TP": "0", "FP": "0", "FN": "0", "TN": "5",
            "Precision": "n/a", "Recall": "n/a", "F1": "n/a", "Recall 95% CI": "n/a",
        }
    ]


def test_confusion_table(evaluation_view: Any) -> None:
    metrics = {
        "predicted_columns": ["a", "b", "no_prediction"],
        "matrix": {
            "a": {"a": 2, "b": 1, "no_prediction": 0},
            "b": {"a": 0, "b": 3, "no_prediction": 1},
        },
    }

    table = evaluation_view.confusion_table(metrics)

    assert list(table.columns) == ["Actual \\ Predicted", "a", "b", "no_prediction", "Total"]
    assert table.to_dict("records") == [
        {"Actual \\ Predicted": "a", "a": "2", "b": "1", "no_prediction": "0", "Total": "3"},
        {"Actual \\ Predicted": "b", "a": "0", "b": "3", "no_prediction": "1", "Total": "4"},
    ]


def test_latency_table(evaluation_view: Any) -> None:
    table = evaluation_view.latency_table({"n": 3, "min_ms": 1.0, "median_ms": 2.0, "max_ms": 3.5, "mean_ms": 2.1666})

    assert table.to_dict("records") == [{"Runs": "3", "Min ms": "1.0", "Median ms": "2.0", "Max ms": "3.5", "Mean ms": "2.2"}]


def test_view_helpers_accept_the_real_report_structure(evaluation_view: Any, eval_report: dict[str, Any]) -> None:
    detection = eval_report["deterministic"]["metrics"]
    mock = eval_report["mock"]["metrics"]

    categories = evaluation_view.rate_table(detection["per_category"], "Category")
    classes = evaluation_view.rate_table(mock["per_class"], "Class")
    confusion = evaluation_view.confusion_table(mock)

    assert list(categories["Category"]) == list(detection["per_category"])
    assert list(classes["Class"]) == list(mock["per_class"])
    assert list(confusion["Total"]) == [str(sum(row.values())) for row in mock["matrix"].values()]
    assert len(evaluation_view.latency_table(eval_report["deterministic"]["latency"])) == 1


# --- Drawing the view with a recording stand-in for streamlit ------------------------------------------------


def test_view_draws_a_real_report(evaluation_view: Any, monkeypatch: pytest.MonkeyPatch, eval_report: dict[str, Any]) -> None:
    calls = draw_view(evaluation_view, monkeypatch, available(eval_report))

    text = rendered_text(calls)
    names = [name for name, _args, _kwargs in calls]
    assert "error" not in names
    assert "could not be displayed" not in text
    assert eval_report["fingerprints"]["results"] in text
    assert "MOCK results" in text and "not an LLM" in text  # the mock section is labelled
    assert evaluation_view.fmt_rate(eval_report["mock"]["metrics"]["accuracy"]) in text
    assert evaluation_view.fmt_rate(eval_report["deterministic"]["metrics"]["micro"]["precision"]) in text
    assert eval_report["llm"]["reason"] in text  # the LLM section says why nothing was run
    assert "Not collected." in text  # and so does the review section


def test_view_shows_only_the_reason_when_there_is_no_report(evaluation_view: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = draw_view(evaluation_view, monkeypatch, EVAL_MISSING)

    text = rendered_text(calls)
    names = {name for name, _args, _kwargs in calls}
    assert EVAL_MISSING["reason"] in text
    assert evaluation_view.REGENERATE_COMMAND in text
    assert not names & {"metric", "dataframe", "bar_chart", "line_chart"}  # no numbers, tables or charts


def test_view_shows_a_genuine_llm_section_when_it_ran(evaluation_view: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    report = harness.build_report(repeats=1, warmup=False, llm_provider=StubLlm())

    calls = draw_view(evaluation_view, monkeypatch, available(report))

    text = rendered_text(calls)
    assert "single, non-reproducible run" in text
    assert report["llm"]["cost"]["reason"] in text  # cost is reported as not measured
    assert "llm-stub" in text
    assert "could not be displayed" not in text


def test_view_explains_an_api_error_without_drawing_numbers(evaluation_view: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ApiError] = []
    monkeypatch.setattr(evaluation_view.common, "show_api_error", seen.append)

    calls = draw_view(evaluation_view, monkeypatch, ApiError(ApiErrorKind.UNREACHABLE, "Could not connect to the API."))

    assert len(seen) == 1 and seen[0].kind is ApiErrorKind.UNREACHABLE
    assert not {name for name, _args, _kwargs in calls} & {"metric", "dataframe"}


def test_one_malformed_section_does_not_break_the_others(
    evaluation_view: Any, monkeypatch: pytest.MonkeyPatch, eval_report: dict[str, Any]
) -> None:
    broken = json.loads(json.dumps(eval_report))
    broken["deterministic"]["metrics"] = {}

    calls = draw_view(evaluation_view, monkeypatch, available(broken))

    text = rendered_text(calls)
    assert text.count("could not be displayed") == 1
    assert "MOCK results" in text  # the other sections were still drawn


# --- A guard against hard-coded results ----------------------------------------------------------------------------


def test_evaluation_view_has_no_hard_coded_metric_numbers() -> None:
    tree = parse(UI_DIR / "views" / "evaluation.py")
    numbers = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    ]

    assert not [value for value in numbers if isinstance(value, float)]
    assert set(numbers) <= {0, 1, 2, 3, 4, 5, 10}  # structural numbers only (column counts, cache lifetime)
    
# =============================================================================================
# 10. Detection rules in the upload (12-2c)
# =============================================================================================

RULES_CSV = b"sku,quantity,counted\nA1,5,2026-01-05\nA2,-12,2026-02-30\nA3,7,2026-03-01\n"
RULES_TEXT = json.dumps(
    {"numeric_rules": {"quantity": {"min_value": 0}}, "date_rules": {"counted": {"date_format": "%Y-%m-%d"}}}
)


class UploadedStub:
    name = "inv.csv"
    size = 10

    def getvalue(self) -> bytes:
        return b"a\n1\n"


class FakeUploadClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []
        self.error = error

    def upload_scan(self, filename: str, content: bytes, *, dataset_name: str | None = None, rules: str | None = None) -> Any:
        self.calls.append((filename, dataset_name, rules))
        if self.error is not None:
            raise self.error
        return {"scan_run_id": 1, "dataset_id": 1, "status": "completed", "finding_count": 0, "findings": []}


def upload_stub(infos: list[str] | None = None) -> Any:
    import contextlib
    from types import SimpleNamespace

    return SimpleNamespace(
        session_state={},
        spinner=lambda text: contextlib.nullcontext(),
        error=lambda text: None,
        info=infos.append if infos is not None else (lambda text: None),
    )


def literal_constant(path: Path, name: str) -> Any:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path.name}")


@pytest.fixture(scope="module")
def upload_view() -> Any:
    pytest.importorskip("streamlit")
    from ui.views import upload

    return upload


# --- The client ---------------------------------------------------------------------------------------


def test_upload_scan_sends_the_rules_as_a_form_field() -> None:
    client, session = make_client(FakeResponse(200, UPLOAD_PAYLOAD))

    client.upload_scan("a.csv", b"a\n1\n", rules=RULES_TEXT)

    assert session.calls[0]["data"] == {"rules": RULES_TEXT}


def test_upload_scan_sends_the_name_and_the_rules() -> None:
    client, session = make_client(FakeResponse(200, UPLOAD_PAYLOAD))

    client.upload_scan("a.csv", b"a\n1\n", dataset_name="inv", rules=RULES_TEXT)

    assert session.calls[0]["data"] == {"dataset_name": "inv", "rules": RULES_TEXT}


@pytest.mark.parametrize("rules", [None, ""], ids=["none", "empty"])
def test_upload_scan_without_rules_sends_no_form_data(rules: str | None) -> None:
    client, session = make_client(FakeResponse(200, UPLOAD_PAYLOAD))

    client.upload_scan("a.csv", b"a\n1\n", rules=rules)

    assert session.calls[0]["data"] is None


# --- Against the real API ---------------------------------------------------------------------------------


def test_contract_upload_with_rules(api: ApiClient) -> None:
    result = api.upload_scan("inv.csv", RULES_CSV, rules=RULES_TEXT)

    assert result["findings_by_category"] == {"numeric_rule_violation": 1, "invalid_date": 1}


def test_contract_invalid_rules_are_a_validation_error(api: ApiClient) -> None:
    raised = api_error(lambda: api.upload_scan("inv.csv", RULES_CSV, rules="not json"))

    assert (raised.status, raised.code) == (422, "validation_error")
    assert [location for location, _ in raised.issues] == ["body.rules"]


def test_contract_rules_for_an_absent_column_are_rejected_without_echo(api: ApiClient) -> None:
    rules = json.dumps({"numeric_rules": {"nope": {"min_value": 0}}})

    raised = api_error(lambda: api.upload_scan("inv.csv", RULES_CSV, rules=rules))

    assert raised.code == "validation_error"
    assert raised.issues[0][0] == "body.rules.numeric_rules"
    assert "nope" not in raised.message and all("nope" not in text for pair in raised.issues for text in pair)


# --- The view ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [(None, None), ("", None), ("  \n ", None), (' {"a": 1} ', '{"a": 1}')],
    ids=["none", "empty", "whitespace", "trimmed"],
)
def test_clean_rules_text(upload_view: Any, text: str | None, expected: str | None) -> None:
    assert upload_view.clean_rules_text(text) == expected


def test_the_view_sends_the_cleaned_name_and_rules(upload_view: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = upload_stub()
    monkeypatch.setattr(upload_view, "st", stub)
    monkeypatch.setattr(upload_view.common, "clear_cache", lambda: None)
    client = FakeUploadClient()

    upload_view._run_scan(client, UploadedStub(), "  my name ", f"  {RULES_TEXT}\n")

    assert client.calls == [("inv.csv", "my name", RULES_TEXT)]
    assert upload_view.RESULT_KEY in stub.session_state


def test_blank_rules_and_name_are_sent_as_none(upload_view: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload_view, "st", upload_stub())
    monkeypatch.setattr(upload_view.common, "clear_cache", lambda: None)
    client = FakeUploadClient()

    upload_view._run_scan(client, UploadedStub(), "", "   ")

    assert client.calls == [("inv.csv", None, None)]


def test_a_validation_error_shows_the_rules_hint(upload_view: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    infos: list[str] = []
    seen: list[ApiError] = []
    stub = upload_stub(infos)
    monkeypatch.setattr(upload_view, "st", stub)
    monkeypatch.setattr(upload_view.common, "show_api_error", seen.append)
    error = ApiError(
        ApiErrorKind.HTTP_ERROR,
        "The request failed validation.",
        status=422,
        code="validation_error",
        issues=[("body.rules", "The rules must be valid JSON.")],
    )

    upload_view._run_scan(FakeUploadClient(error), UploadedStub(), "", "oops")

    assert seen == [error]
    assert any("rules" in text for text in infos)
    assert upload_view.RESULT_KEY not in stub.session_state


def test_the_rules_limit_matches_the_api() -> None:
    from app.schemas.rules import MAX_RULES_CHARS

    assert literal_constant(UI_DIR / "views" / "upload.py", "RULES_MAX_CHARS") == MAX_RULES_CHARS


def test_the_example_rules_are_valid(upload_view: Any) -> None:
    from app.schemas.rules import parse_rules

    assert parse_rules(upload_view.EXAMPLE_RULES) is not None