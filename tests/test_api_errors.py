import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.services import query_service

EXPECTED_PATHS = {
    "/health",
    "/scans",
    "/scans/{scan_id}",
    "/datasets",
    "/datasets/{dataset_id}",
    "/datasets/{dataset_id}/columns",
    "/findings",
    "/findings/{finding_id}",
    "/findings/{finding_id}/review",
    "/metrics/summary",
    "/evaluation/report",
}


def issue_locations(response: Any) -> list[str]:
    return [issue["location"] for issue in response.json()["error"]["issues"]]


# --- Routing errors --------------------------------------------------------------


def test_unknown_route_returns_shared_error(client: TestClient) -> None:
    response = client.get("/nope")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found", "message": "Not Found"}}


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/health"), ("DELETE", "/datasets/1")],
    ids=["post-health", "delete-dataset"],
)
def test_wrong_method_returns_405(client: TestClient, method: str, path: str) -> None:
    response = client.request(method, path)

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"
    assert "GET" in response.headers["allow"]


# --- Validation errors -----------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/scans/0",
        "/scans/abc",
        "/scans/2147483648",
        "/datasets/0",
        "/datasets/-1",
        "/datasets/abc/columns",
        "/findings/0",
        "/findings/9999999999",
    ],
)
def test_invalid_ids_are_rejected(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert all(location.startswith("path.") for location in issue_locations(response))


def test_review_path_id_is_validated(client: TestClient) -> None:
    response = client.post("/findings/0/review", json={"decision": "approved"})

    assert response.status_code == 422
    assert any(loc.startswith("path.") for loc in issue_locations(response))


@pytest.mark.parametrize(
    "path",
    [
        "/datasets?limit=0",
        "/datasets?limit=201",
        "/datasets?offset=-1",
        "/datasets?limit=abc",
        "/findings?limit=0",
        "/findings?offset=1000001",
    ],
)
def test_pagination_parameters_are_validated(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert all(location.startswith("query.") for location in issue_locations(response))


@pytest.mark.parametrize(
    ("query", "field"),
    [
        ("severity=critical", "severity"),
        ("review_status=done", "review_status"),
        ("category=bogus", "category"),
        ("dataset_id=0", "dataset_id"),
        ("scan_id=abc", "scan_id"),
    ],
)
def test_filter_parameters_are_validated(client: TestClient, query: str, field: str) -> None:
    response = client.get(f"/findings?{query}")

    assert response.status_code == 422
    assert any(loc.startswith(f"query.{field}") for loc in issue_locations(response))


def test_missing_file_field(client: TestClient) -> None:
    response = client.post("/scans", data={"dataset_name": "x"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "body.file" in issue_locations(response)


def test_text_field_instead_of_file(client: TestClient) -> None:
    response = client.post("/scans", data={"file": "a,b\n1,SECRET-CELL\n"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "SECRET" not in response.text


def test_invalid_json_body(client: TestClient) -> None:
    response = client.post(
        "/findings/1/review",
        content=b"{not json SECRET-BODY",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "SECRET" not in response.text


def test_validation_errors_do_not_echo_submitted_values(client: TestClient) -> None:
    response = client.post("/findings/1/review", json={"decision": "SECRET-DECISION"})

    assert response.status_code == 422
    assert "SECRET" not in response.text
    assert "body.decision" in issue_locations(response)


# --- Unexpected errors -----------------------------------------------------------


def test_unhandled_error_returns_generic_body_and_safe_log(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise ValueError("SECRET-VALUE inside the exception")

    monkeypatch.setattr(query_service, "get_metrics_summary", explode)
    # Tracebacks are only logged when DEBUG is on; force it off so the log check is stable.
    monkeypatch.setattr("app.api.errors.get_settings", lambda: SimpleNamespace(debug=False))
    caplog.set_level(logging.DEBUG, logger="app.api.errors")

    response = client.get("/metrics/summary")

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "An unexpected error occurred."}
    }
    assert "SECRET" not in response.text
    assert "Unhandled error" in caplog.text
    assert "error_type=ValueError" in caplog.text
    assert "SECRET" not in caplog.text


# --- Upload-size middleware ------------------------------------------------------


def test_middleware_rejects_oversized_declared_upload(
    client: TestClient, upload: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.api.middleware.get_settings", lambda: SimpleNamespace(max_upload_bytes=1000)
    )
    big = b"a\n" + b"x\n" * 35_000  # about 70 KB: beyond the 1,000-byte limit plus the 64 KiB margin

    response = upload(big)

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "file_too_large"
    assert error["message"] == "The upload exceeds the limit of 1000 bytes."
    assert client.get("/datasets").json()["total"] == 0


def test_middleware_allows_uploads_within_margin(
    upload: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Between the limit and limit plus 64 KiB the middleware lets the request through;
    the route and ingestion limits (5 MiB here) make the real decision."""
    monkeypatch.setattr(
        "app.api.middleware.get_settings", lambda: SimpleNamespace(max_upload_bytes=1000)
    )
    medium = b"a\n" + b"1\n" * 2500  # about 5 KB

    assert upload(medium).status_code == 201


# --- Contract --------------------------------------------------------------------


def test_all_errors_share_one_shape(client: TestClient, upload: Callable[..., Any]) -> None:
    responses = [
        client.get("/nope"),
        client.get("/datasets/999"),
        client.post("/health"),
        client.get("/datasets?limit=0"),
        upload(b"a,a\n1,2\n"),
    ]

    assert [r.status_code for r in responses] == [404, 404, 405, 422, 422]
    for response in responses:
        body = response.json()
        assert set(body) == {"error"}
        assert set(body["error"]) <= {"code", "message", "issues"}
        assert isinstance(body["error"]["code"], str)
        assert isinstance(body["error"]["message"], str)
        assert response.headers["content-type"].startswith("application/json")


def test_openapi_lists_every_endpoint(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert set(paths) == EXPECTED_PATHS
    assert "post" in paths["/scans"]
    assert "post" in paths["/findings/{finding_id}/review"]
    assert "404" in paths["/datasets/{dataset_id}"]["get"]["responses"]
    assert "413" in paths["/scans"]["post"]["responses"]


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_docs_pages_are_enabled(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")