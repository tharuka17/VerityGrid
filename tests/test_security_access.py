import logging
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import access
from app.api.access import ReadOnlyModeMiddleware, read_mode
from app.main import app


@pytest.fixture()
def set_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], None]]:
    def apply(mode: str) -> None:
        monkeypatch.setenv("VERITYGRID_MODE", mode)
        access.load_mode.cache_clear()

    yield apply
    access.load_mode.cache_clear()


def csv_file(content: bytes = b"a,b\n1,2\n") -> dict[str, Any]:
    return {"file": ("data.csv", content, "text/csv")}


def assert_read_only(response: Any) -> None:
    assert response.status_code == 403
    assert response.json() == {"error": {"code": "read_only_mode", "message": access.READ_ONLY_MESSAGE}}


# --- Mode resolution -----------------------------------------------------------------------


def test_the_default_mode_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERITYGRID_MODE", raising=False)

    assert read_mode(None) == "demo"  # no environment variable and no .env file


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("development", "development", id="development"),
        pytest.param("demo", "demo", id="demo"),
        pytest.param("Development", "development", id="mixed-case"),
        pytest.param("  DEMO ", "demo", id="spaces"),
    ],
)
def test_explicit_modes(monkeypatch: pytest.MonkeyPatch, value: str, expected: str) -> None:
    monkeypatch.setenv("VERITYGRID_MODE", value)

    assert read_mode(None) == expected


def test_an_unknown_mode_fails_closed_without_logging_the_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="app.api.access")
    monkeypatch.setenv("VERITYGRID_MODE", "SECRET-MODE-VALUE")

    assert read_mode(None) == "demo"
    assert "error_type=" in caplog.text
    assert "SECRET-MODE-VALUE" not in caplog.text


def test_the_suite_itself_runs_in_development_mode() -> None:
    """If this fails, the root conftest.py is not being collected and many tests will fail with 403."""
    assert access.load_mode() == "development"


# --- Read-only mode ------------------------------------------------------------------------


def test_demo_mode_blocks_uploads_and_reviews(
    client: TestClient, upload: Callable[..., Any], mixed_csv: bytes, set_mode: Callable[[str], None]
) -> None:
    assert upload(mixed_csv, "mixed.csv").status_code == 201
    finding_id = client.get("/findings").json()["items"][0]["id"]
    set_mode("demo")

    assert_read_only(client.post("/scans", files=csv_file()))
    assert_read_only(client.post(f"/findings/{finding_id}/review", json={"decision": "approved"}))

    assert client.get("/datasets").json()["total"] == 1  # nothing was written
    finding = client.get(f"/findings/{finding_id}").json()
    assert finding["review_status"] == "pending" and finding["review_decisions"] == []


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_demo_mode_blocks_every_unsafe_method(
    client: TestClient, set_mode: Callable[[str], None], method: str
) -> None:
    set_mode("demo")

    for path in ("/scans", "/findings/1/review", "/datasets/1", "/nope"):
        assert_read_only(client.request(method, path))  # even routes that do not exist


def test_demo_mode_still_serves_reads(
    client: TestClient, upload: Callable[..., Any], mixed_csv: bytes, set_mode: Callable[[str], None]
) -> None:
    assert upload(mixed_csv, "mixed.csv").status_code == 201
    set_mode("demo")

    for path in ("/health", "/datasets", "/findings", "/metrics/summary", "/evaluation/report", "/openapi.json"):
        assert client.get(path).status_code == 200, path
    assert client.head("/health").status_code != 403


def test_the_access_check_runs_before_the_upload_checks(
    client: TestClient, set_limits: Callable[..., None], set_mode: Callable[[str], None]
) -> None:
    set_limits(max_bytes=3)
    set_mode("demo")

    assert_read_only(client.post("/scans", files=csv_file(b"a,b\n1,2\n3,4\n")))  # 403, not 413


def test_rejected_writes_are_logged_without_the_path(
    client: TestClient, set_mode: Callable[[str], None], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="app.api.access")
    set_mode("demo")

    client.post("/scans/SECRET-PATH-TEXT")

    assert "method=POST" in caplog.text
    assert "SECRET-PATH-TEXT" not in caplog.text


def test_the_middleware_is_installed_and_outermost() -> None:
    classes = [middleware.cls for middleware in app.user_middleware]

    assert ReadOnlyModeMiddleware in classes
    assert classes[0] is ReadOnlyModeMiddleware, "it must be added last so that it runs first"


def test_development_mode_allows_writes(
    client: TestClient, mixed_csv: bytes, set_mode: Callable[[str], None]
) -> None:
    set_mode("development")

    assert client.post("/scans", files=csv_file(mixed_csv)).status_code == 201


# --- Hostile text is data, never code ---------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["x'; DROP TABLE datasets; --", "<script>alert(1)</script>", "../../etc/passwd"],
    ids=["sql", "html", "path"],
)
def test_hostile_dataset_names_are_stored_literally(client: TestClient, mixed_csv: bytes, name: str) -> None:
    response = client.post("/scans", files=csv_file(mixed_csv), data={"dataset_name": name})

    assert response.status_code == 201
    datasets = client.get("/datasets").json()
    assert datasets["total"] == 1 and datasets["items"][0]["name"] == name
    assert client.get("/findings").json()["total"] == 3  # the tables are intact
    assert client.get("/metrics/summary").status_code == 200


@pytest.mark.parametrize(
    ("comment", "reviewer"),
    [
        ("'); DROP TABLE findings; --", "x' OR '1'='1"),
        ("<img src=x onerror=alert(1)>", "<script>alert(1)</script>"),
    ],
    ids=["sql", "html"],
)
def test_hostile_review_text_is_stored_literally(
    client: TestClient, upload: Callable[..., Any], mixed_csv: bytes, comment: str, reviewer: str
) -> None:
    assert upload(mixed_csv, "mixed.csv").status_code == 201
    finding_id = client.get("/findings").json()["items"][0]["id"]

    response = client.post(
        f"/findings/{finding_id}/review",
        json={"decision": "approved", "comment": comment, "reviewer": reviewer},
    )

    assert response.status_code == 201
    history = client.get(f"/findings/{finding_id}").json()["review_decisions"]
    assert [(item["comment"], item["reviewer"]) for item in history] == [(comment, reviewer)]
    assert client.get("/findings").json()["total"] == 3


@pytest.mark.parametrize(
    "params",
    [
        {"severity": "high' OR '1'='1"},
        {"review_status": "pending;DROP TABLE findings"},
        {"category": "x' --"},
        {"detection_method": "regex_email_pattern' OR 1=1"},
        {"dataset_id": "1 OR 1=1"},
        {"scan_id": "1;--"},
        {"limit": "1;DROP"},
        {"offset": "-1"},
    ],
    ids=["severity", "review-status", "category", "method", "dataset-id", "scan-id", "limit", "offset"],
)
def test_hostile_query_values_are_rejected_not_executed(client: TestClient, params: dict[str, str]) -> None:
    response = client.get("/findings", params=params)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    for value in params.values():
        if len(value) > 6:
            assert value not in response.text  # the submitted value is never echoed


@pytest.mark.parametrize(
    "path",
    ["/findings/1;DROP TABLE findings", "/datasets/abc", "/scans/0", "/findings/99999999999"],
    ids=["injection", "letters", "zero", "too-large"],
)
def test_hostile_path_ids_are_rejected(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 422


# --- Credentials ----------------------------------------------------------------------------------


def test_backend_secrets_never_appear_in_responses(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, mixed_csv: bytes
) -> None:
    sentinel = "SECRET-KEY-SENTINEL"
    monkeypatch.setenv("GEMINI_API_KEY", sentinel)
    monkeypatch.setenv("GOOGLE_API_KEY", sentinel)

    responses = [
        client.get("/health"),
        client.get("/openapi.json"),
        client.get("/metrics/summary"),
        client.get("/nope"),
        client.post("/health"),
        client.post("/scans", files=csv_file(mixed_csv)),
        client.post("/scans", files=csv_file(b"")),
    ]

    for response in responses:
        assert sentinel not in response.text
        assert sentinel not in str(dict(response.headers))