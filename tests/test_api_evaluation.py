import ast
import json
import warnings
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes.evaluation import get_report_path
from app.evaluation import harness
from app.main import app
from app.services import evaluation_service as service

ROOT = Path(__file__).resolve().parent.parent
UNAVAILABLE_KEYS = {"available", "reason", "file_modified_at", "report"}


@pytest.fixture(scope="module")
def real_report() -> dict[str, Any]:
    """A real report built by the harness (deterministic and mock parts; no provider, no network)."""
    return harness.build_report(repeats=1, warmup=False)


@pytest.fixture()
def report_file(tmp_path: Path, client: TestClient) -> Iterator[Path]:
    """Point the endpoint at a file inside tmp_path for the duration of one test."""
    path = tmp_path / "evaluation_report.json"
    app.dependency_overrides[get_report_path] = lambda: path
    yield path
    app.dependency_overrides.pop(get_report_path, None)


def get(client: TestClient) -> dict[str, Any]:
    response = client.get("/evaluation/report")
    assert response.status_code == 200
    return response.json()


def assert_unavailable(body: dict[str, Any], fragment: str) -> None:
    assert set(body) == UNAVAILABLE_KEYS
    assert body["available"] is False
    assert body["report"] is None and body["file_modified_at"] is None
    assert fragment in body["reason"]


# --- Normal answers ------------------------------------------------------------------------


def test_missing_file_is_a_normal_unavailable_answer(client: TestClient, report_file: Path) -> None:
    body = get(client)

    assert_unavailable(body, "No evaluation report has been generated yet")
    assert "app.evaluation.harness" in body["reason"]


def test_a_real_report_is_served_unchanged(client: TestClient, report_file: Path, real_report: dict[str, Any]) -> None:
    report_file.write_text(json.dumps(real_report, sort_keys=True), encoding="utf-8")

    body = get(client)

    assert body["available"] is True and body["reason"] is None
    assert body["report"] == json.loads(report_file.read_text(encoding="utf-8"))
    assert body["report"]["fingerprints"]["results"] == real_report["fingerprints"]["results"]
    modified = datetime.fromisoformat(body["file_modified_at"])
    assert modified.tzinfo is not None
    assert abs(datetime.now(timezone.utc) - modified) < timedelta(minutes=5)


def test_reading_never_modifies_the_file(client: TestClient, report_file: Path, real_report: dict[str, Any]) -> None:
    report_file.write_text(json.dumps(real_report), encoding="utf-8")
    before = (report_file.read_bytes(), report_file.stat().st_mtime_ns)

    get(client)
    get(client)

    assert (report_file.read_bytes(), report_file.stat().st_mtime_ns) == before


# --- Unusable files ------------------------------------------------------------------------


def test_invalid_json_is_unavailable_and_never_echoed(client: TestClient, report_file: Path) -> None:
    report_file.write_text('SECRET-CONTENT {"not": json', encoding="utf-8")

    response = client.get("/evaluation/report")

    assert response.status_code == 200
    assert_unavailable(response.json(), "valid JSON")
    assert "SECRET" not in response.text


def test_invalid_utf8_is_unavailable(client: TestClient, report_file: Path) -> None:
    report_file.write_bytes(b"\xff\xfe\x00SECRET")

    response = client.get("/evaluation/report")

    assert_unavailable(response.json(), "UTF-8")
    assert "SECRET" not in response.text


@pytest.mark.parametrize(
    "content",
    ["[]", '"text"', "42", "null", "{}"],
    ids=["list", "string", "number", "null", "empty-object"],
)
def test_json_that_is_not_a_report_is_unavailable(client: TestClient, report_file: Path, content: str) -> None:
    report_file.write_text(content, encoding="utf-8")

    assert_unavailable(get(client), "not a supported evaluation report")


@pytest.mark.parametrize("version", [2, 0, "1", True], ids=["two", "zero", "string", "boolean"])
def test_other_versions_are_unavailable(
    client: TestClient, report_file: Path, real_report: dict[str, Any], version: Any
) -> None:
    report_file.write_text(json.dumps({**real_report, "report_version": version}), encoding="utf-8")

    assert_unavailable(get(client), "not a supported evaluation report")


@pytest.mark.parametrize("key", sorted(service.REQUIRED_KEYS))
def test_a_missing_section_is_unavailable(
    client: TestClient, report_file: Path, real_report: dict[str, Any], key: str
) -> None:
    report_file.write_text(json.dumps({k: v for k, v in real_report.items() if k != key}), encoding="utf-8")

    assert_unavailable(get(client), "not a supported evaluation report")


def test_nan_is_unavailable_not_a_server_error(
    client: TestClient, report_file: Path, real_report: dict[str, Any]
) -> None:
    report_file.write_text(json.dumps({**real_report, "run": {"bad": float("nan")}}), encoding="utf-8")

    assert_unavailable(get(client), "valid JSON")  # a 200, not a 500 from the JSON encoder


def test_a_file_over_the_size_limit_is_unavailable(
    client: TestClient, report_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "MAX_REPORT_BYTES", 10)
    report_file.write_text("x" * 100, encoding="utf-8")

    assert_unavailable(get(client), "larger than the API will read")


def test_a_directory_is_not_a_report(tmp_path: Path) -> None:
    result = service.load_report(tmp_path)

    assert result.available is False
    assert "not a regular file" in (result.reason or "")


# --- The route and the contract with the harness -----------------------------------------------


def test_route_is_registered_and_get_only(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    assert set(spec["paths"]["/evaluation/report"]) == {"get"}
    assert "EvaluationReportOut" in spec["components"]["schemas"]


def test_other_verbs_are_rejected(client: TestClient) -> None:
    response = client.post("/evaluation/report")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"
    assert "GET" in response.headers["allow"]


def test_default_path_matches_where_the_harness_writes() -> None:
    assert service.DEFAULT_REPORT_PATH == harness.DEFAULT_OUTPUT_DIR / harness.REPORT_JSON


def test_required_keys_match_a_real_report(real_report: dict[str, Any]) -> None:
    assert set(real_report) == set(service.REQUIRED_KEYS)
    assert real_report["report_version"] == service.SUPPORTED_REPORT_VERSION == harness.REPORT_VERSION


def test_the_api_side_never_runs_an_evaluation_or_calls_a_provider() -> None:
    forbidden_prefixes = ("app.evaluation", "app.ai", "app.workflows")
    forbidden_top = {"google", "requests", "httpx", "httpx2", "urllib", "socket", "langgraph", "langchain", "pandas"}

    for relative in ("app/services/evaluation_service.py", "app/api/routes/evaluation.py"):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        for module in modules:
            assert not module.startswith(forbidden_prefixes), f"{relative} imports {module}"
            assert module.split(".")[0] not in forbidden_top, f"{relative} imports {module}"
            
def test_the_route_is_registered_exactly_once() -> None:
    """Counting app.routes is not reliable (newer FastAPI keeps included routers as nested objects),
    so check the symptom: building the OpenAPI document must not report a duplicate operation."""
    app.openapi_schema = None  # force a fresh build
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema = app.openapi()

    assert [str(w.message) for w in caught if "Duplicate Operation ID" in str(w.message)] == []
    assert set(schema["paths"]["/evaluation/report"]) == {"get"}