import ast
import asyncio
import json
import logging
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.api import access, middleware
from app.api.middleware import MULTIPART_OVERHEAD_BYTES, UploadSizeLimitMiddleware
from app.api.routes import scans as scans_route
from app.core.config import get_settings

ROOT = Path(__file__).resolve().parent.parent
TEST_MAX_BYTES = 1000
LIMIT = TEST_MAX_BYTES + MULTIPART_OVERHEAD_BYTES


# --- A tiny ASGI harness (no framework behaviour involved) -----------------------------------------


def run_asgi(
    app: Any,
    *,
    method: str = "POST",
    path: str = "/scans",
    headers: Sequence[tuple[bytes, bytes]] = (),
    chunks: Sequence[bytes] = (),
    scope_type: str = "http",
) -> tuple[list[dict[str, Any]], list[int]]:
    """Call an ASGI app once. Returns (messages sent, sizes of the body chunks the client side delivered)."""
    queue = list(chunks)
    delivered: list[int] = []
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if queue:
            body = queue.pop(0)
            delivered.append(len(body))
            return {"type": "http.request", "body": body, "more_body": bool(queue)}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": scope_type, "method": method, "path": path, "headers": list(headers)}
    asyncio.run(app(scope, receive, send))
    return sent, delivered


def statuses(sent: list[dict[str, Any]]) -> list[int]:
    return [message["status"] for message in sent if message["type"] == "http.response.start"]


def body_json(sent: list[dict[str, Any]]) -> Any:
    return json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))


async def answering_app(scope: Any, receive: Any, send: Any) -> None:
    """Reads the whole body, then answers 200."""
    while True:
        message = await receive()
        if message["type"] != "http.request" or not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def failing_on_disconnect_app(scope: Any, receive: Any, send: Any) -> None:
    """Like FastAPI: when the body cannot be read, answers 400."""
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": b"bad"})
            return
        if not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def exploding_on_disconnect_app(scope: Any, receive: Any, send: Any) -> None:
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise RuntimeError("boom")


async def exploding_app(scope: Any, receive: Any, send: Any) -> None:
    raise RuntimeError("boom")


@pytest.fixture()
def small_limit(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(middleware, "get_settings", lambda: SimpleNamespace(max_upload_bytes=TEST_MAX_BYTES))
    return LIMIT


# =============================================================================================
# 1. The upload size limit counts the bytes that actually arrive
# =============================================================================================


def test_a_body_within_the_limit_passes_through(small_limit: int) -> None:
    sent, delivered = run_asgi(UploadSizeLimitMiddleware(answering_app), chunks=[b"x" * 300] * 3)

    assert statuses(sent) == [200]
    assert delivered == [300, 300, 300]


def test_a_body_over_the_limit_without_content_length_is_stopped(small_limit: int) -> None:
    chunks = [b"x" * 40_000] * 10  # 400,000 bytes and no Content-Length header

    sent, delivered = run_asgi(UploadSizeLimitMiddleware(answering_app), chunks=chunks)

    assert statuses(sent) == [413]  # the application's own 200 was discarded
    assert body_json(sent)["error"]["code"] == "file_too_large"
    assert "1000 bytes" in body_json(sent)["error"]["message"]
    assert delivered == [40_000, 40_000]  # the rest of the body was never read


def test_a_declared_length_over_the_limit_is_stopped_before_any_read(small_limit: int) -> None:
    headers = [(b"content-length", str(small_limit + 1).encode())]

    sent, delivered = run_asgi(UploadSizeLimitMiddleware(answering_app), headers=headers, chunks=[b"x"])

    assert statuses(sent) == [413]
    assert delivered == []


@pytest.mark.parametrize(("method", "path"), [("POST", "/other"), ("GET", "/scans")], ids=["other-path", "get-method"])
def test_other_requests_are_not_counted(small_limit: int, method: str, path: str) -> None:
    sent, delivered = run_asgi(
        UploadSizeLimitMiddleware(answering_app), method=method, path=path, chunks=[b"x" * 40_000] * 10
    )

    assert statuses(sent) == [200]
    assert len(delivered) == 10


def test_a_downstream_error_after_the_limit_is_replaced_by_413(small_limit: int) -> None:
    sent, _ = run_asgi(UploadSizeLimitMiddleware(failing_on_disconnect_app), chunks=[b"x" * 40_000] * 4)

    assert statuses(sent) == [413]  # not the 400 the application tried to send


def test_an_exception_after_the_limit_is_replaced_by_413(small_limit: int) -> None:
    sent, _ = run_asgi(UploadSizeLimitMiddleware(exploding_on_disconnect_app), chunks=[b"x" * 40_000] * 4)

    assert statuses(sent) == [413]


def test_an_exception_within_the_limit_still_propagates(small_limit: int) -> None:
    with pytest.raises(RuntimeError):
        run_asgi(UploadSizeLimitMiddleware(exploding_app), chunks=[b"x" * 10])


def test_non_http_scopes_pass_through(small_limit: int) -> None:
    seen: list[str] = []

    async def recording_app(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    run_asgi(UploadSizeLimitMiddleware(recording_app), scope_type="lifespan")

    assert seen == ["lifespan"]


def test_the_real_app_stops_a_body_without_content_length(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(middleware, "get_settings", lambda: SimpleNamespace(max_upload_bytes=100))
    monkeypatch.setattr(middleware, "_declared_content_length", lambda scope: None)  # skip the header check
    boundary = "xyz"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="data.csv"\r\n'
            "Content-Type: text/csv\r\n\r\n"
        ).encode()
        + b"a,b\n" * 20_000
        + f"\r\n--{boundary}--\r\n".encode()
    )
    assert len(body) > 100 + MULTIPART_OVERHEAD_BYTES  # a valid body that is over the limit

    response = client.post(
        "/scans",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"
    assert "100 bytes" in response.json()["error"]["message"]


# =============================================================================================
# 2. The effective access mode
# =============================================================================================


@pytest.fixture()
def configure(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str, str], str]]:
    def apply(mode: str, environment: str) -> str:
        monkeypatch.setenv("VERITYGRID_MODE", mode)
        monkeypatch.setenv("ENVIRONMENT", environment)
        get_settings.cache_clear()
        access.load_mode.cache_clear()
        return access.load_mode()

    yield apply
    get_settings.cache_clear()
    access.load_mode.cache_clear()


@pytest.mark.parametrize(
    ("mode", "environment", "expected"),
    [
        pytest.param("development", "development", "development", id="local-development"),
        pytest.param("development", "production", "demo", id="development-in-production-is-refused"),
        pytest.param("development", "PRODUCTION", "demo", id="case-is-ignored"),
        pytest.param("development", " prod ", "demo", id="short-name-and-spaces"),
        pytest.param("demo", "production", "demo", id="demo-in-production"),
        pytest.param("demo", "development", "demo", id="demo-locally"),
    ],
)
def test_the_effective_mode(configure: Callable[[str, str], str], mode: str, environment: str, expected: str) -> None:
    assert configure(mode, environment) == expected


def test_development_mode_in_production_is_logged_and_blocks_writes(
    client: TestClient,
    configure: Callable[[str, str], str],
    mixed_csv: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="app.api.access")

    assert configure("development", "production") == "demo"
    response = client.post("/scans", files={"file": ("data.csv", mixed_csv, "text/csv")})

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "read_only_mode"
    assert "refused" in caplog.text


def test_announce_mode_warns_loudly_in_development(
    configure: Callable[[str, str], str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="app.api.access")
    configure("development", "development")

    assert access.announce_mode() == "development"

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("OPEN" in message for message in warnings)


def test_announce_mode_says_read_only_in_demo(
    configure: Callable[[str, str], str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="app.api.access")
    configure("demo", "development")

    assert access.announce_mode() == "demo"

    assert any("read-only" in record.getMessage() for record in caplog.records)


def test_the_lifespan_announces_the_mode_before_creating_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(main_module, "announce_mode", lambda: calls.append("announce"))
    monkeypatch.setattr(main_module, "init_db", lambda: calls.append("init"))

    async def enter() -> None:
        async with main_module.lifespan(main_module.app):
            calls.append("running")

    asyncio.run(enter())

    assert calls == ["announce", "init", "running"]


# =============================================================================================
# 3. Review states
# =============================================================================================


def first_finding_id(client: TestClient, upload: Callable[..., Any], mixed_csv: bytes) -> int:
    assert upload(mixed_csv, "mixed.csv").status_code == 201
    return int(client.get("/findings").json()["items"][0]["id"])


def test_review_status_always_equals_the_latest_decision(
    client: TestClient, upload: Callable[..., Any], mixed_csv: bytes
) -> None:
    finding_id = first_finding_id(client, upload, mixed_csv)
    sequence = ["approved", "rejected", "approved", "rejected"]

    for position, decision in enumerate(sequence, start=1):
        assert client.post(f"/findings/{finding_id}/review", json={"decision": decision}).status_code == 201

        finding = client.get(f"/findings/{finding_id}").json()
        assert finding["review_status"] == decision
        assert sorted(item["decision"] for item in finding["review_decisions"]) == sorted(sequence[:position])


@pytest.mark.parametrize(
    "value",
    ["pending", "approve", "", None, 1],
    ids=["pending", "wrong-word", "empty", "null", "number"],
)
def test_invalid_decisions_are_rejected_and_change_nothing(
    client: TestClient, upload: Callable[..., Any], mixed_csv: bytes, value: Any
) -> None:
    finding_id = first_finding_id(client, upload, mixed_csv)

    response = client.post(f"/findings/{finding_id}/review", json={"decision": value})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    finding = client.get(f"/findings/{finding_id}").json()
    assert finding["review_status"] == "pending" and finding["review_decisions"] == []
    assert client.get("/metrics/summary").json()["review_decision_count"] == 0


# =============================================================================================
# 4. No model call can be caused over HTTP
# =============================================================================================


def test_the_upload_route_never_passes_a_classifier(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, mixed_csv: bytes
) -> None:
    captured: dict[str, Any] = {}
    original = scans_route.run_scan

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(scans_route, "run_scan", spy)

    response = client.post("/scans", files={"file": ("data.csv", mixed_csv, "text/csv")})

    assert response.status_code == 201
    assert "classifier" not in captured  # so run_scan classifies nothing and no provider is called


def test_the_api_layer_imports_no_provider_code() -> None:
    files = sorted((ROOT / "app" / "api").rglob("*.py"))
    assert files

    for path in files:
        modules: list[str] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        for module in modules:
            assert not module.startswith(("app.ai", "app.evaluation")), f"{path.name} imports {module}"
            assert module.split(".")[0] not in {"google", "langgraph"}, f"{path.name} imports {module}"