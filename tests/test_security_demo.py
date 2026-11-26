import ast
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import access
from app.db.models import Base
from app.db.session import build_engine
from app.services.ingestion import ingest_csv
from scripts import seed_demo
from ui.api_client import ApiError, ApiErrorKind

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample"


@pytest.fixture()
def set_mode() -> Iterator[Callable[[str], None]]:
    def apply(mode: str) -> None:
        pytest.MonkeyPatch().setenv("VERITYGRID_MODE", mode)
        access.load_mode.cache_clear()

    yield apply
    access.load_mode.cache_clear()


def demo_url(tmp_path: Path, name: str = "seed_demo.db") -> str:
    return f"sqlite:///{tmp_path / name}"


def dump_database(url: str) -> str:
    engine = build_engine(url)
    try:
        with Session(engine) as db:
            parts = []
            for table in Base.metadata.sorted_tables:
                parts.extend(repr(tuple(row)) for row in db.execute(select(table)).all())
        return "\n".join(parts)
    finally:
        engine.dispose()


# --- The seed script ------------------------------------------------------------------------------


def test_seed_creates_the_demo_data(tmp_path: Path) -> None:
    counts = seed_demo.seed(demo_url(tmp_path), sample_dir=SAMPLE_DIR)

    assert counts["datasets"] == 3
    assert counts["findings"] >= 5
    assert (counts["approved"], counts["rejected"]) == (1, 1)
    assert counts["pending"] == counts["findings"] - 2


def test_seed_refuses_a_database_that_already_has_data(tmp_path: Path) -> None:
    url = demo_url(tmp_path)
    seed_demo.seed(url, sample_dir=SAMPLE_DIR)

    with pytest.raises(seed_demo.SeedError, match="already contains datasets"):
        seed_demo.seed(url, sample_dir=SAMPLE_DIR)


def test_reset_recreates_a_demo_database(tmp_path: Path) -> None:
    url = demo_url(tmp_path)
    seed_demo.seed(url, sample_dir=SAMPLE_DIR)

    counts = seed_demo.seed(url, reset=True, sample_dir=SAMPLE_DIR)

    assert counts["datasets"] == 3  # recreated, not doubled


def test_reset_is_refused_for_a_database_not_named_demo(tmp_path: Path) -> None:
    url = demo_url(tmp_path, "development.db")
    seed_demo.seed(url, sample_dir=SAMPLE_DIR)

    with pytest.raises(seed_demo.SeedError, match="contains 'demo'"):
        seed_demo.seed(url, reset=True, sample_dir=SAMPLE_DIR)

    assert (tmp_path / "development.db").exists()  # nothing was deleted


@pytest.mark.parametrize(
    "url",
    ["sqlite:///:memory:", "postgresql://user:pw@localhost/demo"],
    ids=["in-memory", "not-sqlite"],
)
def test_reset_is_refused_when_there_is_no_file_to_delete(url: str) -> None:
    with pytest.raises(seed_demo.SeedError, match="SQLite database file"):
        seed_demo.prepare_target(url, reset=True)


def test_the_seeded_database_holds_no_cell_values(tmp_path: Path) -> None:
    url = demo_url(tmp_path)
    seed_demo.seed(url, sample_dir=SAMPLE_DIR)
    emails: list[str] = []
    for filename in ("customers.csv", "inventory.csv"):
        frame = ingest_csv((SAMPLE_DIR / filename).read_bytes()).dataframe
        for column in frame.columns:
            emails += [str(value) for value in frame[column].dropna() if "@" in str(value)]

    dump = dump_database(url)

    assert emails, "the sample files should contain email addresses"
    assert not [email for email in emails if email in dump]


def test_the_seed_never_calls_an_ai_provider(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    original = seed_demo.run_scan
    calls: list[dict[str, Any]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(seed_demo, "run_scan", spy)

    seed_demo.seed_into(session, SAMPLE_DIR)

    assert len(calls) == 3
    assert all(kwargs.get("classifier") is None for kwargs in calls)


def test_the_seed_script_imports_no_network_or_provider_code() -> None:
    tree = ast.parse((ROOT / "scripts" / "seed_demo.py").read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)

    forbidden_top = {"requests", "httpx", "httpx2", "urllib", "urllib3", "socket", "aiohttp", "google"}
    for module in modules:
        assert module.split(".")[0] not in forbidden_top, module
        assert not module.startswith(("app.ai", "app.evaluation", "app.workflows")), module


# --- The seeded data, served read-only ----------------------------------------------------------------------


def test_seeded_data_is_served_read_only(
    client: TestClient, session: Session, set_mode: Callable[[str], None]
) -> None:
    result = seed_demo.seed_into(session, SAMPLE_DIR)
    set_mode("demo")

    assert result["datasets"] == 3
    assert client.get("/datasets").json()["total"] == 3
    assert client.get("/metrics/summary").json()["review_decision_count"] == 2
    finding_id = client.get("/findings", params={"review_status": "pending"}).json()["items"][0]["id"]

    upload = client.post("/scans", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    review = client.post(f"/findings/{finding_id}/review", json={"decision": "approved"})

    for response in (upload, review):
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "read_only_mode"
    assert client.get("/metrics/summary").json()["review_decision_count"] == 2  # nothing was written


# --- The dashboard's handling of the refusal --------------------------------------------------------------------


def test_the_read_only_rejection_gets_a_friendly_message(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from ui import common

    calls: list[tuple[str, tuple[Any, ...]]] = []

    class Mini:
        def __getattr__(self, name: str) -> Callable[..., None]:
            def call(*args: Any, **kwargs: Any) -> None:
                calls.append((name, args))

            return call

    monkeypatch.setattr(common, "st", Mini())

    common.show_api_error(
        ApiError(ApiErrorKind.HTTP_ERROR, access.READ_ONLY_MESSAGE, status=403, code="read_only_mode")
    )

    names = [name for name, _args in calls]
    text = " ".join(str(arg) for _name, args in calls for arg in args)
    assert "info" in names and "error" not in names
    assert "read-only" in text


def test_the_read_only_code_matches_between_the_api_and_the_dashboard() -> None:
    pytest.importorskip("streamlit")
    from ui import common

    assert common.READ_ONLY_CODE == access.READ_ONLY_CODE