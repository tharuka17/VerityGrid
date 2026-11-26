import asyncio
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import app.main as main_module
from app.api import access
from app.core.config import PRODUCTION_ENVIRONMENTS, Settings, get_settings
from scripts import seed_demo

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample"


@pytest.fixture()
def set_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], None]]:
    def apply(mode: str) -> None:
        monkeypatch.setenv("VERITYGRID_MODE", mode)
        monkeypatch.setenv("ENVIRONMENT", "development")
        get_settings.cache_clear()
        access.load_mode.cache_clear()

    yield apply
    get_settings.cache_clear()
    access.load_mode.cache_clear()


# --- DEBUG is ignored in production ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("environment", "debug", "expected"),
    [
        pytest.param("production", True, False, id="production"),
        pytest.param("PRODUCTION ", True, False, id="case-and-spaces"),
        pytest.param("prod", True, False, id="short-name"),
        pytest.param("development", True, True, id="development-keeps-debug"),
        pytest.param("staging", True, True, id="other-environment-keeps-debug"),
        pytest.param("production", False, False, id="production-without-debug"),
    ],
)
def test_debug_is_ignored_in_production(environment: str, debug: bool, expected: bool) -> None:
    assert Settings(_env_file=None, environment=environment, debug=debug).debug is expected


def test_ignoring_debug_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="app.core.config")

    Settings(_env_file=None, environment="production", debug=True)

    assert "DEBUG is ignored" in caplog.text


def test_the_production_names_have_one_definition() -> None:
    assert access.PRODUCTION_ENVIRONMENTS is PRODUCTION_ENVIRONMENTS
    assert PRODUCTION_ENVIRONMENTS == {"production", "prod"}


# --- Demo mode needs an existing database ----------------------------------------------------------


def test_development_mode_never_checks_the_file(set_mode: Callable[[str], None], tmp_path: Path) -> None:
    set_mode("development")

    access.check_demo_database(f"sqlite:///{tmp_path / 'missing.db'}")  # no error


def test_demo_mode_refuses_a_missing_file_without_naming_it(set_mode: Callable[[str], None], tmp_path: Path) -> None:
    set_mode("demo")

    with pytest.raises(RuntimeError) as excinfo:
        access.check_demo_database(f"sqlite:///{tmp_path / 'secret-name.db'}")

    message = str(excinfo.value)
    assert "seed_demo" in message
    assert "secret-name" not in message and str(tmp_path) not in message


def test_demo_mode_accepts_an_existing_file(set_mode: Callable[[str], None], tmp_path: Path) -> None:
    set_mode("demo")
    (tmp_path / "demo.db").write_bytes(b"")

    access.check_demo_database(f"sqlite:///{tmp_path / 'demo.db'}")  # no error


@pytest.mark.parametrize(
    "url",
    ["sqlite:///:memory:", "sqlite://", "postgresql://user:pw@localhost/demo"],
    ids=["in-memory", "bare-sqlite", "postgresql"],
)
def test_demo_mode_does_not_check_other_databases(set_mode: Callable[[str], None], url: str) -> None:
    set_mode("demo")

    access.check_demo_database(url)  # no error


def test_a_relative_path_is_resolved_against_the_working_directory(
    set_mode: Callable[[str], None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_mode("demo")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError):
        access.check_demo_database("sqlite:///./demo.db")
    (tmp_path / "demo.db").write_bytes(b"")
    access.check_demo_database("sqlite:///./demo.db")  # now it exists


def test_a_freshly_seeded_demo_database_passes_the_check(set_mode: Callable[[str], None], tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'seeded_demo.db'}"
    seed_demo.seed(url, sample_dir=SAMPLE_DIR)
    set_mode("demo")

    access.check_demo_database(url)  # the Docker image relies on this


# --- The check runs at startup, before any table is created ---------------------------------------------


def run_lifespan(monkeypatch: pytest.MonkeyPatch, database_url: str) -> tuple[list[str], Exception | None]:
    calls: list[str] = []
    monkeypatch.setattr(main_module, "announce_mode", lambda: calls.append("announce"))
    monkeypatch.setattr(main_module, "init_db", lambda: calls.append("init"))
    monkeypatch.setattr(main_module.settings, "database_url", database_url)

    async def enter() -> None:
        async with main_module.lifespan(main_module.app):
            calls.append("running")

    error: Exception | None = None
    try:
        asyncio.run(enter())
    except RuntimeError as exc:
        error = exc
    return calls, error


def test_startup_stops_before_creating_tables_when_the_demo_database_is_missing(
    set_mode: Callable[[str], None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_mode("demo")

    calls, error = run_lifespan(monkeypatch, f"sqlite:///{tmp_path / 'missing_demo.db'}")

    assert calls == ["announce"]  # init_db was never reached, so no empty database was created
    assert error is not None and "seed_demo" in str(error)
    assert not (tmp_path / "missing_demo.db").exists()


def test_startup_continues_when_the_demo_database_exists(
    set_mode: Callable[[str], None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_mode("demo")
    (tmp_path / "demo.db").write_bytes(b"")

    calls, error = run_lifespan(monkeypatch, f"sqlite:///{tmp_path / 'demo.db'}")

    assert error is None
    assert calls == ["announce", "init", "running"]


def test_development_startup_is_unchanged(
    set_mode: Callable[[str], None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_mode("development")

    calls, error = run_lifespan(monkeypatch, f"sqlite:///{tmp_path / 'new.db'}")

    assert error is None
    assert calls == ["announce", "init", "running"]