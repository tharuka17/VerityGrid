"""Shared fixtures for the HTTP-level API tests.

Isolation: every test gets its own temporary SQLite database. The FastAPI database dependency
is overridden to use it, and TestClient is created WITHOUT a `with` block so the application's
lifespan hook (which calls init_db() on the real database) never runs. Dependency overrides
are cleared after each test.

Older test files define their own `engine` and `session` fixtures; those take precedence over
the ones here, so they keep working unchanged.
"""

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_ingestion_limits
from app.db.session import build_engine, get_db, init_db
from app.main import app
from app.services.ingestion import IngestionLimits

# Scanned with the default detection config this produces exactly three findings:
#   high    missing_values  notes    (4 of 4 empty)
#   medium  missing_values  memo     (1 of 4 empty = 25%)
#   medium  potential_email contact  (1 of 4 values contains an email: 25%)
MIXED_CSV = (
    b"sku,contact,qty,notes,memo\n"
    b"A1,a@example.com,5,,x\n"
    b"A2,call dock,6,,y\n"
    b"A3,leave door,7,,z\n"
    b"A4,ok,8,,\n"
)


@pytest.fixture()
def engine(tmp_path: Path) -> Generator[Engine, None, None]:
    """A fresh SQLite database in a pytest temporary folder, never your local database."""
    test_engine = build_engine(f"sqlite:///{(tmp_path / 'test_api.db').as_posix()}")
    init_db(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture()
def session(engine: Engine) -> Generator[Session, None, None]:
    """A separate session for assertions that read the database directly."""
    with sessionmaker(bind=engine)() as db:
        yield db


@pytest.fixture()
def client(engine: Engine) -> Generator[TestClient, None, None]:
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db() -> Generator[Session, None, None]:
        db = factory()
        try:
            yield db
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    # raise_server_exceptions=False: unexpected errors come back as the 500 response a real
    # client would see, instead of being re-raised inside the test.
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture()
def set_limits(client: TestClient) -> Callable[..., None]:
    """Override the upload limits for one test, for example set_limits(max_bytes=1000)."""

    def apply(**limits: int) -> None:
        app.dependency_overrides[get_ingestion_limits] = lambda: IngestionLimits(**limits)

    return apply


@pytest.fixture()
def upload(client: TestClient) -> Callable[..., Any]:
    """POST a CSV to /scans: upload(content, filename="data.csv", dataset_name="...")."""

    def send(content: bytes, filename: str = "data.csv", **form: str) -> Any:
        return client.post(
            "/scans",
            files={"file": (filename, content, "text/csv")},
            data=form or None,
        )

    return send


@pytest.fixture()
def mixed_csv() -> bytes:
    return MIXED_CSV


@pytest.fixture()
def scanned(upload: Callable[..., Any], mixed_csv: bytes) -> dict[str, Any]:
    """The JSON result of uploading MIXED_CSV as dataset 'mixed_sample' (file 'mixed.csv')."""
    response = upload(mixed_csv, "mixed.csv", dataset_name="mixed_sample")
    assert response.status_code == 201, response.text
    return response.json()