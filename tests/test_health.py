from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

client = TestClient(app)
settings = get_settings()


def test_app_is_importable_fastapi_instance() -> None:
    """The application module imports cleanly and exposes a FastAPI app."""
    assert isinstance(app, FastAPI)


def test_health_returns_200() -> None:
    """GET /health responds with HTTP 200 and a JSON body."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_health_response_structure() -> None:
    """The JSON body has exactly the expected keys and values."""
    body = client.get("/health").json()

    assert set(body) == {"status", "app", "environment"}
    assert body["status"] == "ok"
    # Compare against loaded settings so the test passes whatever your .env contains.
    assert body["app"] == settings.app_name
    assert body["environment"] == settings.environment