from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from app.ai.classifier import ColumnClassifier, ResultStatus
from app.ai.factory import build_classifier, describe_provider, get_classification_provider
from app.ai.gemini_provider import GeminiClassificationProvider
from app.ai.mock_provider import MockClassificationProvider
from app.ai.provider import ClassificationLabel, ClassificationSource
from app.core.config import Settings
from app.services.detectors import run_detectors
from app.services.profiling import profile_dataframe

SECRET_KEY = "SECRET-KEY-VALUE"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own keys and settings out of these tests."""
    for name in (
        "AI_PROVIDER",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_TIMEOUT_SECONDS",
        "GEMINI_MAX_ATTEMPTS",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


# --- Settings ----------------------------------------------------------------


def test_defaults_select_the_mock_provider() -> None:
    settings = Settings(_env_file=None)

    assert settings.ai_provider == "mock"
    assert settings.gemini_api_key is None
    assert settings.gemini_model == ""
    assert settings.gemini_timeout_seconds == 20.0
    assert settings.gemini_max_attempts == 2
    provider = get_classification_provider(settings)
    assert isinstance(provider, MockClassificationProvider)
    assert provider.source is ClassificationSource.MOCK


def test_unknown_provider_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_provider="openai")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gemini_timeout_seconds": 0},
        {"gemini_timeout_seconds": 121},
        {"gemini_max_attempts": 0},
        {"gemini_max_attempts": 4},
    ],
    ids=["timeout-zero", "timeout-too-large", "attempts-zero", "attempts-too-many"],
)
def test_gemini_settings_are_range_checked(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


def test_values_are_read_from_an_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AI_PROVIDER=gemini\n"
        f"GEMINI_API_KEY={SECRET_KEY}\n"
        "GEMINI_MODEL=test-model\n"
        "GEMINI_TIMEOUT_SECONDS=30\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=str(env_file))

    assert settings.ai_provider == "gemini"
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == SECRET_KEY
    assert settings.gemini_model == "test-model"
    assert settings.gemini_timeout_seconds == 30.0


def test_the_key_is_hidden_in_settings_output() -> None:
    settings = Settings(
        _env_file=None, ai_provider="gemini", gemini_api_key=SECRET_KEY, gemini_model="test-model"
    )

    assert SECRET_KEY not in repr(settings)
    assert SECRET_KEY not in str(settings)
    assert SECRET_KEY not in str(describe_provider(settings))


# --- Selection ---------------------------------------------------------------


def test_gemini_is_selected_only_when_configured() -> None:
    settings = Settings(
        _env_file=None, ai_provider="gemini", gemini_api_key=SECRET_KEY, gemini_model="test-model"
    )

    provider = get_classification_provider(settings)

    assert isinstance(provider, GeminiClassificationProvider)
    assert provider.name == "gemini:test-model"
    assert provider.source is ClassificationSource.LLM


def test_a_malformed_model_id_is_a_configuration_error() -> None:
    settings = Settings(
        _env_file=None, ai_provider="gemini", gemini_api_key=SECRET_KEY, gemini_model="bad model id"
    )

    with pytest.raises(ValueError):
        get_classification_provider(settings)


def test_gemini_without_a_key_fails_safe_and_never_becomes_the_mock() -> None:
    settings = Settings(_env_file=None, ai_provider="gemini", gemini_model="test-model")
    df = pd.DataFrame({"a": [1, 2]})

    result = build_classifier(settings).classify_column(
        profile_dataframe(df).columns[0], run_detectors(df)
    )

    assert result.status is ResultStatus.PROVIDER_UNAVAILABLE
    assert result.source is ClassificationSource.NONE
    assert result.provider_name == "gemini:test-model"
    assert result.label is ClassificationLabel.UNKNOWN


def test_build_classifier_returns_a_classifier() -> None:
    classifier = build_classifier(Settings(_env_file=None, ai_provider="mock"))

    assert isinstance(classifier, ColumnClassifier)


def test_mock_classifier_end_to_end() -> None:
    df = pd.DataFrame({"contact_email": ["a@example.com", "b@example.com"], "credit_limit": [100, 200]})
    classifier = build_classifier(Settings(_env_file=None, ai_provider="mock"))

    results = classifier.classify_dataset(profile_dataframe(df), run_detectors(df))

    by_name = {r.column_name: r for r in results}
    assert by_name["contact_email"].label is ClassificationLabel.PERSONAL
    assert by_name["credit_limit"].label is ClassificationLabel.FINANCIAL
    for result in results:
        assert result.source is ClassificationSource.MOCK
        assert result.status is ResultStatus.OK
        assert result.explanation.startswith("[MOCK] ")
        assert "not an LLM" in result.display_source


# --- Describing the configuration -------------------------------------------------


def test_describe_mock() -> None:
    assert describe_provider(Settings(_env_file=None)) == {"provider": "mock", "is_mock": True}


@pytest.mark.parametrize(
    ("key", "configured"),
    [(SECRET_KEY, True), ("", False), ("   ", False)],
    ids=["key-present", "key-blank", "key-spaces"],
)
def test_describe_gemini_reports_presence_not_the_key(key: str, configured: bool) -> None:
    settings = Settings(
        _env_file=None, ai_provider="gemini", gemini_api_key=key, gemini_model="test-model"
    )

    description = describe_provider(settings)

    assert description == {
        "provider": "gemini",
        "is_mock": False,
        "model": "test-model",
        "key_configured": configured,
        "timeout_seconds": 20.0,
        "max_attempts": 2,
    }
    assert "SECRET" not in str(description)


def test_describe_gemini_without_a_model() -> None:
    description = describe_provider(Settings(_env_file=None, ai_provider="gemini"))

    assert description["model"] is None
    assert description["key_configured"] is False