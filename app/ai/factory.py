"""Selects the classification provider from settings. The mock provider is the default.

A misconfigured real provider never silently turns into the mock: that would present rule-based
placeholders as if the real model had answered. Selecting `gemini` without a key or model gives
a provider whose calls fail safely (the classifier records provider_unavailable), and a
malformed model ID is rejected immediately with ValueError.

With no argument these functions follow the process settings (`.env` and environment variables).
So if `.env` says AI_PROVIDER=gemini, build_classifier() will call the real API. Pass explicit
Settings where that must not happen.
"""

from typing import Any

from app.ai.classifier import ColumnClassifier
from app.ai.gemini_provider import GeminiClassificationProvider
from app.ai.mock_provider import MockClassificationProvider
from app.ai.provider import ClassificationProvider
from app.core.config import Settings, get_settings


def get_classification_provider(settings: Settings | None = None) -> ClassificationProvider:
    active = settings if settings is not None else get_settings()
    if active.ai_provider == "gemini":
        return GeminiClassificationProvider.from_settings(active)
    return MockClassificationProvider()


def build_classifier(settings: Settings | None = None) -> ColumnClassifier:
    return ColumnClassifier(get_classification_provider(settings))


def describe_provider(settings: Settings | None = None) -> dict[str, Any]:
    """A secret-free summary of the active configuration, safe to log or show in a UI."""
    active = settings if settings is not None else get_settings()
    if active.ai_provider != "gemini":
        return {"provider": "mock", "is_mock": True}
    secret = active.gemini_api_key
    return {
        "provider": "gemini",
        "is_mock": False,
        "model": active.gemini_model.strip() or None,
        "key_configured": bool(secret is not None and secret.get_secret_value().strip()),
        "timeout_seconds": active.gemini_timeout_seconds,
        "max_attempts": active.gemini_max_attempts,
    }