"""Test suite settings that apply to every test.

The API defaults to read-only (demo) mode, but the existing tests exercise uploads and reviews, so the suite
runs in development mode. Tests of the read-only behavior switch the mode themselves.
"""

from collections.abc import Iterator

import pytest

from app.api.access import load_mode


@pytest.fixture(autouse=True)
def development_mode_for_the_suite(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VERITYGRID_MODE", "development")
    load_mode.cache_clear()
    yield
    load_mode.cache_clear()