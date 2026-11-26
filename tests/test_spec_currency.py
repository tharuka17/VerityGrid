import re
from pathlib import Path

import pytest

from app.db import models
from app.schemas.findings import FindingCategory

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "PROJECT_SPEC.md"
HEADINGS = [
    "## 1. Purpose and boundaries",
    "## 2. Problem statement",
    "## 3. User journey as built",
    "## 4. Technology stack",
    "## 5. Repository layout",
    "## 6. Data model",
    "## 7. Data flow",
    "## 8. Separation of responsibilities",
    "## 9. Privacy, security and human-review requirements: how they are met",
    "## 10. Testing and evaluation",
    "## 11. Development phases as built",
    "## 12. Key design decisions",
    "## 13. Known limitations and assumptions",
    "## 14. Differences from the Phase 0 plan",
]
PATHISH = re.compile(r"^[A-Za-z0-9_.\-/]+$")
SUFFIXES = (".py", ".md", ".yml", ".yaml", ".csv", ".json", ".txt", ".example", ".toml", ".ini")
GENERATED_PREFIXES = ("reports", "veritygrid", ".venv")
DEPLOYMENT_CLAIMS = ("is deployed", "deployed at", "live demo", "hosted at", "try it live")


def spec() -> str:
    return SPEC.read_text(encoding="utf-8-sig")


@pytest.mark.parametrize("heading", HEADINGS)
def test_every_section_is_present(heading: str) -> None:
    assert heading in spec().splitlines()


def test_the_phase_0_draft_is_gone() -> None:
    text = spec()

    assert "Phase 0 draft. Update only" not in text
    assert not re.search(r"\[P\d+\]", text), "the old planned-phase tags are still in the layout"
    assert "Revised on" in text


def test_every_backticked_path_exists() -> None:
    missing = []
    for token in re.findall(r"`([^`\n]+)`", spec()):
        if not PATHISH.match(token) or token.startswith("/") or token == ".env":
            continue
        if token.startswith(GENERATED_PREFIXES):
            continue
        if "/" in token or token.endswith(SUFFIXES):
            if not (ROOT / token).exists():
                missing.append(token)

    assert sorted(set(missing)) == [], f"paths named in the specification that do not exist: {sorted(set(missing))}"


def test_every_application_package_is_in_the_layout() -> None:
    text = spec()
    packages = [path.name for path in (ROOT / "app").iterdir() if path.is_dir() and not path.name.startswith(("_", "."))]

    assert packages
    for name in packages:
        assert f"app/{name}" in text, f"app/{name} exists but is not described in the specification"


def test_the_top_level_folders_are_in_the_layout() -> None:
    text = spec()

    for name in ("ui/", "ui/views/", "scripts/", "data/sample/", "data/evaluation/", "tests/", "docs/"):
        assert (ROOT / name).is_dir(), name
        assert f"`{name}" in text, name


def test_the_phase_table_covers_the_security_and_packaging_phases() -> None:
    text = spec()

    for row in ("| 10A |", "| 10B |", "| 13 |"):
        assert row in text, row
    assert "**Not started.**" in text


def test_nothing_claims_a_deployment_happened() -> None:
    text = spec().lower()

    for phrase in DEPLOYMENT_CLAIMS:
        assert phrase not in text, phrase
    assert "has not been deployed" in text


def test_the_specification_states_the_access_model() -> None:
    text = spec().lower()

    assert "no authentication" in text
    assert "read-only" in text
    assert "not reachable over http" in text


def test_every_finding_category_is_named() -> None:
    text = spec()

    for category in FindingCategory:
        assert f"`{category.value}`" in text, category.value


def test_every_orm_model_is_named() -> None:
    text = spec()

    for name in ("Dataset", "ColumnProfile", "ScanRun", "Finding", "ReviewDecision", "AuditEvent"):
        assert hasattr(models, name), name
        assert f"**{name}**" in text, name