import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.api import access
from app.core.config import Settings
from app.evaluation import harness
from app.main import app

ROOT = Path(__file__).resolve().parent.parent

CONFIG_FILES = ["Dockerfile.api", "Dockerfile.ui", "docker-compose.yml", "render.yaml"]
MARKDOWN_FILES = [
    "README.md",
    "SECURITY.md",
    "docs/ARCHITECTURE.md",
    "docs/THREAT_MODEL.md",
    "docs/EVALUATION.md",
    "docs/DEPLOYMENT.md",
]
ALL_FILES = [".env.example", ".dockerignore", *CONFIG_FILES, *MARKDOWN_FILES]
LAYOUT = [
    "app/api",
    "app/ai",
    "app/core",
    "app/db",
    "app/evaluation",
    "app/schemas",
    "app/services",
    "app/workflows",
    "ui",
    "ui/views",
    "scripts",
    "data/sample",
    "tests",
    "docs",
]
SECRET_WORDS = ("KEY", "SECRET", "TOKEN", "PASSWORD")
KEY_NAME_PATTERN = re.compile(r"(?i)gemini|api_key|secret|token|password")
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
]
DEPLOYMENT_CLAIMS = ("is deployed", "deployed at", "live demo", "hosted at", "try it live")
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


# --- Helpers ------------------------------------------------------------------------------------


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8-sig")


def active_lines(relative: str) -> list[str]:
    """Non-blank lines that are not comments."""
    return [
        line.strip()
        for line in read(relative).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def strip_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in active_lines(".env.example"):
        key, separator, value = line.partition("=")
        assert separator, f"not KEY=VALUE: {line}"
        values[key.strip()] = value.strip()
    return values


def normalize(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{id}", path)


def openapi_operations() -> set[tuple[str, str]]:
    schema = app.openapi()
    return {
        (method.upper(), normalize(path))
        for path, operations in schema["paths"].items()
        for method in operations
        if method in HTTP_METHODS
    }


def rate(value: Any) -> str:
    return f"{value:.3f}"


@pytest.fixture(scope="module")
def fresh_report() -> dict[str, Any]:
    """A report built by the harness now (deterministic and mock parts; no provider, no network)."""
    return harness.build_report(repeats=1, warmup=False)


# --- The documented files and layout exist -----------------------------------------------------------


@pytest.mark.parametrize("relative", ALL_FILES)
def test_every_documented_file_exists(relative: str) -> None:
    assert (ROOT / relative).is_file(), f"{relative} is missing"


@pytest.mark.parametrize("relative", LAYOUT)
def test_the_documented_layout_exists(relative: str) -> None:
    assert (ROOT / relative).is_dir(), f"{relative} is missing"


# --- .env.example -----------------------------------------------------------------------------------------


def test_env_example_has_no_secret_values() -> None:
    for key, value in env_example().items():
        if any(word in key.upper() for word in SECRET_WORDS):
            assert value == "", f"{key} must be empty in .env.example"


def test_env_example_defaults_are_the_safe_ones() -> None:
    values = env_example()

    assert values["VERITYGRID_MODE"] == "demo"
    assert values["AI_PROVIDER"] == "mock"
    assert values["GEMINI_API_KEY"] == ""
    assert values["GEMINI_MODEL"] == ""


def test_env_example_covers_every_setting() -> None:
    names = {name.upper() for name in Settings.model_fields}

    assert names <= set(env_example()), sorted(names - set(env_example()))


def test_the_readme_documents_every_example_variable() -> None:
    undocumented_on_purpose = {"APP_NAME"}
    readme = read("README.md")

    for key in env_example():
        if key not in undocumented_on_purpose:
            assert key in readme, f"{key} is not mentioned in the README"


# --- Docker and Render files ------------------------------------------------------------------------------


def test_dockerignore_excludes_secrets_and_local_data() -> None:
    lines = set(active_lines(".dockerignore"))

    assert {".env", ".env.*", "*.db", "reports/", ".git/", ".venv/"} <= lines


@pytest.mark.parametrize("dockerfile", ["Dockerfile.api", "Dockerfile.ui"])
def test_dockerfiles_run_unprivileged_on_the_same_base(dockerfile: str) -> None:
    lines = active_lines(dockerfile)

    assert "FROM python:3.14-slim" in lines
    assert "USER verity" in lines


@pytest.mark.parametrize("dockerfile", ["Dockerfile.api", "Dockerfile.ui"])
def test_dockerfiles_never_copy_the_whole_context_or_env_files(dockerfile: str) -> None:
    for line in active_lines(dockerfile):
        if line.startswith(("COPY ", "ADD ")):
            assert not re.match(r"^(?:COPY|ADD)\s+(?:--\S+\s+)*\.{1,2}/?\s", line), line
            assert ".env" not in line, line


@pytest.mark.parametrize(
    ("dockerfile", "expected"),
    [
        ("Dockerfile.api", {"requirements.txt", "app", "scripts", "data/sample", "data/evaluation"}),
        ("Dockerfile.ui", {"requirements.txt", "ui"}),
    ],
    ids=["api", "ui"],
)
def test_each_image_copies_only_what_it_needs(dockerfile: str, expected: set[str]) -> None:
    sources: set[str] = set()
    for line in active_lines(dockerfile):
        if line.startswith("COPY "):
            parts = [part for part in line.split()[1:] if not part.startswith("--")]
            sources.update(parts[:-1])

    assert sources == expected


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_no_key_names_outside_comments(name: str) -> None:
    offenders = [line for line in active_lines(name) if KEY_NAME_PATTERN.search(line)]

    assert offenders == []


def test_the_api_image_defaults_to_the_read_only_demo() -> None:
    text = "\n".join(active_lines("Dockerfile.api"))

    for fragment in (
        "VERITYGRID_MODE=demo",
        "ENVIRONMENT=production",
        "AI_PROVIDER=mock",
        "scripts.seed_demo",
        "--skip-review",
    ):
        assert fragment in text, fragment
    assert "--llm" not in text
    assert "development" not in text.lower()


def test_compose_binds_ports_to_loopback_only() -> None:
    matches = []
    for line in active_lines("docker-compose.yml"):
        match = re.fullmatch(r"-\s*[\"']?((?:\d{1,3}\.){3}\d{1,3}:)?(\d+):(\d+)[\"']?", line)
        if match:
            matches.append(match)
            assert match.group(1) == "127.0.0.1:", line

    assert len(matches) == 2


def test_compose_structure() -> None:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(read("docker-compose.yml"))

    assert data["name"] == "veritygrid"
    assert set(data["services"]) == {"api", "ui"}
    for name, service in data["services"].items():
        assert service["cap_drop"] == ["ALL"], name
        assert service["security_opt"] == ["no-new-privileges:true"], name
        assert "env_file" not in service, name
        assert all(str(port).startswith("127.0.0.1:") for port in service["ports"]), name
    assert data["services"]["api"]["environment"] == {
        "VERITYGRID_MODE": "demo",
        "ENVIRONMENT": "production",
        "AI_PROVIDER": "mock",
        "DATABASE_URL": "sqlite:///./veritygrid_demo.db",
    }
    assert data["services"]["ui"]["environment"] == {"VERITYGRID_API_URL": "http://api:8000"}
    assert data["services"]["ui"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_render_blueprint_structure() -> None:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(read("render.yaml"))
    services = {service["name"]: service for service in data["services"]}

    assert set(services) == {"veritygrid-api", "veritygrid-ui"}
    for service in services.values():
        assert service["type"] == "web" and service["runtime"] == "docker"
        assert service["plan"] == "free"
        assert service["autoDeploy"] is False
        assert (ROOT / service["dockerfilePath"]).is_file()
        for variable in service["envVars"]:
            assert not any(word in variable["key"].upper() for word in SECRET_WORDS), variable["key"]
    api_env = {item["key"]: item.get("value") for item in services["veritygrid-api"]["envVars"]}
    assert api_env == {
        "VERITYGRID_MODE": "demo",
        "ENVIRONMENT": "production",
        "AI_PROVIDER": "mock",
        "DATABASE_URL": "sqlite:///./veritygrid_demo.db",
    }
    assert services["veritygrid-ui"]["envVars"] == [{"key": "VERITYGRID_API_URL", "sync": False}]


# --- Documents -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MARKDOWN_FILES)
def test_relative_links_resolve(name: str) -> None:
    text = strip_code_blocks(read(name))

    for match in re.finditer(r"\[[^\]]*\]\(([^)\s]+)\)", text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        resolved = (ROOT / name).parent / target.split("#")[0]
        assert resolved.exists(), f"{name} links to {target}, which does not exist"


@pytest.mark.parametrize("name", MARKDOWN_FILES)
def test_code_fences_are_balanced(name: str) -> None:
    fences = [line for line in read(name).splitlines() if line.strip().startswith("```")]

    assert len(fences) % 2 == 0, f"{name} has an unclosed code block"


def test_mermaid_diagrams_have_a_known_type() -> None:
    counts: dict[str, int] = {}
    for name in ("README.md", "docs/ARCHITECTURE.md"):
        blocks = re.findall(r"```mermaid\r?\n(.*?)```", read(name), flags=re.DOTALL)
        counts[name] = len(blocks)
        for block in blocks:
            first = next(line.strip() for line in block.splitlines() if line.strip())
            assert first.startswith(("flowchart", "sequenceDiagram", "graph")), first

    assert counts["README.md"] >= 1
    assert counts["docs/ARCHITECTURE.md"] >= 4


@pytest.mark.parametrize("name", ALL_FILES)
def test_no_secret_patterns_in_any_deliverable(name: str) -> None:
    text = read(name)

    for pattern in SECRET_PATTERNS:
        assert not pattern.search(text), f"{name} matches a secret pattern: {pattern.pattern}"


def test_nothing_claims_a_deployment_happened() -> None:
    for name in [*MARKDOWN_FILES, "render.yaml"]:
        text = read(name).lower()
        for phrase in DEPLOYMENT_CLAIMS:
            assert phrase not in text, f"{name} contains '{phrase}'"

    assert "has not been deployed anywhere" in read("docs/DEPLOYMENT.md").lower()
    assert "not deployed" in read("README.md").lower()


def test_public_writes_are_not_recommended() -> None:
    assert "not recommended" in read("README.md").lower()
    assert "not recommended" in read("docs/DEPLOYMENT.md").lower()
    assert "not authentication" in read("SECURITY.md").lower()
    assert "no authentication" in read("docs/THREAT_MODEL.md").lower()


def test_the_deployment_guide_covers_accounts_and_costs() -> None:
    text = read("docs/DEPLOYMENT.md").lower()

    for word in ("account", "paid", "external", "free"):
        assert word in text, word


SNAPSHOT = ROOT / "data" / "evaluation" / "llm_run_2026-11-19.json"


def load_snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_the_real_model_run_is_described_honestly() -> None:
    for name in ("README.md", "docs/EVALUATION.md"):
        text = read(name).lower()
        assert "single, non-reproducible run" in text, name
        assert "not distinguishable" in text, name
        assert "not yet run" not in text, name  # it has been run, and the documents must say so


def test_the_real_model_numbers_match_the_snapshot() -> None:
    llm = load_snapshot()["llm"]
    coverage = f"{llm['coverage']['answered']} of {llm['coverage']['columns']}"
    both = [f"{llm['metrics']['accuracy']:.3f}", f"{llm['answered_only']['accuracy']:.3f}", coverage]
    macro = [f"{llm['metrics']['macro'][key]:.3f}" for key in ("precision", "recall", "f1")]

    for name in ("README.md", "docs/EVALUATION.md"):
        text = read(name)
        for number in both:
            assert number in text, f"{number} is not in {name}"
    for number in macro:
        assert number in read("docs/EVALUATION.md"), number


def test_the_real_model_class_table_matches_the_snapshot() -> None:
    text = read("docs/EVALUATION.md")

    for label, item in load_snapshot()["llm"]["metrics"]["per_class"].items():
        row = (
            f"| `{label}` | {item['support']} | {item['tp']} | {item['fp']} | {item['fn']} | "
            f"{item['precision']:.3f} | {item['recall']:.3f} |"
        )
        assert row in text, row


def test_the_paired_comparison_matches_the_snapshot() -> None:
    snapshot = load_snapshot()

    def wrong(section: str) -> set[tuple[str, str]]:
        return {(item["case"], item["column"]) for item in snapshot[section]["misclassified"]}

    mock, model = wrong("mock"), wrong("llm")
    text = read("docs/EVALUATION.md")

    assert f"wrong for both: {len(mock & model)}" in text
    assert f"wrong only for the mock: {len(mock - model)}" in text
    assert f"wrong only for the model: {len(model - mock)}" in text


# --- Documents against the code ----------------------------------------------------------------------------------


def test_the_evaluation_document_matches_a_fresh_report(fresh_report: dict[str, Any]) -> None:
    text = read("docs/EVALUATION.md")
    fingerprints = fresh_report["fingerprints"]
    detection = fresh_report["deterministic"]["metrics"]
    mock = fresh_report["mock"]["metrics"]

    assert fingerprints["results"] in text
    assert fingerprints["dataset"][:12] in text
    expected = [
        *(rate(detection["micro"][key]) for key in ("precision", "recall", "f1")),
        *(rate(detection["macro"][key]) for key in ("precision", "recall", "f1")),
        *(rate(mock["macro"][key]) for key in ("precision", "recall", "f1")),
        rate(mock["accuracy"]),
        rate(mock["majority_baseline_accuracy"]),
    ]
    for number in expected:
        assert number in text, f"{number} from the fresh report is not in docs/EVALUATION.md"


def test_the_readme_numbers_match_a_fresh_report(fresh_report: dict[str, Any]) -> None:
    text = read("README.md")
    detection = fresh_report["deterministic"]["metrics"]
    mock = fresh_report["mock"]["metrics"]

    for number in (
        *(rate(detection["micro"][key]) for key in ("precision", "recall", "f1")),
        rate(mock["accuracy"]),
        rate(mock["majority_baseline_accuracy"]),
    ):
        assert number in text, f"{number} from the fresh report is not in the README"


def test_the_readme_lists_every_api_operation() -> None:
    text = strip_code_blocks(read("README.md"))

    missing = [f"{method} {path}" for method, path in sorted(openapi_operations()) if f"{method} {path}" not in text]

    assert missing == [], f"not in the README's API section: {missing}"


def test_the_readme_mentions_only_real_api_operations() -> None:
    text = strip_code_blocks(read("README.md"))
    operations = openapi_operations()

    mentioned = {
        (method, normalize(path))
        for method, path in re.findall(r"\b(GET|POST|PUT|PATCH|DELETE) (/[A-Za-z0-9_/{}.\-]*)", text)
    }

    assert mentioned - operations == set()


def test_the_readme_documents_the_read_only_code() -> None:
    assert access.READ_ONLY_CODE in read("README.md")


def test_gitignore_excludes_env_files() -> None:
    path = ROOT / ".gitignore"
    if not path.is_file():
        pytest.skip("no .gitignore yet (the project is not a Git repository)")
    lines = {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()}

    assert lines & {".env", "/.env", ".env*", "*.env"}, ".gitignore does not exclude .env"