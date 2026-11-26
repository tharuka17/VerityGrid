import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["README.md", "SECURITY.md", "docs/ARCHITECTURE.md", "docs/THREAT_MODEL.md", "docs/EVALUATION.md", "docs/DEPLOYMENT.md"]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8-sig")


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+|\n+", text)


# --- Statements that are no longer true must be gone ------------------------------------------------------


@pytest.mark.parametrize("name", DOCS)
def test_no_document_says_the_rules_are_unreachable_over_http(name: str) -> None:
    text = read(name).lower()

    for phrase in (
        "not configurable over http",
        "never reported over http",
        "rules are not active for uploads",
        "rules are never active",
        "not active for uploads yet",
    ):
        assert phrase not in text, f"{name} still says: {phrase}"


@pytest.mark.parametrize("name", ["SECURITY.md", "docs/THREAT_MODEL.md"])
def test_fixed_risks_are_not_listed_as_open(name: str) -> None:
    text = read(name).lower()

    assert "silently creates an empty" not in text, f"{name} lists the empty-database risk as open"
    assert "`debug=true` is not guarded" not in text
    assert "debug=true is not guarded" not in text


def test_the_findings_caption_matches_the_feature() -> None:
    source = (ROOT / "ui" / "views" / "findings.py").read_text(encoding="utf-8")

    assert "not active for uploads yet" not in source
    assert "detection rules" in source


# --- What the documents must now say --------------------------------------------------------------------------


def test_the_documents_describe_the_rules_feature() -> None:
    assert "rules" in read("README.md").lower() and "POST /scans" in read("README.md")
    security = read("SECURITY.md")
    for fragment in ("20,000", "allowlist", "never echoed", "never stored"):
        assert fragment in security, fragment
    assert "T16" in read("docs/THREAT_MODEL.md")


def test_the_documents_describe_the_startup_and_debug_guards() -> None:
    security = read("SECURITY.md").lower()

    assert "refuses to start" in security
    assert "forced off" in security
    assert "test_startup_safety" in security
    assert "refuses to start" in read("README.md").lower()


def test_the_documents_describe_the_regex_fix() -> None:
    assert "test_detector_robustness" in read("SECURITY.md")
    assert "T17" in read("docs/THREAT_MODEL.md")


def test_the_documents_describe_the_paced_real_model_command() -> None:
    assert "llm_run" in read("README.md")
    assert "llm_run" in read("docs/EVALUATION.md")
    assert "test_llm_run" in read("SECURITY.md")


def test_the_documents_describe_the_snapshot_in_the_image() -> None:
    assert "llm_run_2026-11-19.json" in read("docs/DEPLOYMENT.md")
    assert "snapshot" in read("SECURITY.md").lower()
    assert "T18" in read("docs/THREAT_MODEL.md")


def test_the_demo_numbers_are_current() -> None:
    for name in ("README.md", "docs/DEPLOYMENT.md"):
        text = read(name)
        assert "12 findings" in text, name
        assert "9 findings" not in text, name


def test_the_documents_do_not_claim_security_they_lack() -> None:
    for name in DOCS:
        text = read(name).lower()
        for claim in ("enterprise-grade security", "fully secure", "production-ready", "soc 2", "iso 27001"):
            if claim in text:
                allowed = [s for s in sentences(text) if claim in s]
                assert all(("not" in s or "no " in s or "nothing" in s) for s in allowed), f"{name}: {allowed}"


def test_the_threat_model_threats_are_numbered_without_gaps() -> None:
    ids = [int(match) for match in re.findall(r"^\| T(\d+) \|", read("docs/THREAT_MODEL.md"), flags=re.MULTILINE)]

    assert ids == list(range(1, len(ids) + 1))
    assert len(ids) >= 18


def test_every_test_file_named_in_the_security_notes_exists() -> None:
    named = set(re.findall(r"`?(test_[a-z_]+)\.py`?", read("SECURITY.md") + read("docs/THREAT_MODEL.md")))
    named |= set(re.findall(r"`(test_[a-z_]+)`", read("SECURITY.md")))
    existing = {path.stem for path in (ROOT / "tests").glob("test_*.py")}

    missing = sorted(name for name in named if name not in existing and not name.startswith("test_the_") and not name.startswith("test_review_") and not name.startswith("test_backend_") and not name.startswith("test_llm_run_"))
    assert missing == [], missing
    
def test_the_architecture_shows_the_rules_step() -> None:
    assert "optional rules" in read("docs/ARCHITECTURE.md")
    
def test_the_documents_describe_automatic_deploys() -> None:
    guide = read("docs/DEPLOYMENT.md")

    assert "autoDeployTrigger" in guide and "build filter" in guide.lower()
    for name in ("README.md", "PROJECT_SPEC.md", "docs/DEPLOYMENT.md"):
        assert "automatically" in read(name).lower(), name
    for name in DOCS:
        assert "`autoDeploy` is off" not in read(name), name
        
def test_the_status_file_is_current() -> None:
    text = read("IMPLEMENTATION_STATUS.md")

    for stale in ("Nothing has been deployed", "`autoDeploy: false`", "not been deployed", "**Not started.** Instructions"):
        assert stale not in text, stale
    for fact in ("014506f", "2b9a025", "74b2af3", "af688be", "autoDeployTrigger", "veritygrid-ui.onrender.com"):
        assert fact in text, fact


def test_the_deployment_guide_records_the_auto_deploy_result() -> None:
    guide = read("docs/DEPLOYMENT.md")

    assert "af688be" in guide and "New commit via Auto-Deploy" in guide