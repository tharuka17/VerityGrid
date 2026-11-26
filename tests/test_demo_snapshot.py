import json
import re
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Finding
from app.db.session import build_engine
from app.evaluation import harness
from app.schemas.rules import check_rule_columns, parse_rules
from app.services import evaluation_service
from app.services.ingestion import ingest_csv
from scripts import seed_demo

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "evaluation" / "llm_run_2026-11-19.json"
SAMPLE_DIR = ROOT / "data" / "sample"
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
]


def roundtrip(value: Any) -> Any:
    """What a JSON file would hold: tuples become lists."""
    return json.loads(json.dumps(value))


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fresh() -> dict[str, Any]:
    """A report built by the harness now (deterministic and mock parts; no provider, no network)."""
    return harness.build_report(repeats=1, warmup=False)


# --- The committed snapshot of the real-model run ----------------------------------------------------------


def test_the_snapshot_is_accepted_by_the_real_service(snapshot: dict[str, Any]) -> None:
    answer = evaluation_service.load_report(SNAPSHOT)

    assert answer.available is True
    assert answer.report == snapshot


def test_the_snapshot_fingerprints_match_the_current_code(snapshot: dict[str, Any], fresh: dict[str, Any]) -> None:
    assert snapshot["fingerprints"]["dataset"] == fresh["fingerprints"]["dataset"]
    assert snapshot["fingerprints"]["results"] == fresh["fingerprints"]["results"]


def test_the_snapshot_results_match_the_current_code(snapshot: dict[str, Any], fresh: dict[str, Any]) -> None:
    assert snapshot["deterministic"]["metrics"] == roundtrip(fresh["deterministic"]["metrics"])
    assert snapshot["mock"]["metrics"] == roundtrip(fresh["mock"]["metrics"])


def test_the_snapshot_is_a_full_real_model_run(snapshot: dict[str, Any]) -> None:
    llm = snapshot["llm"]

    assert llm["ran"] is True and llm["provider"].startswith("gemini:")
    assert llm["reproducible"] is False
    assert llm["cost"]["measured"] is False and llm["token_usage"]["measured"] is False
    assert snapshot["run"]["is_subset"] is False and len(snapshot["run"]["cases_selected"]) == 13
    assert llm["metrics"]["n"] == 63


def test_the_snapshot_counts_are_consistent(snapshot: dict[str, Any]) -> None:
    llm = snapshot["llm"]
    coverage, metrics = llm["coverage"], llm["metrics"]
    correct = sum(item["tp"] for item in metrics["per_class"].values())

    assert coverage["columns"] == metrics["n"] == 63
    assert coverage["answered"] + coverage["unanswered"] == 63
    assert llm["outcomes"].get("ok", 0) == coverage["answered"] and sum(llm["outcomes"].values()) == 63
    assert metrics["no_prediction_count"] == coverage["unanswered"]
    assert llm["answered_only"]["n"] == coverage["answered"]
    assert correct == round(metrics["accuracy"] * 63)  # a hand-edited accuracy would not match the counts
    assert llm["answered_only"]["accuracy"] == pytest.approx(correct / coverage["answered"])


def test_the_snapshot_confusion_matrix_rows_match_the_supports(snapshot: dict[str, Any]) -> None:
    metrics = snapshot["llm"]["metrics"]

    for label, row in metrics["matrix"].items():
        assert sum(row.values()) == metrics["per_class"][label]["support"], label


def test_the_snapshot_run_was_not_cut_short(snapshot: dict[str, Any]) -> None:
    llm = snapshot["llm"]

    assert llm["guard"]["stopped_early"] is False and llm["guard"]["calls_skipped_after_stop"] == 0
    assert llm["guard"]["calls_sent"] == llm["provider_calls"]["count"] == 63


def test_no_secret_patterns_in_the_snapshot() -> None:
    text = SNAPSHOT.read_text(encoding="utf-8")

    for pattern in SECRET_PATTERNS:
        assert not pattern.search(text), pattern.pattern


def test_the_api_image_installs_the_snapshot() -> None:
    lines = [
        line.strip()
        for line in (ROOT / "Dockerfile.api").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    text = "\n".join(lines)

    assert any(line.startswith("COPY ") and "data/evaluation" in line for line in lines)
    assert SNAPSHOT.name in text
    assert "--skip-review" in text and "--llm" not in text


# --- The demo seed applies sample rules ------------------------------------------------------------------------


def test_the_demo_seed_applies_the_sample_rules(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'rules_demo.db'}"
    counts = seed_demo.seed(url, sample_dir=SAMPLE_DIR)

    engine = build_engine(url)
    try:
        with Session(engine) as db:
            by_category = dict(db.execute(select(Finding.category, func.count()).group_by(Finding.category)).all())
    finally:
        engine.dispose()

    assert by_category == {
        "duplicate_rows": 3,
        "missing_values": 1,
        "potential_email": 3,
        "potential_phone": 2,
        "numeric_rule_violation": 2,
        "invalid_date": 1,
    }
    assert (counts["findings"], counts["pending"], counts["approved"], counts["rejected"]) == (12, 10, 1, 1)


def test_the_sample_rules_are_valid_for_their_files() -> None:
    sample_files = {filename for filename, _ in seed_demo.SAMPLES}

    assert set(seed_demo.SAMPLE_RULES) <= sample_files
    for filename, text in seed_demo.SAMPLE_RULES.items():
        config = parse_rules(text)
        assert config is not None
        check_rule_columns(config, ingest_csv((SAMPLE_DIR / filename).read_bytes()).dataframe.columns)