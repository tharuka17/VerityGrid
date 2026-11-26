import ast
import builtins
import json
from pathlib import Path
from typing import Any

import pytest

from app.ai.provider import ClassificationProvider, ClassificationSource, ProviderUnavailableError
from app.evaluation import dataset as ds
from app.evaluation import harness, llm_run

ROOT = Path(__file__).resolve().parent.parent
GOOD = {
    "label": "internal",
    "explanation": "Operational data.",
    "supporting_refs": ["column_name"],
    "review_recommendation": "review_recommended",
}
DOWN = ProviderUnavailableError("offline")
FIRST = ds.CASES[0]
FIRST_COLUMNS = len(FIRST.columns)
FIRST_INTERNAL = sum(1 for column in FIRST.columns if column.label == "internal")
TOTAL = sum(len(case.columns) for case in ds.CASES)
ALL_INTERNAL = sum(1 for case in ds.CASES for column in case.columns if column.label == "internal")


class StubLlm(ClassificationProvider):
    name = "llm-stub"
    source = ClassificationSource.LLM
    calls = 0

    def classify(self, metadata: Any) -> Any:
        self.calls += 1
        return GOOD


class FailingLlm(StubLlm):
    def classify(self, metadata: Any) -> Any:
        self.calls += 1
        raise DOWN


class SequenceLlm(StubLlm):
    def __init__(self, outcomes: list[Any], default: Any) -> None:
        self.outcomes = list(outcomes)
        self.default = default

    def classify(self, metadata: Any) -> Any:
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else self.default
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def pace(inner: ClassificationProvider, *, pause: float = 0.0, stop: int | None = None, sleeps: list[float] | None = None) -> Any:
    sink = sleeps if sleeps is not None else []
    return llm_run.pace_provider(inner, pause_seconds=pause, stop_after_failures=stop, sleep=sink.append)


@pytest.fixture(scope="module")
def full_run() -> tuple[dict[str, Any], list[float]]:
    sleeps: list[float] = []
    report = llm_run.run_llm_evaluation(StubLlm(), pause_seconds=1.5, sleep=sleeps.append)
    return report, sleeps


# --- Pacing and the run-wide stop ---------------------------------------------------------------


def test_pacing_waits_between_calls_but_not_before_the_first() -> None:
    sleeps: list[float] = []
    paced = pace(StubLlm(), pause=2.5, sleeps=sleeps)

    for _ in range(3):
        paced.classify(object())

    assert sleeps == [2.5, 2.5]
    assert paced.calls == 3 and paced.waited_seconds == 5.0


def test_no_pause_means_no_sleep() -> None:
    sleeps: list[float] = []
    paced = pace(StubLlm(), pause=0, sleeps=sleeps)

    paced.classify(object())
    paced.classify(object())

    assert sleeps == []


def test_the_run_stops_after_consecutive_failures() -> None:
    inner = FailingLlm()
    sleeps: list[float] = []
    paced = pace(inner, pause=1.0, stop=2, sleeps=sleeps)

    for _ in range(2):
        with pytest.raises(ProviderUnavailableError):
            paced.classify(object())
    with pytest.raises(ProviderUnavailableError):
        paced.classify(object())  # not sent, and no pause before it

    assert paced.stopped is True and paced.skipped == 1
    assert inner.calls == 2
    assert sleeps == [1.0]


def test_a_success_resets_the_failure_streak() -> None:
    inner = SequenceLlm([DOWN, GOOD, DOWN, GOOD], default=GOOD)
    paced = pace(inner, stop=2)

    for expect_failure in (True, False, True, False):
        if expect_failure:
            with pytest.raises(ProviderUnavailableError):
                paced.classify(object())
        else:
            paced.classify(object())

    assert paced.stopped is False and inner.calls == 4


def test_the_wrapper_keeps_the_provider_identity() -> None:
    paced = pace(StubLlm())

    assert (paced.name, paced.source) == ("llm-stub", ClassificationSource.LLM)


# --- The evaluation run ---------------------------------------------------------------------------


def test_a_fully_answered_run(full_run: tuple[dict[str, Any], list[float]]) -> None:
    report, sleeps = full_run
    llm = report["llm"]

    assert sleeps == [1.5] * (TOTAL - 1)
    assert llm["coverage"] == {"columns": TOTAL, "answered": TOTAL, "unanswered": 0, "answered_share": 1.0}
    assert llm["answered_only"]["accuracy"] == llm["metrics"]["accuracy"]
    assert llm["metrics"]["accuracy"] == pytest.approx(ALL_INTERNAL / TOTAL)
    assert llm["guard"] == {
        "stopped_early": False,
        "calls_skipped_after_stop": 0,
        "calls_sent": TOTAL,
        "seconds_waited": round(1.5 * (TOTAL - 1), 1),
    }
    assert llm["run_config"]["pause_seconds"] == 1.5
    assert llm["provider_calls"]["count"] == TOTAL and llm["provider_calls"]["failures"] == 0


def test_the_deterministic_and_mock_results_are_unchanged(full_run: tuple[dict[str, Any], list[float]]) -> None:
    report, _ = full_run

    assert report["fingerprints"]["results"] == harness.build_report(repeats=1, warmup=False)["fingerprints"]["results"]
    assert report["run"]["is_subset"] is False


def test_the_latency_note_mentions_the_pauses(full_run: tuple[dict[str, Any], list[float]]) -> None:
    report, _ = full_run

    assert "pause" in report["llm"]["provider_calls"]["note"].lower()


def test_a_partly_answered_run_reports_both_accuracies() -> None:
    provider = SequenceLlm([GOOD] * FIRST_COLUMNS, default=DOWN)

    report = llm_run.run_llm_evaluation(provider, pause_seconds=0, stop_after_failures=None)

    llm = report["llm"]
    assert (llm["coverage"]["answered"], llm["coverage"]["unanswered"]) == (FIRST_COLUMNS, TOTAL - FIRST_COLUMNS)
    assert llm["answered_only"]["n"] == FIRST_COLUMNS
    assert llm["metrics"]["no_prediction_count"] == TOTAL - FIRST_COLUMNS
    assert llm["metrics"]["accuracy"] == pytest.approx(FIRST_INTERNAL / TOTAL)  # unanswered columns count as wrong
    assert llm["answered_only"]["accuracy"] == pytest.approx(FIRST_INTERNAL / FIRST_COLUMNS)


def test_the_run_wide_stop_limits_the_calls() -> None:
    provider = FailingLlm()

    report = llm_run.run_llm_evaluation(provider, pause_seconds=0, stop_after_failures=4)

    llm = report["llm"]
    assert provider.calls == 4 == llm["provider_calls"]["count"]
    assert llm["provider_calls"]["failures"] == 4
    assert llm["provider_calls"]["latency"]["n"] == 4
    assert llm["guard"]["stopped_early"] is True and llm["guard"]["calls_skipped_after_stop"] > 0
    assert llm["coverage"]["answered"] == 0 and llm["answered_only"] is None


def test_a_partial_run_is_marked() -> None:
    report = llm_run.run_llm_evaluation(StubLlm(), cases=ds.CASES[:1], pause_seconds=0)

    assert report["run"]["is_subset"] is True and report["run"]["cases_selected"] == [FIRST.case_id]
    assert report["dataset"]["columns"] == FIRST_COLUMNS
    assert report["llm"]["coverage"]["columns"] == FIRST_COLUMNS
    assert "PARTIAL RUN" in llm_run.render_markdown(report)


def test_the_markdown_has_the_coverage_section(full_run: tuple[dict[str, Any], list[float]]) -> None:
    text = llm_run.render_markdown(full_run[0])

    assert f"Coverage: {TOTAL} of {TOTAL} columns" in text
    assert "over answered columns only" in text
    assert "single, non-reproducible run" in text
    assert "PARTIAL RUN" not in text


# --- The command ------------------------------------------------------------------------------------------


def test_the_default_options() -> None:
    args = llm_run.parse_args([])

    assert (args.pause_seconds, args.stop_after_failures, args.cases, args.yes) == (6.0, 8, None, False)


@pytest.mark.parametrize(
    "argv",
    [["--pause-seconds", "-1"], ["--pause-seconds", "301"], ["--stop-after-failures", "0"], ["--stop-after-failures", "1001"]],
    ids=["pause-negative", "pause-too-long", "stop-zero", "stop-too-large"],
)
def test_bad_options_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        llm_run.parse_args(argv)

    assert excinfo.value.code == 2


def test_unknown_case_ids_are_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        llm_run.parse_args(["--cases", "nope"])

    assert excinfo.value.code == 2


def test_the_command_runs_and_writes_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(llm_run, "load_provider", lambda: (StubLlm(), None))

    code = llm_run.main(["--yes", "--pause-seconds", "0", "--cases", FIRST.case_id, "--output-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert code == 0
    data = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    assert data["llm"]["ran"] is True and data["llm"]["coverage"]["answered"] == FIRST_COLUMNS
    assert "Coverage:" in (tmp_path / "evaluation_report.md").read_text(encoding="utf-8")
    assert "single, non-reproducible run" in out and "PARTIAL RUN" in out


def test_default_folders_never_overwrite_the_reproducible_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(llm_run, "load_provider", lambda: (StubLlm(), None))

    assert llm_run.main(["--yes", "--pause-seconds", "0", "--cases", FIRST.case_id]) == 0
    assert llm_run.main(["--yes", "--pause-seconds", "0"]) == 0

    assert (tmp_path / "reports" / "llm_trial" / "evaluation_report.json").exists()
    assert (tmp_path / "reports" / "llm_run" / "evaluation_report.json").exists()
    assert not (tmp_path / "reports" / "evaluation_report.json").exists()


def test_declining_the_confirmation_makes_no_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = StubLlm()
    monkeypatch.setattr(llm_run, "load_provider", lambda: (provider, None))
    monkeypatch.setattr(builtins, "input", lambda prompt="": "no")

    code = llm_run.main(["--pause-seconds", "0", "--cases", FIRST.case_id, "--output-dir", str(tmp_path)])

    assert code == 1 and provider.calls == 0
    assert list(tmp_path.iterdir()) == []


def test_no_provider_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_run, "load_provider", lambda: (None, "Not configured."))

    assert llm_run.main(["--yes", "--output-dir", str(tmp_path)]) == 1
    assert list(tmp_path.iterdir()) == []


def test_a_mock_provider_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_run, "load_provider", lambda: (harness.MockClassificationProvider(), None))

    assert llm_run.main(["--yes", "--output-dir", str(tmp_path)]) == 1
    assert list(tmp_path.iterdir()) == []


def test_an_interrupt_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(llm_run, "load_provider", lambda: (StubLlm(), None))
    monkeypatch.setattr(llm_run, "run_llm_evaluation", interrupt)

    assert llm_run.main(["--yes", "--output-dir", str(tmp_path)]) == 130
    assert list(tmp_path.iterdir()) == []


def test_a_settings_error_is_reported_without_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> Any:
        raise RuntimeError("SECRET-DETAIL")

    monkeypatch.setattr(llm_run, "load_provider", broken)

    assert llm_run.main(["--yes", "--output-dir", str(tmp_path)]) == 1
    assert "SECRET" not in capsys.readouterr().out


def test_the_command_imports_no_network_code() -> None:
    tree = ast.parse((ROOT / "app" / "evaluation" / "llm_run.py").read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)

    allowed = {"argparse", "json", "time", "collections", "pathlib", "typing", "app"}
    for module in modules:
        assert module.split(".")[0] in allowed, module