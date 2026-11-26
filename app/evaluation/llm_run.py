"""Paced evaluation of a genuine language model.

    python -m app.evaluation.llm_run [--cases ID [ID ...]] [--pause-seconds 6] [--stop-after-failures 8] [--yes]

The reproducible harness (python -m app.evaluation.harness) sends provider calls back to back, so a rate-limited
free tier turns quota errors into "wrong answers". This command wraps the configured provider with pacing and a
run-wide stop, runs the SAME evaluation (harness.build_report) and adds what a run like this needs to be read
honestly: how many columns received any answer (coverage) and the accuracy over answered columns only.

It is a single, non-reproducible run. Cost and token usage are not measured: read them in the provider's usage page.
Output goes to reports/llm_run (reports/llm_trial for a partial run), never over the reproducible report. The
deterministic and mock parts of the report, and its results fingerprint, are unchanged by this command.
"""

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from app.ai.provider import ClassificationProvider, ClassificationSource, ProviderUnavailableError
from app.evaluation import dataset as ds
from app.evaluation import harness
from app.evaluation import metrics as mx

DEFAULT_PAUSE_SECONDS = 6.0
DEFAULT_STOP_AFTER_FAILURES = 8
RUN_DIR = Path("reports") / "llm_run"
TRIAL_DIR = Path("reports") / "llm_trial"


def load_provider() -> tuple[ClassificationProvider | None, str | None]:
    """The configured provider (follows .env) and a note when there is none. Tests replace this function."""
    return harness.resolve_llm_provider(harness.Settings())


def pace_provider(
    inner: ClassificationProvider,
    *,
    pause_seconds: float,
    stop_after_failures: int | None,
    sleep: Callable[[float], None] = time.sleep,
) -> ClassificationProvider:
    """Wait `pause_seconds` before every call except the first, and stop sending calls after
    `stop_after_failures` consecutive failed calls (later calls fail at once, without reaching the provider).

    The wrapper keeps its own record of the calls that really reached the provider (count, failures and the
    latency of each call without the pause), because the harness also counts the calls this wrapper rejects.
    """

    class _Paced(ClassificationProvider):
        name = inner.name
        source = inner.source

        def __init__(self) -> None:
            self.calls = 0
            self.failures = 0
            self.latencies_ms: list[float] = []
            self.consecutive_failures = 0
            self.stopped = False
            self.skipped = 0
            self.waited_seconds = 0.0

        def classify(self, metadata: Any) -> Any:
            if self.stopped:
                self.skipped += 1
                raise ProviderUnavailableError("The run stopped calling the provider after repeated failures.")
            if self.calls > 0 and pause_seconds > 0:
                sleep(pause_seconds)
                self.waited_seconds += pause_seconds
            self.calls += 1
            started = time.perf_counter()
            try:
                answer = inner.classify(metadata)
            except Exception:
                self.failures += 1
                self.consecutive_failures += 1
                if stop_after_failures is not None and self.consecutive_failures >= stop_after_failures:
                    self.stopped = True
                raise
            finally:
                self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
            self.consecutive_failures = 0
            return answer

    return _Paced()


def _prediction_pairs(llm: dict[str, Any], cases: Sequence[ds.LabeledCase]) -> list[tuple[str, str]]:
    """(true label, predicted label) for every column, rebuilt from the report's list of wrong or unanswered columns."""
    wrong = {(item["case"], item["column"]): item["predicted"] for item in llm["misclassified"]}
    return [
        (column.label, wrong.get((case.case_id, column.name), column.label))
        for case in cases
        for column in case.columns
    ]


def run_llm_evaluation(
    provider: ClassificationProvider,
    *,
    cases: Sequence[ds.LabeledCase] = ds.CASES,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    stop_after_failures: int | None = DEFAULT_STOP_AFTER_FAILURES,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    paced = pace_provider(
        provider, pause_seconds=pause_seconds, stop_after_failures=stop_after_failures, sleep=sleep
    )
    report = harness.build_report(repeats=1, warmup=False, llm_provider=paced, cases=cases)
    llm = report["llm"]
    if not llm.get("ran"):
        raise RuntimeError("The provider was not run: it is not a genuine language model.")

    pairs = _prediction_pairs(llm, cases)
    if len(pairs) != llm["metrics"]["n"]:
        raise RuntimeError("The scored columns do not match the dataset; the report would be misleading.")
    answered = [pair for pair in pairs if pair[1] != mx.NO_PREDICTION]
    total = len(pairs)
    llm["coverage"] = {
        "columns": total,
        "answered": len(answered),
        "unanswered": total - len(answered),
        "answered_share": round(len(answered) / total, 4) if total else None,
    }
    llm["answered_only"] = mx.multiclass_report(ds.CLASSIFICATION_LABELS, answered) if answered else None
    llm["guard"] = {
        "stopped_early": paced.stopped,
        "calls_skipped_after_stop": paced.skipped,
        "calls_sent": paced.calls,
        "seconds_waited": round(paced.waited_seconds, 1),
    }
    llm["run_config"] = {
        "pause_seconds": pause_seconds,
        "stop_after_consecutive_failures": stop_after_failures,
        "workflow_max_consecutive_failures": 3,
    }
        # The harness meters every classify() call, including those the stop rule rejected without sending, so its
    # count and latencies would overstate what reached the provider. Use the wrapper's own record instead.
    llm["provider_calls"] = {
        "count": paced.calls,
        "failures": paced.failures,
        "latency": mx.summarize_latencies(paced.latencies_ms),
        "note": "Calls that actually reached the provider (calls rejected after the run-wide stop are not counted; "
        "see the pacing line). Per-call latency excludes the pauses; the adapter may retry internally, so HTTP "
        "requests can exceed this number.",
    }
    report["run"]["cases_selected"] = [case.case_id for case in cases]
    report["run"]["is_subset"] = len(cases) != len(ds.CASES)
    return report


def _rate(value: object) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    """The harness's Markdown report plus coverage and pacing (and a banner for a partial run)."""
    llm, run = report["llm"], report["run"]
    coverage, answered, guard, config = llm["coverage"], llm["answered_only"], llm["guard"], llm["run_config"]
    lines: list[str] = []
    if run.get("is_subset"):
        lines += [
            f"**PARTIAL RUN: only {len(run['cases_selected'])} of {len(ds.CASES)} cases were sent to the model "
            f"({', '.join(run['cases_selected'])}). Do not compare these numbers with a full run.**",
            "",
        ]
    answered_text = (
        f"{_rate(answered['accuracy'])} over {answered['n']} answered columns" if answered else "n/a (no column was answered)"
    )
    lines += [
        harness.render_markdown(report),
        "",
        "## Coverage and pacing of the real-model run",
        "",
        f"Coverage: {coverage['answered']} of {coverage['columns']} columns received a valid answer; "
        f"{coverage['unanswered']} did not.",
        "",
        f"Accuracy counting unanswered columns as wrong: {_rate(llm['metrics']['accuracy'])}. "
        f"Accuracy over answered columns only: {answered_text}.",
        "",
        "Unanswered columns (rate limits, outages, invalid output) say nothing about the model's judgement, so the "
        "answered-only figure leaves them out. That can flatter the model if the failures are not random, and fewer "
        "answered columns mean wider uncertainty.",
        "",
        f"Pacing: {guard['calls_sent']} calls sent, {guard['seconds_waited']} s spent waiting "
        f"(pause {config['pause_seconds']} s between calls); stopped early: "
        f"{'yes, ' + str(guard['calls_skipped_after_stop']) + ' further calls were not sent' if guard['stopped_early'] else 'no'}.",
        "",
        "This is a single, non-reproducible run. Cost and token usage are not measured: read them in the provider's "
        "usage page. In the real-model section above, workflow latency includes the pauses between calls; the "
        "deterministic and mock sections do not.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.llm_run",
        description="Run the evaluation against the configured language model, paced to respect rate limits.",
    )
    parser.add_argument("--cases", nargs="+", metavar="CASE_ID", default=None, help="evaluate only these case ids (a trial)")
    parser.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS, help="wait between provider calls (0-300)")
    parser.add_argument(
        "--stop-after-failures", type=int, default=DEFAULT_STOP_AFTER_FAILURES, help="stop sending calls after this many failures in a row (1-1000)"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="default: reports/llm_run, or reports/llm_trial with --cases")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args(argv)
    if not 0 <= args.pause_seconds <= 300:
        parser.error("--pause-seconds must be between 0 and 300")
    if not 1 <= args.stop_after_failures <= 1000:
        parser.error("--stop-after-failures must be between 1 and 1000")
    if args.cases is not None:
        known = {case.case_id for case in ds.CASES}
        unknown = sorted(set(args.cases) - known)
        if unknown:
            parser.error(f"unknown case id(s): {', '.join(unknown)}. Known ids: {', '.join(sorted(known))}")
    return args


def _confirm(provider: ClassificationProvider, cases: Sequence[ds.LabeledCase], pause_seconds: float, assume_yes: bool) -> bool:
    calls = sum(len(case.columns) for case in cases)
    print(
        f"About to make up to {calls} calls through '{provider.name}': about {calls * pause_seconds / 60:.1f} minutes of "
        f"pauses at {pause_seconds:g} s, plus the calls themselves."
    )
    print(
        "Whether they are billed depends on the billing tier of the project behind your key (check it in the provider's "
        "console). On a free tier the provider may use the content to improve its products. Only sanitized column "
        "metadata of the synthetic dataset is sent, never cell values."
    )
    if assume_yes:
        return True
    try:
        return input("Type 'yes' to continue: ").strip().lower() == "yes"
    except EOFError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    chosen = set(args.cases) if args.cases is not None else None
    selected = ds.CASES if chosen is None else tuple(case for case in ds.CASES if case.case_id in chosen)
    output_dir = args.output_dir or (TRIAL_DIR if chosen is not None else RUN_DIR)

    try:
        provider, note = load_provider()
    except Exception as error:
        print(f"The provider settings could not be loaded ({type(error).__name__}).")
        return 1
    if provider is None:
        print(f"No language-model provider is configured. {note or ''}".strip())
        return 1
    if provider.source is not ClassificationSource.LLM:
        print(f"The configured provider '{provider.name}' is not a genuine language model, so nothing was run.")
        return 1
    if not _confirm(provider, selected, args.pause_seconds, args.yes):
        print("Cancelled. No provider calls were made.")
        return 1

    try:
        report = run_llm_evaluation(
            provider, cases=selected, pause_seconds=args.pause_seconds, stop_after_failures=args.stop_after_failures
        )
    except KeyboardInterrupt:
        print("Interrupted. No report was written.")
        return 130

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / harness.REPORT_JSON
    markdown_path = output_dir / harness.REPORT_MARKDOWN
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    llm = report["llm"]
    coverage, answered, guard = llm["coverage"], llm["answered_only"], llm["guard"]
    print("VerityGrid paced language-model evaluation (a single, non-reproducible run)")
    print(f"  provider: {llm['provider']}")
    if report["run"]["is_subset"]:
        print(f"  PARTIAL RUN: {len(selected)} of {len(ds.CASES)} cases ({', '.join(report['run']['cases_selected'])})")
    print(f"  coverage: {coverage['answered']} of {coverage['columns']} columns answered")
    print(
        f"  accuracy counting unanswered as wrong: {_rate(llm['metrics']['accuracy'])}; "
        f"over answered columns only: {_rate(answered['accuracy']) if answered else 'n/a'}"
    )
    print(
        f"  calls sent: {guard['calls_sent']} (failures: {llm['provider_calls']['failures']}); waited {guard['seconds_waited']} s; "
        f"stopped early: {'YES' if guard['stopped_early'] else 'no'}"
    )
    print(f"  wrote {json_path}")
    print(f"  wrote {markdown_path}")
    print("  Check token usage and any billing in the provider's console: this command does not measure cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())