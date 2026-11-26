"""A bounded LangGraph workflow for one dataset scan.

Steps (names are used in results and logs):
  validate_input -> profile_dataset -> run_checks -> [classify_columns] -> consolidate
  -> apply_policy -> finalize

This module only orchestrates. Profiling, detection and classification are the existing
services, called as they are. Persisting results is the scan service's job: this workflow
reads and computes, and never writes to a database, a file or the network (apart from whatever
the injected classifier's provider does).

Safety and bounds:
  - The graph is acyclic. The step limit (RECURSION_LIMIT) is only a backstop.
  - Classification is opt-in and cost-aware: at most `max_classified_columns` provider calls,
    and a circuit breaker stops calling the provider after `max_consecutive_failures` failures
    in a row. The workflow adds no retries; the provider owns bounded retries.
  - The DataFrame is NOT part of the graph state. Nodes reach it through closures, so state
    (and anything that might record state later, such as a checkpointer or an opt-in tracing
    integration) holds only profiles, findings and classification results.
  - Nothing here deletes data, changes permissions or sets a review status. Findings keep their
    default `pending` review status; humans decide in the existing review API.

Failure semantics:
  - validate, profile and detect are FATAL: an error there gives status FAILED.
  - classify, consolidate and policy are NON-FATAL: an error there gives status
    PARTIALLY_COMPLETED and keeps the deterministic results.
  - Provider failures never raise out of the classifier, so they surface as failed columns.
  - Issues record a step name, a stable code and an exception class name. Never messages, values
    or column names. Logs follow the same rule.

Retry behavior: the compute phase has no side effects, so re-running a whole workflow is always
safe, and two runs on the same input give the same results (apart from timings).
"""

import logging
import operator
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from app.ai.classifier import ClassificationResult, ColumnClassifier, ResultStatus
from app.ai.provider import ClassificationLabel, ReviewRecommendation
from app.db.models import Severity
from app.schemas.findings import DetectedFinding, DetectionConfig, FindingCategory
from app.services.detectors import run_detectors
from app.services.profiling import DatasetProfileResult, profile_dataframe

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 20  # the graph has at most 7 steps; this is a backstop, not a loop budget
MAX_CLASSIFIED_COLUMNS_LIMIT = 200
MAX_FAILURE_STREAK_LIMIT = 10

STEP_VALIDATE = "validate_input"
STEP_PROFILE = "profile_dataset"
STEP_CHECKS = "run_checks"
STEP_CLASSIFY = "classify_columns"
STEP_CONSOLIDATE = "consolidate"
STEP_POLICY = "apply_policy"
STEP_FINALIZE = "finalize"


# --- Public types -----------------------------------------------------------------


class WorkflowStatus(StrEnum):
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_RISK_RANK = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
_SEVERITY_RISK = {
    Severity.LOW: RiskLevel.LOW,
    Severity.MEDIUM: RiskLevel.MEDIUM,
    Severity.HIGH: RiskLevel.HIGH,
}
_SENSITIVE_LABELS = frozenset({ClassificationLabel.PERSONAL, ClassificationLabel.FINANCIAL})
_PERSONAL_PATTERN_CATEGORIES = frozenset(
    {FindingCategory.POTENTIAL_EMAIL, FindingCategory.POTENTIAL_PHONE}
)


def _highest(levels: Iterable[RiskLevel]) -> RiskLevel:
    return max(levels, key=_RISK_RANK.__getitem__, default=RiskLevel.LOW)


def _check_int(name: str, value: object, low: int, high: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}.")


@dataclass(frozen=True, slots=True)
class WorkflowOptions:
    """What the workflow may do. Classification is off unless requested."""

    classify: bool = False
    max_classified_columns: int = 50  # one provider call per classified column
    max_consecutive_failures: int = 3  # circuit breaker threshold
    detection_config: DetectionConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.classify, bool):
            raise ValueError("classify must be a boolean.")
        _check_int("max_classified_columns", self.max_classified_columns, 1, MAX_CLASSIFIED_COLUMNS_LIMIT)
        _check_int("max_consecutive_failures", self.max_consecutive_failures, 1, MAX_FAILURE_STREAK_LIMIT)
        if self.detection_config is not None and not isinstance(self.detection_config, DetectionConfig):
            raise ValueError("detection_config must be a DetectionConfig.")


@dataclass(frozen=True, slots=True)
class WorkflowIssue:
    """A recorded problem: where, a stable code, and the exception CLASS name if any."""

    step: str
    code: str
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class ClassificationStats:
    requested: bool = False
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped_by_cap: int = 0
    skipped_by_breaker: int = 0
    circuit_opened: bool = False


@dataclass(frozen=True, slots=True)
class ConsolidatedColumn:
    """Everything known about one column after the deterministic and optional AI steps."""

    column_ref: str
    column_name: str
    findings: tuple[DetectedFinding, ...]
    classification: ClassificationResult | None


@dataclass(frozen=True, slots=True)
class ColumnAssessment:
    """Deterministic risk and review policy for one column. The AI label can raise attention but
    never lowers what the deterministic findings established."""

    column_ref: str
    column_name: str
    risk: RiskLevel
    needs_review: bool
    finding_categories: tuple[str, ...]
    label: ClassificationLabel | None  # only for a usable (ok) suggestion
    review_recommendation: ReviewRecommendation | None  # only for a usable suggestion
    classification_status: ResultStatus | None  # None when classification was not attempted


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    status: WorkflowStatus
    profile: DatasetProfileResult | None
    findings: tuple[DetectedFinding, ...]
    classifications: tuple[ClassificationResult, ...]
    assessments: tuple[ColumnAssessment, ...]
    dataset_risk: RiskLevel | None
    stats: ClassificationStats
    issues: tuple[WorkflowIssue, ...]
    steps: tuple[str, ...]
    durations_ms: dict[str, float]

    def to_summary(self) -> dict[str, Any]:
        """Counts, statuses and timings only: safe to store in ScanRun.summary."""
        column_risk = {level.value: 0 for level in RiskLevel}
        for assessment in self.assessments:
            column_risk[assessment.risk.value] += 1
        provider = self.classifications[0].provider_name if self.classifications else None
        return {
            "status": self.status.value,
            "steps": list(self.steps),
            "classification": {
                "requested": self.stats.requested,
                "attempted": self.stats.attempted,
                "succeeded": self.stats.succeeded,
                "failed": self.stats.failed,
                "skipped_by_cap": self.stats.skipped_by_cap,
                "skipped_by_breaker": self.stats.skipped_by_breaker,
                "circuit_opened": self.stats.circuit_opened,
                "provider": provider,
            },
            "risk": {
                "dataset": self.dataset_risk.value if self.dataset_risk is not None else None,
                "columns": column_risk,
                "needs_review_columns": sum(1 for a in self.assessments if a.needs_review),
            },
            "issues": [
                {"step": issue.step, "code": issue.code, "error_type": issue.error_type}
                for issue in self.issues
            ],
            "durations_ms": dict(self.durations_ms),
        }


# --- Graph state -----------------------------------------------------------------------


class ScanState(TypedDict, total=False):
    """Typed graph state. No raw rows and no DataFrame. List fields with a reducer are appended to
    by each step; every other field is replaced. Node names differ from state keys on purpose."""

    failed: bool
    steps: Annotated[list[str], operator.add]
    issues: Annotated[list[WorkflowIssue], operator.add]
    timings: Annotated[list[tuple[str, float]], operator.add]
    dataset_profile: DatasetProfileResult
    findings: list[DetectedFinding]
    classifications: list[ClassificationResult]
    stats: ClassificationStats
    consolidated: list[ConsolidatedColumn]
    assessments: list[ColumnAssessment]
    dataset_risk: RiskLevel
    result: WorkflowResult


# --- Deterministic policy (pure, independently testable) ----------------------------------


def assess_column(view: ConsolidatedColumn) -> ColumnAssessment:
    """Risk is the highest finding severity, raised to at least MEDIUM by a usable personal or
    financial suggestion. A column needs review when it has a HIGH finding, an email or phone
    pattern finding, or a usable suggestion that asks for review."""
    result = view.classification
    usable = result is not None and result.status is ResultStatus.OK
    label = result.label if (usable and result is not None) else None
    recommendation = result.review_recommendation if (usable and result is not None) else None

    risk = _highest(_SEVERITY_RISK[finding.severity] for finding in view.findings)
    if label in _SENSITIVE_LABELS:
        risk = _highest([risk, RiskLevel.MEDIUM])

    needs_review = (
        any(finding.severity is Severity.HIGH for finding in view.findings)
        or any(finding.category in _PERSONAL_PATTERN_CATEGORIES for finding in view.findings)
        or recommendation is ReviewRecommendation.REQUIRED
    )
    return ColumnAssessment(
        column_ref=view.column_ref,
        column_name=view.column_name,
        risk=risk,
        needs_review=needs_review,
        finding_categories=tuple(sorted({finding.category.value for finding in view.findings})),
        label=label,
        review_recommendation=recommendation,
        classification_status=result.status if result is not None else None,
    )


# --- Graph construction ---------------------------------------------------------------------


def _guarded(
    step: str,
    work: Callable[[ScanState], dict[str, Any]],
    *,
    fatal: bool,
) -> Callable[[ScanState], dict[str, Any]]:
    """Wrap a step: time it, and turn an exception into a recorded issue (never a crash).
    Only the exception class name is logged and stored."""

    def node(state: ScanState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            update = dict(work(state))
        except Exception as error:
            logger.error("Workflow step failed: step=%s error_type=%s", step, type(error).__name__)
            update = {"issues": [WorkflowIssue(step, "step_failed", type(error).__name__)]}
            if fatal:
                update["failed"] = True
        update["steps"] = [step]
        update["timings"] = [(step, (time.perf_counter() - started) * 1000.0)]
        return update

    return node


def _continue_to(next_step: str) -> Callable[[ScanState], str]:
    """Router: go to the next step, or straight to finalize once a fatal error was recorded."""

    def route(state: ScanState) -> str:
        return STEP_FINALIZE if state.get("failed") else next_step

    return route


def build_scan_graph(
    df: pd.DataFrame,
    *,
    options: WorkflowOptions,
    classifier: ColumnClassifier | None,
) -> Any:
    """Compile the workflow for one DataFrame. The DataFrame lives in the node closures only."""
    if options.classify and classifier is None:
        raise ValueError("classify=True requires a classifier.")

    def validate_input(state: ScanState) -> dict[str, Any]:
        if not df.columns.is_unique:
            return {"failed": True, "issues": [WorkflowIssue(STEP_VALIDATE, "invalid_input")]}
        return {}

    def profile_dataset(state: ScanState) -> dict[str, Any]:
        return {"dataset_profile": profile_dataframe(df)}

    def run_checks(state: ScanState) -> dict[str, Any]:
        return {"findings": run_detectors(df, options.detection_config)}

    def classify_columns(state: ScanState) -> dict[str, Any]:
        if classifier is None:  # cannot happen: the node is only added when a classifier exists
            raise RuntimeError("No classifier configured.")
        profile = state["dataset_profile"]
        findings = state.get("findings", [])
        results: list[ClassificationResult] = []
        attempted = succeeded = failed = streak = skipped_by_cap = skipped_by_breaker = 0
        circuit_opened = False

        for position, column in enumerate(profile.columns, start=1):
            if position > options.max_classified_columns:
                skipped_by_cap += 1
                continue
            if circuit_opened:
                skipped_by_breaker += 1
                continue
            result = classifier.classify_column(column, findings, column_ref=f"col_{position}")
            results.append(result)
            attempted += 1
            if result.status is ResultStatus.OK:
                succeeded += 1
                streak = 0
            else:
                failed += 1
                streak += 1
                if streak >= options.max_consecutive_failures:
                    circuit_opened = True

        issues: list[WorkflowIssue] = []
        if failed:
            issues.append(WorkflowIssue(STEP_CLASSIFY, "classification_failed_columns"))
        if skipped_by_cap:
            issues.append(WorkflowIssue(STEP_CLASSIFY, "classification_skipped_by_cap"))
        if skipped_by_breaker:
            issues.append(WorkflowIssue(STEP_CLASSIFY, "classification_circuit_open"))
        return {
            "classifications": results,
            "stats": ClassificationStats(
                requested=True,
                attempted=attempted,
                succeeded=succeeded,
                failed=failed,
                skipped_by_cap=skipped_by_cap,
                skipped_by_breaker=skipped_by_breaker,
                circuit_opened=circuit_opened,
            ),
            "issues": issues,
        }

    def consolidate(state: ScanState) -> dict[str, Any]:
        profile = state["dataset_profile"]
        findings = state.get("findings", [])
        by_ref = {result.column_ref: result for result in state.get("classifications", [])}
        views = []
        for position, column in enumerate(profile.columns, start=1):
            ref = f"col_{position}"
            views.append(
                ConsolidatedColumn(
                    column_ref=ref,
                    column_name=column.column_name,
                    findings=tuple(f for f in findings if f.column_name == column.column_name),
                    classification=by_ref.get(ref),
                )
            )
        return {"consolidated": views}

    def apply_policy(state: ScanState) -> dict[str, Any]:
        views = state.get("consolidated")
        if views is None:  # consolidation already failed and was recorded
            return {}
        assessments = [assess_column(view) for view in views]
        findings = state.get("findings", [])
        dataset_risk = _highest(
            [*(a.risk for a in assessments), *(_SEVERITY_RISK[f.severity] for f in findings)]
        )
        return {"assessments": assessments, "dataset_risk": dataset_risk}

    def finalize(state: ScanState) -> dict[str, Any]:
        issues = tuple(state.get("issues", []))
        if state.get("failed"):
            status = WorkflowStatus.FAILED
        elif issues:
            status = WorkflowStatus.PARTIALLY_COMPLETED
        else:
            status = WorkflowStatus.COMPLETED
        durations = {step: round(ms, 2) for step, ms in state.get("timings", [])}
        result = WorkflowResult(
            status=status,
            profile=state.get("dataset_profile"),
            findings=tuple(state.get("findings", [])),
            classifications=tuple(state.get("classifications", [])),
            assessments=tuple(state.get("assessments", [])),
            dataset_risk=state.get("dataset_risk"),
            stats=state.get("stats", ClassificationStats(requested=options.classify)),
            issues=issues,
            steps=(*state.get("steps", []), STEP_FINALIZE),
            durations_ms=durations,
        )
        return {"result": result}

    builder = StateGraph(ScanState)
    builder.add_node(STEP_VALIDATE, _guarded(STEP_VALIDATE, validate_input, fatal=True))
    builder.add_node(STEP_PROFILE, _guarded(STEP_PROFILE, profile_dataset, fatal=True))
    builder.add_node(STEP_CHECKS, _guarded(STEP_CHECKS, run_checks, fatal=True))
    builder.add_node(STEP_CONSOLIDATE, _guarded(STEP_CONSOLIDATE, consolidate, fatal=False))
    builder.add_node(STEP_POLICY, _guarded(STEP_POLICY, apply_policy, fatal=False))
    builder.add_node(STEP_FINALIZE, _guarded(STEP_FINALIZE, finalize, fatal=True))

    after_checks = STEP_CLASSIFY if options.classify else STEP_CONSOLIDATE
    builder.add_edge(START, STEP_VALIDATE)
    builder.add_conditional_edges(
        STEP_VALIDATE, _continue_to(STEP_PROFILE), {STEP_PROFILE: STEP_PROFILE, STEP_FINALIZE: STEP_FINALIZE}
    )
    builder.add_conditional_edges(
        STEP_PROFILE, _continue_to(STEP_CHECKS), {STEP_CHECKS: STEP_CHECKS, STEP_FINALIZE: STEP_FINALIZE}
    )
    builder.add_conditional_edges(
        STEP_CHECKS, _continue_to(after_checks), {after_checks: after_checks, STEP_FINALIZE: STEP_FINALIZE}
    )
    if options.classify:
        builder.add_node(STEP_CLASSIFY, _guarded(STEP_CLASSIFY, classify_columns, fatal=False))
        builder.add_edge(STEP_CLASSIFY, STEP_CONSOLIDATE)
    builder.add_edge(STEP_CONSOLIDATE, STEP_POLICY)
    builder.add_edge(STEP_POLICY, STEP_FINALIZE)
    builder.add_edge(STEP_FINALIZE, END)
    return builder.compile()


# --- Entry point ------------------------------------------------------------------------------


def _failed_result(
    step: str, code: str, error_type: str | None = None, *, requested: bool = False
) -> WorkflowResult:
    return WorkflowResult(
        status=WorkflowStatus.FAILED,
        profile=None,
        findings=(),
        classifications=(),
        assessments=(),
        dataset_risk=None,
        stats=ClassificationStats(requested=requested),
        issues=(WorkflowIssue(step, code, error_type),),
        steps=(),
        durations_ms={},
    )


def run_scan_workflow(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    options: WorkflowOptions | None = None,
    classifier: ColumnClassifier | None = None,
) -> WorkflowResult:
    """Run the workflow once. Never raises for problems with the data or the steps: those come
    back as a FAILED or PARTIALLY_COMPLETED result. It raises TypeError or ValueError only for
    programming errors (wrong argument types, or classify=True without a classifier)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("run_scan_workflow expects a pandas DataFrame.")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    active = options if options is not None else WorkflowOptions()

    graph = build_scan_graph(df, options=active, classifier=classifier)
    initial: ScanState = {
        "failed": False,
        "steps": [],
        "issues": [],
        "timings": [],
        "findings": [],
        "classifications": [],
    }
    logger.info(
        "Workflow started: rows=%d columns=%d classify=%s", len(df), df.shape[1], active.classify
    )
    try:
        final = graph.invoke(initial, config={"recursion_limit": RECURSION_LIMIT})
    except GraphRecursionError:
        logger.error("Workflow stopped: step limit reached")
        return _failed_result("workflow", "recursion_limit", requested=active.classify)
    except Exception as error:
        logger.error("Workflow error: error_type=%s", type(error).__name__)
        return _failed_result("workflow", "workflow_error", type(error).__name__, requested=active.classify)

    result = final.get("result") if isinstance(final, dict) else None
    if not isinstance(result, WorkflowResult):
        return _failed_result("workflow", "no_result", requested=active.classify)

    logger.info(
        "Workflow finished: status=%s steps=%d issues=%d classified=%d",
        result.status.value,
        len(result.steps),
        len(result.issues),
        len(result.classifications),
    )
    return result