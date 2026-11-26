"""Evaluation endpoint: serves the latest generated evaluation report (read-only)."""

from pathlib import Path as FilePath
from typing import Annotated

from fastapi import APIRouter, Depends

from app.services import evaluation_service
from app.services.evaluation_service import EvaluationReportOut

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def get_report_path() -> FilePath:
    """The report file the evaluation command writes. A dependency so tests can point it elsewhere;
    the path is never taken from the request."""
    return evaluation_service.DEFAULT_REPORT_PATH


@router.get(
    "/report",
    response_model=EvaluationReportOut,
    summary="Get the latest generated evaluation report",
    description=(
        "Serves the report written by `python -m app.evaluation.harness`. It never runs an evaluation "
        "and never calls a model provider. When no usable report exists the answer is still HTTP 200 "
        "with `available: false` and a reason. The report contains metrics, counts and column names "
        "of the synthetic evaluation dataset only, never cell values."
    ),
)
def get_evaluation_report(path: Annotated[FilePath, Depends(get_report_path)]) -> EvaluationReportOut:
    return evaluation_service.load_report(path)