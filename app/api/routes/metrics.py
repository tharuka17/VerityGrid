"""Metrics endpoint: counts calculated from persisted records."""

from fastapi import APIRouter

from app.api.deps import DbSession
from app.schemas.api import MetricsSummary
from app.services import query_service

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "/summary",
    response_model=MetricsSummary,
    summary="Summary metrics from stored records",
    description=(
        "Every number is a count of rows currently stored: nothing is estimated or "
        "pre-filled. 'reviewed_finding_percent' is null while there are no findings."
    ),
)
def get_metrics_summary(db: DbSession) -> MetricsSummary:
    return query_service.get_metrics_summary(db)