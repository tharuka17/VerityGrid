"""Dataset endpoints: list datasets, read one, and read its persisted column profiles."""

from fastapi import APIRouter

from app.api.deps import DbSession, IdPath, Pagination, error_responses
from app.schemas.api import ColumnProfilesResponse, DatasetDetail, DatasetOut, Page
from app.services import query_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get(
    "",
    response_model=Page[DatasetOut],
    summary="List datasets (newest first)",
    responses=error_responses(422),
)
def list_datasets(db: DbSession, page: Pagination) -> Page[DatasetOut]:
    return query_service.list_datasets(db, limit=page.limit, offset=page.offset)


@router.get(
    "/{dataset_id}",
    response_model=DatasetDetail,
    summary="Get a dataset summary",
    responses=error_responses(404, 422),
)
def get_dataset(dataset_id: IdPath, db: DbSession) -> DatasetDetail:
    return query_service.get_dataset(db, dataset_id)


@router.get(
    "/{dataset_id}/columns",
    response_model=ColumnProfilesResponse,
    summary="Get a dataset's column profiles",
    description="Counts and inferred types only. No cell values are stored or returned.",
    responses=error_responses(404, 422),
)
def get_dataset_columns(dataset_id: IdPath, db: DbSession) -> ColumnProfilesResponse:
    return query_service.get_column_profiles(db, dataset_id)