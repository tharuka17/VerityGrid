from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.middleware import UploadSizeLimitMiddleware
from app.api.routes import datasets, evaluation, findings, metrics, scans
from app.api.access import ReadOnlyModeMiddleware, announce_mode, check_demo_database
from app.core.config import get_settings
from app.db.session import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Create any missing tables at startup (development convenience).

    Idempotent, and it never alters existing tables. Deployments are expected to replace
    this with explicit setup (Phase 12/13). Tests that must not touch the real database
    should not enter this lifespan (use TestClient(app) without a `with` block).
    """
    announce_mode()
    check_demo_database(settings.database_url)
    init_db()
    yield


# `debug` is deliberately not passed: Starlette's debug mode replaces the catch-all error
# handler with an HTML traceback page that can contain exception text.
app = FastAPI(
    title=settings.app_name,
    description=(
        "AI-Assisted Data Governance & Reliability Workbench. "
        "Independent portfolio prototype using synthetic data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(UploadSizeLimitMiddleware)
app.add_middleware(ReadOnlyModeMiddleware)
register_exception_handlers(app)

for router in (scans.router, datasets.router, findings.router, metrics.router):
    app.include_router(router)
    
app.include_router(evaluation.router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }