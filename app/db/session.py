"""Engine, session factory, FastAPI session dependency and table initialization."""

from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.db.models import Base  # importing models registers every table on Base.metadata


def build_engine(database_url: str) -> Engine:
    """Create an engine for a SQLite or PostgreSQL-style DATABASE_URL.

    Creating an engine does not connect to the database or create any file.
    """
    url = make_url(database_url)

    if url.get_backend_name() != "sqlite":
        # PostgreSQL and others: pre-ping avoids using stale pooled connections.
        return create_engine(database_url, pool_pre_ping=True)

    connect_args: dict[str, Any] = {"check_same_thread": False}
    engine_kwargs: dict[str, Any] = {"connect_args": connect_args}
    if url.database in (None, "", ":memory:"):
        # An in-memory database is per-connection; share one connection across threads.
        engine_kwargs["poolclass"] = StaticPool

    engine = create_engine(database_url, **engine_kwargs)

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection: Any, connection_record: Any) -> None:
        # SQLite ignores foreign keys unless this is switched on for every connection.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


engine = build_engine(get_settings().database_url)

# expire_on_commit=False keeps objects readable after commit (useful for API responses).
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: one session per request.

    Commit behavior: this dependency never commits. Code that changes data must call
    db.commit() explicitly. If the request raises an exception, the session is rolled
    back; it is always closed afterwards.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db(bind: Engine | None = None) -> None:
    """Create any missing tables (development and test convenience).

    Safe to call repeatedly: existing tables are left untouched. This is not a migration
    tool; it will not alter tables whose columns have changed.
    """
    Base.metadata.create_all(bind=bind or engine)