"""Create a pre-seeded demo database from the synthetic sample files (for a read-only public deployment).

Usage, from the project root:
    python -m scripts.seed_demo            # creates veritygrid_demo.db from data/sample/
    python -m scripts.seed_demo --reset    # recreate it first (only a SQLite file whose name contains "demo")

What it does: scans the three synthetic CSV files in data/sample/ with the deterministic checks (no AI provider
is ever called) and records two clearly labelled review decisions, so the review views have something to show.
It uses the same services as the API (run_scan, review_finding), so the data is exactly what the application
would store. It writes no row values. It refuses to add to a database that already contains datasets, and
--reset only deletes a SQLite file whose name contains "demo", so the development database cannot be removed
by mistake.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.db.models import Dataset, Finding
from app.db.session import build_engine, init_db
from app.schemas.api import ReviewRequest
from app.schemas.rules import check_rule_columns, parse_rules
from app.services.ingestion import ingest_csv
from app.services.review_service import review_finding
from app.services.scan_service import run_scan

SAMPLE_DIR = Path("data") / "sample"
SAMPLES = (
    ("customers.csv", "customers_sample"),
    ("orders.csv", "orders_sample"),
    ("inventory.csv", "inventory_sample"),
)
# Detection rules for a sample, written exactly as a user would type them in the dashboard. They are validated
# by the same code as an upload, so the demo exercises numeric and date rules too.
SAMPLE_RULES: dict[str, str] = {
    "inventory.csv": (
        '{"numeric_rules": {"quantity_on_hand": {"min_value": 0}, "reorder_point": {"min_value": 0}}, '
        '"date_rules": {"last_counted_date": {"date_format": "%Y-%m-%d"}}}'
    ),
}
# (finding category, decision, comment): the first finding of each category gets the decision.
REVIEW_PLAN = (
    ("potential_email", "approved", "Seeded example: the column holds contact data, so the finding is accepted."),
    ("duplicate_rows", "rejected", "Seeded example: repeated rows are expected in this synthetic sample."),
)
REVIEWER = "demo-seed"
DEFAULT_DATABASE_URL = "sqlite:///./veritygrid_demo.db"


class SeedError(Exception):
    """A problem with the target database or the sample files. Messages never contain data."""


def sqlite_file(database_url: str) -> Path | None:
    """The file a SQLite URL points to, or None for in-memory and non-SQLite databases."""
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
        return Path(url.database)
    return None


def prepare_target(database_url: str, *, reset: bool) -> None:
    """Apply the safety rules before anything is written."""
    path = sqlite_file(database_url)
    if not reset:
        return
    if path is None:
        raise SeedError("--reset only works for a SQLite database file.")
    if "demo" not in path.name.lower():
        raise SeedError("--reset only deletes a database file whose name contains 'demo'.")
    if path.exists():
        path.unlink()


def seed_into(db: Session, sample_dir: Path = SAMPLE_DIR) -> dict[str, int]:
    """Scan the sample files and record the example reviews. Never calls an AI provider."""
    if (db.scalar(select(func.count()).select_from(Dataset)) or 0) > 0:
        raise SeedError("The database already contains datasets, so the seed will not add to it.")

    for filename, name in SAMPLES:
        path = sample_dir / filename
        if not path.is_file():
            raise SeedError(f"Sample file not found: {filename}")
        frame = ingest_csv(path.read_bytes()).dataframe
        config = parse_rules(SAMPLE_RULES.get(filename))
        if config is not None:
            check_rule_columns(config, frame.columns)
        run_scan(db, frame, dataset_name=name, source_filename=filename, config=config)  # no classifier

    reviewed = {"approved": 0, "rejected": 0}
    for category, decision, comment in REVIEW_PLAN:
        finding_id = db.scalar(
            select(Finding.id).where(Finding.category == category).order_by(Finding.id).limit(1)
        )
        if finding_id is None:
            continue
        review_finding(db, finding_id, ReviewRequest(decision=decision, comment=comment, reviewer=REVIEWER))
        reviewed[decision] += 1

    total = db.scalar(select(func.count()).select_from(Finding)) or 0
    return {
        "datasets": db.scalar(select(func.count()).select_from(Dataset)) or 0,
        "findings": total,
        "approved": reviewed["approved"],
        "rejected": reviewed["rejected"],
        "pending": total - reviewed["approved"] - reviewed["rejected"],
    }


def seed(
    database_url: str = DEFAULT_DATABASE_URL, *, reset: bool = False, sample_dir: Path = SAMPLE_DIR
) -> dict[str, int]:
    prepare_target(database_url, reset=reset)
    engine = build_engine(database_url)
    try:
        init_db(engine)
        with Session(engine) as db:
            return seed_into(db, sample_dir)
    finally:
        engine.dispose()  # release the file (Windows and OneDrive keep locks on open files)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.seed_demo",
        description="Create a pre-seeded, synthetic-only demo database.",
    )
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL, help="database to create (default: a separate demo file)")
    parser.add_argument("--reset", action="store_true", help="delete an existing demo database file first")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        counts = seed(args.database_url, reset=args.reset)
    except SeedError as error:
        print(f"Seed failed: {error}")
        return 1
    path = sqlite_file(args.database_url)
    print(f"Seeded {path if path is not None else 'the database'} with synthetic data only.")
    print(
        f"  datasets {counts['datasets']}, findings {counts['findings']} "
        f"(approved {counts['approved']}, rejected {counts['rejected']}, pending {counts['pending']})"
    )
    print("  Run the API against it in read-only mode: see the commands in the project notes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())