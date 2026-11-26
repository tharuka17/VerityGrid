"""Read-only access to the latest evaluation report written by the evaluation command.

The API never runs an evaluation: a web request must not start a batch job, and nothing here can
trigger a provider call. It only serves the report file that `python -m app.evaluation.harness`
wrote, after checking that the file is safe to read and looks like a report. A missing or unusable
file is a normal "not available" answer with a reason, not an error. File contents are never
echoed, and the path is fixed (never taken from a request).
"""

import json
import logging
import stat as stat_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Relative to the working directory (the project folder when the API is started as documented).
# Must equal the harness's default output: a test compares the two.
DEFAULT_REPORT_PATH = Path("reports") / "evaluation_report.json"
MAX_REPORT_BYTES = 5 * 1024 * 1024
SUPPORTED_REPORT_VERSION = 1
REQUIRED_KEYS = frozenset(
    {
        "report_version",
        "run",
        "dataset",
        "deterministic",
        "mock",
        "llm",
        "human_review",
        "limitations",
        "how_to_read",
        "fingerprints",
    }
)
REGENERATE_HINT = "Run `python -m app.evaluation.harness --skip-review` from the project folder to generate it."


class EvaluationReportOut(BaseModel):
    """The answer of GET /evaluation/report. `report` is the generated report, unchanged."""

    model_config = ConfigDict(frozen=True)

    available: bool
    reason: str | None
    file_modified_at: datetime | None
    report: dict[str, Any] | None


def _unavailable(reason: str) -> EvaluationReportOut:
    return EvaluationReportOut(available=False, reason=reason, file_modified_at=None, report=None)


def _reject_constant(name: str) -> Any:
    raise ValueError("NaN and Infinity are not valid JSON")


def load_report(path: Path = DEFAULT_REPORT_PATH) -> EvaluationReportOut:
    """Return the report if the file is usable, otherwise a reason it is not. Never raises for
    problems with the file, and never writes."""
    try:
        info = path.stat()
    except FileNotFoundError:
        return _unavailable(f"No evaluation report has been generated yet. {REGENERATE_HINT}")
    except OSError as error:
        logger.warning("Evaluation report not accessible: error_type=%s", type(error).__name__)
        return _unavailable("The evaluation report file could not be accessed.")
    if not stat_module.S_ISREG(info.st_mode):
        return _unavailable("The evaluation report path is not a regular file.")
    if info.st_size > MAX_REPORT_BYTES:
        return _unavailable("The evaluation report file is larger than the API will read.")

    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except OSError as error:
        logger.warning("Evaluation report not readable: error_type=%s", type(error).__name__)
        return _unavailable("The evaluation report file could not be read.")
    except UnicodeDecodeError:
        return _unavailable("The evaluation report file is not valid UTF-8 text.")
    except (ValueError, RecursionError):
        return _unavailable("The evaluation report file does not contain valid JSON.")

    version = data.get("report_version") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or type(version) is not int
        or version != SUPPORTED_REPORT_VERSION
        or not REQUIRED_KEYS.issubset(data)
    ):
        return _unavailable(f"The file is not a supported evaluation report. {REGENERATE_HINT}")

    return EvaluationReportOut(
        available=True,
        reason=None,
        file_modified_at=datetime.fromtimestamp(info.st_mtime, tz=timezone.utc),
        report=data,
    )