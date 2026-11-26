"""Safe, in-memory CSV ingestion.

Input is raw bytes, never a filesystem path. The content is decoded and parsed as data only:
nothing in it is evaluated or executed, and nothing is written to disk. The service has no
dependency on FastAPI, so it can be tested directly.

Validation order (cheapest and safest checks first):
  1. type check (bytes only)         4. decode as UTF-8 / UTF-8 with BOM
  2. byte-size limit                 5. structure check with the csv module (header rules,
  3. empty / NUL-byte checks            field counts, row and column limits)
                                     6. type-inferring parse with pandas, verified against step 5

Privacy: error messages and log lines contain counts, positions and limits only. They never
contain cell values or header text. IngestionResult.dataframe is excluded from repr().

HTTP mapping (a later phase): FileTooLargeError -> 413; other IngestionError subclasses -> 422.
"""

import codecs
import csv
import io
import logging
import unicodedata
import warnings
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB
DEFAULT_MAX_ROWS = 50_000
DEFAULT_MAX_COLUMNS = 200
MAX_HEADER_LENGTH = 255  # matches the column_name length in the database model


# --- Exceptions --------------------------------------------------------------


class IngestionError(Exception):
    """Base class for all expected ingestion failures. `code` is stable and machine-readable."""

    code: str = "ingestion_error"


class FileTooLargeError(IngestionError):
    code = "file_too_large"


class EmptyFileError(IngestionError):
    code = "empty_file"


class InvalidEncodingError(IngestionError):
    code = "invalid_encoding"


class MalformedCsvError(IngestionError):
    code = "malformed_csv"


class InvalidHeaderError(IngestionError):
    code = "invalid_header"


class TooManyRowsError(IngestionError):
    code = "too_many_rows"


class TooManyColumnsError(IngestionError):
    code = "too_many_columns"


# --- Configuration and result types ------------------------------------------


@dataclass(frozen=True, slots=True)
class IngestionLimits:
    """Upper bounds for one ingestion. All values must be positive integers."""

    max_bytes: int = DEFAULT_MAX_BYTES
    max_rows: int = DEFAULT_MAX_ROWS
    max_columns: int = DEFAULT_MAX_COLUMNS

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_rows", "max_columns"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True, eq=False)
class IngestionResult:
    """A parsed DataFrame plus safe metadata.

    The DataFrame holds raw uploaded values, so it is excluded from repr() to keep it out
    of logs and tracebacks. column_names come from the (untrusted) file header.
    """

    dataframe: pd.DataFrame = field(repr=False)
    row_count: int
    column_count: int
    column_names: tuple[str, ...]
    size_bytes: int
    had_bom: bool


# --- Header validation -------------------------------------------------------


def _clean_header(raw_header: list[str]) -> list[str]:
    """Strip whitespace and reject headers that cannot be handled safely.

    Rejected: empty names, names over MAX_HEADER_LENGTH, names with control or invisible
    characters, and duplicates (compared ignoring case, after stripping). Duplicates are
    rejected rather than renamed because pandas would silently rename them ("a" -> "a.1").
    Messages give column positions, never header text.
    """
    cleaned: list[str] = []
    first_position: dict[str, int] = {}
    for position, raw_name in enumerate(raw_header, start=1):
        name = raw_name.strip()
        if not name:
            raise InvalidHeaderError(f"Column {position} has an empty header.")
        if len(name) > MAX_HEADER_LENGTH:
            raise InvalidHeaderError(
                f"The header of column {position} is longer than {MAX_HEADER_LENGTH} characters."
            )
        if any(unicodedata.category(char).startswith("C") for char in name):
            raise InvalidHeaderError(
                f"The header of column {position} contains control or invisible characters."
            )
        key = name.casefold()
        if key in first_position:
            raise InvalidHeaderError(
                f"Columns {first_position[key]} and {position} have the same header "
                "(comparison ignores case and surrounding spaces)."
            )
        first_position[key] = position
        cleaned.append(name)
    return cleaned


# --- Structure validation ----------------------------------------------------


def _validate_structure(text: str, limits: IngestionLimits) -> tuple[list[str], int]:
    """Check the header, every row's field count and the limits. Returns (header, row count).

    pandas is more forgiving than we want: it pads short rows with nulls and renames
    duplicate headers. This explicit pass makes ragged rows an error instead. Blank lines
    are skipped, matching pandas.
    """
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header: list[str] | None = None
    row_count = 0
    try:
        for record in reader:
            if not record:  # blank line
                continue
            if header is None:
                if len(record) > limits.max_columns:
                    raise TooManyColumnsError(
                        f"The file has {len(record)} columns, which exceeds the limit of "
                        f"{limits.max_columns}."
                    )
                header = _clean_header(record)
                continue
            if len(record) != len(header):
                raise MalformedCsvError(
                    f"Line {reader.line_num} has {len(record)} field(s) but the header "
                    f"defines {len(header)}."
                )
            row_count += 1
            if row_count > limits.max_rows:
                raise TooManyRowsError(
                    f"The file has more than {limits.max_rows} data rows, which exceeds the limit."
                )
    except csv.Error:
        # Deliberately generic and unchained: the original message could mention content.
        raise MalformedCsvError(
            "The file is not valid CSV (for example an unclosed quotation mark, stray "
            "characters after a quoted value, or a field that is too large)."
        ) from None

    if header is None:
        raise EmptyFileError("The file contains no data.")
    return header, row_count


# --- Parsing -----------------------------------------------------------------


def _parse_with_pandas(text: str, header: list[str], row_count: int) -> pd.DataFrame:
    """Parse with pandas for type inference, then verify it agrees with the structure check.

    Default NA handling applies (blank cells and tokens such as "NA" become nulls), the same
    as the profiling samples. nrows is a safety cap: pandas can never return more than one
    row beyond the count already validated.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            df = pd.read_csv(
                io.StringIO(text), header=0, nrows=row_count + 1, low_memory=False
            )
    except (ValueError, pd.errors.ParserWarning):
        raise MalformedCsvError("The file could not be parsed as CSV.") from None

    if df.shape != (row_count, len(header)):
        raise MalformedCsvError("The file could not be read consistently as CSV.")

    df.columns = header  # the validated, whitespace-stripped names
    return df


# --- Public API --------------------------------------------------------------


def _ingest(data: bytes, limits: IngestionLimits) -> IngestionResult:
    size_bytes = len(data)
    if size_bytes > limits.max_bytes:
        raise FileTooLargeError(
            f"The file is {size_bytes} bytes, which exceeds the limit of {limits.max_bytes} bytes."
        )
    if size_bytes == 0:
        raise EmptyFileError("The file is empty.")
    if b"\x00" in data:
        raise InvalidEncodingError(
            "The file contains binary data (NUL bytes) and is not a UTF-8 text CSV file."
        )

    try:
        text = data.decode("utf-8-sig")  # accepts UTF-8 with or without a BOM
    except UnicodeDecodeError:
        raise InvalidEncodingError(
            "The file is not valid UTF-8 text. Save it as UTF-8 (with or without a BOM) "
            "and try again."
        ) from None

    if not text.strip():
        raise EmptyFileError("The file contains no data.")

    header, row_count = _validate_structure(text, limits)
    if row_count == 0:
        raise EmptyFileError("The file has a header but no data rows.")

    df = _parse_with_pandas(text, header, row_count)
    return IngestionResult(
        dataframe=df,
        row_count=row_count,
        column_count=len(header),
        column_names=tuple(header),
        size_bytes=size_bytes,
        had_bom=data.startswith(codecs.BOM_UTF8),
    )


def ingest_csv(data: bytes | bytearray, limits: IngestionLimits | None = None) -> IngestionResult:
    """Validate and parse CSV content.

    `data` must be the CSV content itself. Strings and paths are rejected with TypeError, so
    this function can never be made to open a file. Raises an IngestionError subclass for any
    problem with the content. The original bytes are not stored anywhere.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(
            "ingest_csv expects the CSV content as bytes; file paths and text are not accepted."
        )
    active_limits = limits if limits is not None else IngestionLimits()

    try:
        result = _ingest(bytes(data), active_limits)
    except IngestionError as error:
        logger.info("CSV rejected: code=%s", error.code)
        raise

    logger.info(
        "CSV ingested: rows=%d columns=%d bytes=%d",
        result.row_count,
        result.column_count,
        result.size_bytes,
    )
    return result