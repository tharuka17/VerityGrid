import codecs
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.services.ingestion import (
    EmptyFileError,
    FileTooLargeError,
    IngestionError,
    IngestionLimits,
    IngestionResult,
    InvalidEncodingError,
    InvalidHeaderError,
    MalformedCsvError,
    TooManyColumnsError,
    TooManyRowsError,
    ingest_csv,
)
from app.services.profiling import profile_dataframe

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


# --- Valid input -------------------------------------------------------------


def test_valid_csv_returns_dataframe_and_metadata() -> None:
    data = b"id,name,score\n1,Ann,5.5\n2,Bob,7.0\n"

    result = ingest_csv(data)

    assert isinstance(result, IngestionResult)
    assert result.row_count == 2
    assert result.column_count == 3
    assert result.column_names == ("id", "name", "score")
    assert result.size_bytes == len(data)
    assert result.had_bom is False
    assert result.dataframe.shape == (2, 3)
    assert list(result.dataframe.columns) == ["id", "name", "score"]
    assert pd.api.types.is_integer_dtype(result.dataframe["id"])
    assert pd.api.types.is_float_dtype(result.dataframe["score"])


def test_utf8_bom_is_accepted_and_stripped() -> None:
    result = ingest_csv(codecs.BOM_UTF8 + b"id,name\n1,Ann\n")

    assert result.had_bom is True
    assert result.column_names == ("id", "name")
    assert all("\ufeff" not in name for name in result.column_names)


def test_non_ascii_utf8_content() -> None:
    result = ingest_csv("city\nJosé\n".encode("utf-8"))

    assert result.dataframe.iloc[0, 0] == "José"


def test_crlf_line_endings_and_blank_lines() -> None:
    result = ingest_csv(b"a,b\r\n1,2\r\n\r\n3,4\r\n")

    assert result.row_count == 2


def test_quoted_fields_with_commas_and_newlines() -> None:
    result = ingest_csv(b'a,b\n"x, y","line1\nline2"\n')

    assert result.dataframe.shape == (1, 2)


def test_missing_values_become_nulls() -> None:
    result = ingest_csv(b"a,b\n1,\n,2\n")

    assert result.dataframe.isna().sum().tolist() == [1, 1]


def test_header_whitespace_is_stripped() -> None:
    result = ingest_csv(b" id , name \n1,Ann\n")

    assert result.column_names == ("id", "name")


def test_bytearray_is_accepted() -> None:
    assert ingest_csv(bytearray(b"a\n1\n")).row_count == 1


def test_result_feeds_the_profiling_service() -> None:
    result = ingest_csv(b"id,name\n1,Ann\n2,\n")

    profile = profile_dataframe(result.dataframe)

    assert profile.row_count == 2
    assert {c.column_name: c.null_count for c in profile.columns} == {"id": 0, "name": 1}


@pytest.mark.parametrize("name", ["customers", "orders", "inventory"])
def test_sample_files_ingest_cleanly(name: str) -> None:
    path = SAMPLE_DIR / f"{name}.csv"

    result = ingest_csv(path.read_bytes())
    expected = pd.read_csv(path)

    assert result.dataframe.shape == expected.shape
    assert list(result.dataframe.columns) == list(expected.columns)


# --- Limits ------------------------------------------------------------------


def test_default_limits() -> None:
    limits = IngestionLimits()

    assert limits.max_bytes == 5 * 1024 * 1024
    assert limits.max_rows == 50_000
    assert limits.max_columns == 200


@pytest.mark.parametrize(
    "kwargs",
    [{"max_bytes": 0}, {"max_rows": -1}, {"max_columns": 0}, {"max_rows": True}],
    ids=["zero-bytes", "negative-rows", "zero-columns", "boolean-rows"],
)
def test_limits_must_be_positive_integers(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        IngestionLimits(**kwargs)


def test_size_limit_is_inclusive() -> None:
    data = b"a\n1\n"

    assert ingest_csv(data, IngestionLimits(max_bytes=len(data))).row_count == 1
    with pytest.raises(FileTooLargeError):
        ingest_csv(data, IngestionLimits(max_bytes=len(data) - 1))


def test_size_is_checked_before_decoding() -> None:
    with pytest.raises(FileTooLargeError):
        ingest_csv(b"\xff" * 20, IngestionLimits(max_bytes=10))


def test_row_limit() -> None:
    limits = IngestionLimits(max_rows=2)

    assert ingest_csv(b"a\n1\n2\n", limits).row_count == 2
    with pytest.raises(TooManyRowsError):
        ingest_csv(b"a\n1\n2\n3\n", limits)


def test_column_limit() -> None:
    limits = IngestionLimits(max_columns=2)

    assert ingest_csv(b"a,b\n1,2\n", limits).column_count == 2
    with pytest.raises(TooManyColumnsError):
        ingest_csv(b"a,b,c\n1,2,3\n", limits)


# --- Empty files -------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [b"", b"   \n\n  ", codecs.BOM_UTF8],
    ids=["zero-bytes", "whitespace-only", "bom-only"],
)
def test_empty_inputs_are_rejected(data: bytes) -> None:
    with pytest.raises(EmptyFileError):
        ingest_csv(data)


@pytest.mark.parametrize("data", [b"a,b\n", b"a,b\n\n\n"], ids=["header-only", "header-and-blanks"])
def test_header_without_data_rows_is_rejected(data: bytes) -> None:
    with pytest.raises(EmptyFileError, match="no data rows"):
        ingest_csv(data)


# --- Encoding ----------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b"name\ncaf\xe9\n",
        "a,b\n1,2\n".encode("utf-16"),
        b"a,b\n1,\x002\n",
    ],
    ids=["latin-1", "utf-16", "nul-byte"],
)
def test_invalid_encodings_and_binary_data_are_rejected(data: bytes) -> None:
    with pytest.raises(InvalidEncodingError):
        ingest_csv(data)


# --- Malformed CSV -----------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b'a,b\n1,"unterminated\n',
        b"a,b\n1,2,3\n",
        b"a,b\n1\n",
        b'a,b\n1,"x"y\n',
    ],
    ids=["unclosed-quote", "too-many-fields", "too-few-fields", "text-after-quote"],
)
def test_malformed_csv_is_rejected(data: bytes) -> None:
    with pytest.raises(MalformedCsvError):
        ingest_csv(data)


# --- Header rules ------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b"a,,c\n1,2,3\n",
        b"a,b,\n1,2,3\n",
        b"a,a\n1,2\n",
        b"id,ID\n1,2\n",
        b" a ,a\n1,2\n",
        b"a,b\x07c\n1,2\n",
        ("a," + "x" * 256 + "\n1,2\n").encode(),
    ],
    ids=[
        "empty-header",
        "trailing-comma",
        "exact-duplicate",
        "case-insensitive-duplicate",
        "duplicate-after-strip",
        "control-character",
        "header-too-long",
    ],
)
def test_invalid_headers_are_rejected(data: bytes) -> None:
    with pytest.raises(InvalidHeaderError):
        ingest_csv(data)


# --- Safety ------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b"SECRET-HEADER,SECRET-HEADER\nSECRET-VALUE,x\n",
        b"a,b\nSECRET-VALUE,1,2\n",
        b'a,b\nSECRET-VALUE,"unclosed\n',
        b"a,SECRET-HEADER\nSECRET-VALUE,caf\xe9\n",
    ],
    ids=["duplicate-header", "wrong-field-count", "unclosed-quote", "bad-encoding"],
)
def test_error_messages_do_not_echo_uploaded_content(data: bytes) -> None:
    with pytest.raises(IngestionError) as excinfo:
        ingest_csv(data)

    assert "SECRET" not in str(excinfo.value)


def test_exception_hierarchy_and_codes() -> None:
    classes = [
        FileTooLargeError,
        EmptyFileError,
        InvalidEncodingError,
        MalformedCsvError,
        InvalidHeaderError,
        TooManyRowsError,
        TooManyColumnsError,
    ]

    assert all(issubclass(cls, IngestionError) for cls in classes)
    codes = [cls.code for cls in classes]
    assert len(set(codes)) == len(codes)


@pytest.mark.parametrize(
    "value",
    ["data/sample/customers.csv", Path("data/sample/customers.csv")],
    ids=["string-path", "path-object"],
)
def test_paths_and_text_are_rejected(value: Any) -> None:
    with pytest.raises(TypeError):
        ingest_csv(value)


def test_path_like_bytes_are_treated_as_content_not_opened() -> None:
    """Bytes that look like a path are parsed as CSV text (a header with no rows)."""
    with pytest.raises(EmptyFileError):
        ingest_csv(b"data/sample/customers.csv")


def test_result_repr_does_not_contain_cell_values() -> None:
    result = ingest_csv(b"a,b\nSECRET-VALUE,1\n")

    assert "SECRET" not in repr(result)
    assert "SECRET" not in str(result)


def test_logging_never_contains_uploaded_values(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="app.services.ingestion")

    ingest_csv(b"a,b\nSECRET-VALUE,1\n")
    with pytest.raises(IngestionError):
        ingest_csv(b"a,b\nSECRET-VALUE,1,2\n")

    assert "SECRET" not in caplog.text
    assert "rows=1" in caplog.text
    assert "code=malformed_csv" in caplog.text


def test_ingestion_writes_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    ingest_csv(b"a,b\n1,2\n")
    with pytest.raises(IngestionError):
        ingest_csv(b"a,a\n1,2\n")

    assert list(tmp_path.iterdir()) == []