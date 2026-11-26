"""In-memory column profiling for an already parsed pandas DataFrame.

Privacy: this module returns counts, percentages and type labels only. It never stores,
returns or logs cell values or sample records. It performs no database writes, applies no
detection rules and makes no LLM calls.
"""

from dataclasses import dataclass

import pandas as pd
from pandas.api import types as pdt
from enum import StrEnum


class InferredType(StrEnum):
    """Coarse, human-readable type labels.

    EMPTY    no non-null values (a zero-row frame or an all-null column)
    BOOLEAN  True/False values
    INTEGER  whole numbers (see the float rule in _infer_type)
    FLOAT    numbers with fractional parts
    DATETIME datetime values, or text where EVERY non-null value looks like an ISO date
    STRING   text that is not entirely ISO-date-like
    MIXED    several kinds of value in one column, or a kind this profiler does not label
    """

    EMPTY = "empty"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    DATETIME = "datetime"
    STRING = "string"
    MIXED = "mixed"


# ISO-8601 style: 2026-09-01, 2026-09-01 14:30, 2026-09-01T14:30:05.
# This checks the TEXT SHAPE only. "2026-02-30" has the right shape, so it counts as date-like.
_ISO_DATE_PATTERN = r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?"


# Named *Result* so they are not confused with the ORM models in app/db/models.py.
@dataclass(frozen=True, slots=True)
class ColumnProfileResult:
    column_name: str
    pandas_dtype: str  # the dtype pandas observed, for example "int64", "float64", "object", "str"
    inferred_type: InferredType
    row_count: int
    null_count: int
    null_percent: float  # 0.0 to 100.0, rounded to 2 decimals; 0.0 when there are no rows
    distinct_count: int | None  # non-null distinct values; None if the values cannot be hashed


@dataclass(frozen=True, slots=True)
class DatasetProfileResult:
    row_count: int
    column_count: int
    columns: tuple[ColumnProfileResult, ...]


def _infer_type(series: pd.Series) -> InferredType:
    """Infer a coarse type from the non-null values of one column. Deterministic."""
    non_null = series.dropna()
    if non_null.empty:
        return InferredType.EMPTY

    # 1. Trust a numeric, boolean or datetime dtype that pandas already assigned.
    if pdt.is_bool_dtype(non_null):
        return InferredType.BOOLEAN
    if pdt.is_integer_dtype(non_null):
        return InferredType.INTEGER
    if pdt.is_float_dtype(non_null):
        # pandas stores an integer column that has blanks as float64 (blanks become NaN).
        # If every non-null value is a whole number we call it INTEGER, because the float
        # dtype is an artifact of loading, not a property of the data. inf/-inf are not whole.
        return InferredType.INTEGER if non_null.mod(1).eq(0).all() else InferredType.FLOAT
    if pdt.is_datetime64_any_dtype(non_null):
        return InferredType.DATETIME

    # 2. Text or object columns: ask pandas what kinds of Python values are really inside.
    kind = pdt.infer_dtype(non_null, skipna=True)
    if kind == "boolean":
        return InferredType.BOOLEAN
    if kind == "integer":
        return InferredType.INTEGER
    if kind in ("floating", "mixed-integer-float", "decimal"):
        return InferredType.FLOAT
    if kind in ("datetime", "datetime64", "date"):
        return InferredType.DATETIME
    if kind == "string":
        # ALL-or-nothing: one value that is not ISO-date-like keeps the column STRING.
        if bool(non_null.str.fullmatch(_ISO_DATE_PATTERN).all()):
            return InferredType.DATETIME
        return InferredType.STRING

    return InferredType.MIXED  # mixed-integer, mixed, timedelta, categorical, and so on


def _distinct_count(series: pd.Series) -> int | None:
    """Number of distinct non-null values, or None when values are unhashable."""
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return None


def _profile_column(name: str, series: pd.Series, row_count: int) -> ColumnProfileResult:
    # "Null" means what pandas treats as missing: NaN, None, NaT and pd.NA.
    # Empty or whitespace-only strings are NOT counted as null here.
    null_count = int(series.isna().sum())
    null_percent = round(100.0 * null_count / row_count, 2) if row_count else 0.0

    return ColumnProfileResult(
        column_name=name,
        pandas_dtype=str(series.dtype),
        inferred_type=_infer_type(series),
        row_count=row_count,
        null_count=null_count,
        null_percent=null_percent,
        distinct_count=_distinct_count(series),
    )


def profile_dataframe(df: pd.DataFrame) -> DatasetProfileResult:
    """Profile every column of a DataFrame without keeping or returning any cell values.

    Columns are read by position, so duplicate column names are handled safely.
    A DataFrame with no rows or no columns is valid and returns zero counts.
    """
    row_count = len(df)
    columns = tuple(
        _profile_column(str(df.columns[position]), df.iloc[:, position], row_count)
        for position in range(df.shape[1])
    )
    return DatasetProfileResult(
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
    )