import dataclasses
import io
from pathlib import Path

import pandas as pd
import pytest

from app.services.profiling import (
    ColumnProfileResult,
    DatasetProfileResult,
    InferredType,
    profile_dataframe,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def profile_columns(df: pd.DataFrame) -> dict[str, ColumnProfileResult]:
    """Profile a DataFrame and index the column results by name."""
    return {column.column_name: column for column in profile_dataframe(df).columns}


# --- Ordinary data ---------------------------------------------------------


def test_ordinary_dataframe_with_mixed_column_types() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "price": [1.5, 2.25, 3.0, 4.75],
            "name": ["a", "b", "c", "d"],
            "is_active": [True, False, True, True],
            "created_at": pd.to_datetime(
                ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]
            ),
        }
    )

    profile = profile_dataframe(df)

    assert isinstance(profile, DatasetProfileResult)
    assert profile.row_count == 4
    assert profile.column_count == 5
    assert [c.column_name for c in profile.columns] == [
        "id",
        "price",
        "name",
        "is_active",
        "created_at",
    ]

    columns = {c.column_name: c for c in profile.columns}
    assert columns["id"].inferred_type == InferredType.INTEGER
    assert columns["price"].inferred_type == InferredType.FLOAT
    assert columns["name"].inferred_type == InferredType.STRING
    assert columns["is_active"].inferred_type == InferredType.BOOLEAN
    assert columns["created_at"].inferred_type == InferredType.DATETIME

    assert columns["price"].pandas_dtype == "float64"
    assert {c.row_count for c in profile.columns} == {4}
    assert {c.null_count for c in profile.columns} == {0}
    assert {c.null_percent for c in profile.columns} == {0.0}
    assert columns["id"].distinct_count == 4
    assert columns["is_active"].distinct_count == 2


# --- Duplicates ------------------------------------------------------------


def test_duplicate_values_reduce_distinct_count() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "category": ["a", "a", "b", "b", "b"],
            "constant": ["x", "x", "x", "x", "x"],
        }
    )

    columns = profile_columns(df)

    assert columns["id"].distinct_count == 5
    assert columns["category"].distinct_count == 2
    assert columns["constant"].distinct_count == 1
    assert columns["category"].row_count == 5


def test_fully_duplicated_rows_still_count_as_rows() -> None:
    df = pd.DataFrame({"sku": ["A", "B", "A"], "qty": [1, 2, 1]})

    profile = profile_dataframe(df)
    columns = {c.column_name: c for c in profile.columns}

    assert profile.row_count == 3
    assert columns["sku"].distinct_count == 2
    assert columns["qty"].distinct_count == 2


def test_duplicate_column_names_are_profiled_by_position() -> None:
    df = pd.DataFrame([[1, "a"], [1, "b"]], columns=["x", "x"])

    profile = profile_dataframe(df)

    assert profile.column_count == 2
    first, second = profile.columns
    assert (first.column_name, second.column_name) == ("x", "x")
    assert first.inferred_type == InferredType.INTEGER
    assert first.distinct_count == 1
    assert second.inferred_type == InferredType.STRING
    assert second.distinct_count == 2


def test_non_string_column_names_are_converted_to_strings() -> None:
    df = pd.DataFrame([[1, 2], [3, 4]], columns=[0, 1])

    profile = profile_dataframe(df)

    assert [c.column_name for c in profile.columns] == ["0", "1"]


# --- Nulls and blanks ------------------------------------------------------


def test_null_count_and_percent() -> None:
    df = pd.DataFrame({"a": [1.5, None, 3.5], "b": [1, None, None]})

    columns = profile_columns(df)

    assert columns["a"].null_count == 1
    assert columns["a"].null_percent == pytest.approx(33.33)
    assert columns["a"].distinct_count == 2  # nulls are not counted as a distinct value
    assert columns["a"].inferred_type == InferredType.FLOAT

    assert columns["b"].null_count == 2
    assert columns["b"].null_percent == pytest.approx(66.67)
    assert columns["b"].distinct_count == 1
    assert columns["b"].inferred_type == InferredType.INTEGER  # whole-number floats


def test_blank_and_whitespace_strings_are_not_null() -> None:
    df = pd.DataFrame({"text": ["a", "", " ", None]})

    column = profile_columns(df)["text"]

    assert column.null_count == 1  # only None; "" and " " are real (if empty) strings
    assert column.null_percent == pytest.approx(25.0)
    assert column.distinct_count == 3  # "a", "" and " "
    assert column.inferred_type == InferredType.STRING


def test_csv_blank_cells_and_na_tokens_are_null() -> None:
    csv_text = "id,name,score\n1,Ann,\n2,,5.5\n3,NA,7.5\n"
    df = pd.read_csv(io.StringIO(csv_text))

    columns = profile_columns(df)

    assert columns["id"].inferred_type == InferredType.INTEGER
    assert columns["id"].null_count == 0
    assert columns["name"].null_count == 2  # one blank cell and the "NA" token
    assert columns["name"].null_percent == pytest.approx(66.67)
    assert columns["name"].distinct_count == 1
    assert columns["score"].null_count == 1
    assert columns["score"].inferred_type == InferredType.FLOAT


def test_csv_integer_column_with_blanks_is_integer() -> None:
    df = pd.read_csv(io.StringIO("id,limit\n1,100\n2,\n3,200\n"))

    column = profile_columns(df)["limit"]

    assert column.pandas_dtype == "float64"  # pandas stores integers with blanks as floats
    assert column.inferred_type == InferredType.INTEGER  # the profiler corrects that artifact
    assert column.null_count == 1


def test_all_null_column_is_empty() -> None:
    df = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})

    columns = profile_columns(df)

    assert columns["a"].inferred_type == InferredType.EMPTY
    assert columns["a"].null_count == 3
    assert columns["a"].null_percent == pytest.approx(100.0)
    assert columns["a"].distinct_count == 0
    assert columns["b"].inferred_type == InferredType.INTEGER


def test_nullable_integer_column() -> None:
    df = pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64")})

    column = profile_columns(df)["n"]

    assert column.inferred_type == InferredType.INTEGER
    assert column.null_count == 1
    assert column.distinct_count == 2


# --- Empty datasets --------------------------------------------------------


def test_empty_dataframe_without_columns() -> None:
    profile = profile_dataframe(pd.DataFrame())

    assert profile.row_count == 0
    assert profile.column_count == 0
    assert profile.columns == ()


def test_headers_only_csv_has_columns_but_no_rows() -> None:
    df = pd.read_csv(io.StringIO("a,b\n"))

    profile = profile_dataframe(df)

    assert profile.row_count == 0
    assert profile.column_count == 2
    for column in profile.columns:
        assert column.row_count == 0
        assert column.null_count == 0
        assert column.null_percent == 0.0  # no division by zero
        assert column.distinct_count == 0
        assert column.inferred_type == InferredType.EMPTY


def test_zero_row_frame_ignores_declared_dtype() -> None:
    df = pd.DataFrame({"n": pd.Series([], dtype="int64")})

    column = profile_columns(df)["n"]

    assert column.pandas_dtype == "int64"  # the observed dtype is still reported
    assert column.inferred_type == InferredType.EMPTY  # but there are no values to infer from


# --- Ambiguous and mixed values --------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param([1.0, 2.0, 3.0], InferredType.INTEGER, id="whole-number-floats"),
        pytest.param([1.0, 2.5], InferredType.FLOAT, id="fractional-floats"),
        pytest.param(["2026-09-01", "2026-09-02"], InferredType.DATETIME, id="iso-date-text"),
        pytest.param(
            ["2026-09-01 14:30", "2026-09-02T08:15:00"],
            InferredType.DATETIME,
            id="iso-datetime-text",
        ),
        pytest.param(
            ["2026-02-30", "2026-09-01"],
            InferredType.DATETIME,
            id="impossible-date-passes-shape-check",
        ),
        pytest.param(["2026-09-01", "not a date"], InferredType.STRING, id="one-bad-date-value"),
        pytest.param(["001", "002", "010"], InferredType.STRING, id="numeric-looking-text"),
        pytest.param(["a", "b"], InferredType.STRING, id="plain-text"),
        pytest.param([1, "a", 2.5], InferredType.MIXED, id="mixed-int-text-float"),
        pytest.param([1, "x"], InferredType.MIXED, id="mixed-int-text"),
        pytest.param([True, None, False], InferredType.BOOLEAN, id="booleans-with-null"),
    ],
)
def test_ambiguous_column_inference(values: list[object], expected: InferredType) -> None:
    column = profile_columns(pd.DataFrame({"col": values}))["col"]

    assert column.inferred_type == expected


def test_csv_placeholder_text_makes_numeric_column_string() -> None:
    df = pd.read_csv(io.StringIO("reorder_point\n40\nTBD\n30\n"))

    column = profile_columns(df)["reorder_point"]

    assert column.inferred_type == InferredType.STRING
    assert column.null_count == 0
    assert column.distinct_count == 3


# --- Privacy: no raw cell values -------------------------------------------


def test_output_contains_no_raw_cell_values() -> None:
    df = pd.DataFrame(
        {
            "email": ["jane.doe@example.com", "SECRET-VALUE-ALPHA"],
            "phone": ["555-0199", "555-0199"],
            "amount": [987654.321, 12.5],
        }
    )

    profile = profile_dataframe(df)
    rendered = f"{profile!r} {profile} {dataclasses.asdict(profile)}"

    for raw_value in ("jane.doe@example.com", "SECRET-VALUE-ALPHA", "555-0199", "987654"):
        assert raw_value not in rendered


def test_result_fields_are_summary_scalars_only() -> None:
    df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    profile = profile_dataframe(df)

    assert {f.name for f in dataclasses.fields(DatasetProfileResult)} == {
        "row_count",
        "column_count",
        "columns",
    }
    assert {f.name for f in dataclasses.fields(ColumnProfileResult)} == {
        "column_name",
        "pandas_dtype",
        "inferred_type",
        "row_count",
        "null_count",
        "null_percent",
        "distinct_count",
    }
    for column in profile.columns:
        for field in dataclasses.fields(column):
            value = getattr(column, field.name)
            assert isinstance(value, (str, int, float, type(None)))


def test_results_are_immutable() -> None:
    column = profile_dataframe(pd.DataFrame({"a": [1]})).columns[0]

    with pytest.raises(dataclasses.FrozenInstanceError):
        column.null_count = 99  # type: ignore[misc]


# --- Purity and determinism ------------------------------------------------


def test_profiling_does_not_modify_input() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
    snapshot = df.copy(deep=True)

    profile_dataframe(df)

    pd.testing.assert_frame_equal(df, snapshot)


def test_profiling_is_deterministic() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "x"]})

    assert profile_dataframe(df) == profile_dataframe(df)


# --- Synthetic sample files ------------------------------------------------


@pytest.mark.parametrize("name", ["customers", "orders", "inventory"])
def test_sample_files_profile_cleanly(name: str) -> None:
    df = pd.read_csv(SAMPLE_DIR / f"{name}.csv")

    profile = profile_dataframe(df)

    assert profile.row_count == len(df)
    assert profile.column_count == df.shape[1]
    assert [c.column_name for c in profile.columns] == list(df.columns)
    for column in profile.columns:
        assert column.row_count == len(df)
        assert 0 <= column.null_count <= len(df)
        assert 0.0 <= column.null_percent <= 100.0
        assert column.distinct_count is not None
        assert column.distinct_count <= len(df) - column.null_count