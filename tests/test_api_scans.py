from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Base
from app.workflows import scan_graph

CLEAN_CSV = b"a,b\n1,x\n2,y\n"
RAW_VALUES = ("a@example.com", "call dock", "leave door")
SCAN_RESULT_KEYS = {
    "dataset_id",
    "scan_run_id",
    "status",
    "started_at",
    "finished_at",
    "row_count",
    "column_count",
    "finding_count",
    "findings_by_severity",
    "findings_by_category",
    "findings",
}
SCAN_SUMMARY = {
    "row_count": 4,
    "column_count": 5,
    "finding_count": 3,
    "findings_by_severity": {"low": 0, "medium": 2, "high": 1},
    "findings_by_category": {"missing_values": 2, "potential_email": 1},
}


# --- Successful uploads ----------------------------------------------------------


def test_upload_creates_completed_scan(upload: Callable[..., Any], mixed_csv: bytes) -> None:
    response = upload(mixed_csv, "mixed.csv", dataset_name="mixed_sample")

    assert response.status_code == 201
    body = response.json()
    assert set(body) == SCAN_RESULT_KEYS
    assert body["status"] == "completed"
    assert (body["row_count"], body["column_count"], body["finding_count"]) == (4, 5, 3)
    assert body["findings_by_severity"] == SCAN_SUMMARY["findings_by_severity"]
    assert body["findings_by_category"] == SCAN_SUMMARY["findings_by_category"]
    assert [(f["severity"], f["category"], f["column_name"]) for f in body["findings"]] == [
        ("high", "missing_values", "notes"),
        ("medium", "missing_values", "memo"),
        ("medium", "potential_email", "contact"),
    ]
    assert {f["review_status"] for f in body["findings"]} == {"pending"}
    assert set(body["findings"][0]) == {
        "id",
        "category",
        "severity",
        "column_name",
        "title",
        "review_status",
    }


def test_upload_response_contains_no_raw_values(upload: Callable[..., Any], mixed_csv: bytes) -> None:
    response = upload(mixed_csv, "mixed.csv")

    for raw_value in RAW_VALUES:
        assert raw_value not in response.text


def test_database_stores_no_raw_values(
    upload: Callable[..., Any], mixed_csv: bytes, session: Session
) -> None:
    assert upload(mixed_csv, "mixed.csv").status_code == 201

    dump = "\n".join(
        repr(tuple(row))
        for table in Base.metadata.sorted_tables
        for row in session.execute(select(table)).all()
    )

    assert "contact" in dump  # column names and counts are stored
    for raw_value in RAW_VALUES:
        assert raw_value not in dump


@pytest.mark.parametrize(
    ("form", "filename", "expected_name"),
    [
        pytest.param({"dataset_name": "My Dataset"}, "data.csv", "My Dataset", id="given-name"),
        pytest.param({}, "inventory_2026.csv", "inventory_2026", id="file-stem"),
        pytest.param({"dataset_name": "   "}, "orders.csv", "orders", id="blank-name-uses-stem"),
        pytest.param({"dataset_name": "  padded  "}, "data.csv", "padded", id="name-is-trimmed"),
    ],
)
def test_dataset_name_resolution(
    client: TestClient,
    upload: Callable[..., Any],
    form: dict[str, str],
    filename: str,
    expected_name: str,
) -> None:
    created = upload(CLEAN_CSV, filename, **form)
    assert created.status_code == 201

    dataset = client.get(f"/datasets/{created.json()['dataset_id']}").json()

    assert dataset["name"] == expected_name
    assert dataset["source_filename"] == filename


def test_dataset_name_too_long(upload: Callable[..., Any]) -> None:
    response = upload(CLEAN_CSV, dataset_name="n" * 256)

    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "validation_error"
    assert any(issue["location"] == "body.dataset_name" for issue in body["issues"])


def test_clean_upload_has_no_findings(client: TestClient, upload: Callable[..., Any]) -> None:
    response = upload(CLEAN_CSV)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["finding_count"] == 0
    assert body["findings"] == []
    assert client.get("/findings").json()["total"] == 0


def test_utf8_bom_upload(client: TestClient, upload: Callable[..., Any]) -> None:
    response = upload(b"\xef\xbb\xbf" + CLEAN_CSV)

    assert response.status_code == 201
    columns = client.get(f"/datasets/{response.json()['dataset_id']}/columns").json()["columns"]
    assert [c["column_name"] for c in columns] == ["a", "b"]


def test_each_upload_creates_independent_records(
    client: TestClient, upload: Callable[..., Any]
) -> None:
    first = upload(CLEAN_CSV, dataset_name="same_name").json()
    second = upload(CLEAN_CSV, dataset_name="same_name").json()

    assert first["dataset_id"] != second["dataset_id"]
    assert first["scan_run_id"] != second["scan_run_id"]
    assert client.get("/datasets").json()["total"] == 2


# --- Reading a scan --------------------------------------------------------------


def test_get_scan_returns_status_and_summary(client: TestClient, scanned: dict[str, Any]) -> None:
    response = client.get(f"/scans/{scanned['scan_run_id']}")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "id",
        "dataset_id",
        "status",
        "started_at",
        "finished_at",
        "summary",
        "error_details",
    }
    assert body["id"] == scanned["scan_run_id"]
    assert body["dataset_id"] == scanned["dataset_id"]
    assert body["status"] == "completed"
    assert body["summary"] == SCAN_SUMMARY
    assert body["error_details"] is None
    assert body["started_at"].endswith("Z")
    assert body["finished_at"].endswith("Z")


def test_get_scan_not_found(client: TestClient) -> None:
    response = client.get("/scans/999")

    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "not_found", "message": "Scan 999 was not found."}
    }


# --- Invalid uploads -------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "code"),
    [
        pytest.param(b"", "empty_file", id="zero-bytes"),
        pytest.param(b"a,b\n", "empty_file", id="header-only"),
        pytest.param(b"a,b\n1\n", "malformed_csv", id="too-few-fields"),
        pytest.param(b'a,b\n1,"x\n', "malformed_csv", id="unclosed-quote"),
        pytest.param(b"a,a\n1,2\n", "invalid_header", id="duplicate-header"),
        pytest.param(b"a,b\x07c\n1,2\n", "invalid_header", id="control-character-header"),
        pytest.param(b"name\ncaf\xe9\n", "invalid_encoding", id="latin-1"),
        pytest.param(b"a,b\n1,2\x00\n", "invalid_encoding", id="nul-byte"),
    ],
)
def test_ingestion_failures_are_422_and_store_nothing(
    client: TestClient, upload: Callable[..., Any], content: bytes, code: str
) -> None:
    response = upload(content)

    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == code
    assert body["message"]
    assert client.get("/datasets").json()["total"] == 0


@pytest.mark.parametrize(
    "content",
    [
        b"SECRET-HEADER,SECRET-HEADER\n1,2\n",
        b"a,b\nSECRET-VALUE,1,2\n",
        b"a,SECRET-HEADER\nSECRET-VALUE,caf\xe9\n",
    ],
    ids=["duplicate-header", "wrong-field-count", "bad-encoding"],
)
def test_ingestion_errors_do_not_echo_content(
    upload: Callable[..., Any], content: bytes
) -> None:
    response = upload(content)

    assert response.status_code == 422
    assert "SECRET" not in response.text


# --- Limits ----------------------------------------------------------------------


def test_row_limit(
    client: TestClient, upload: Callable[..., Any], set_limits: Callable[..., None]
) -> None:
    set_limits(max_rows=2)

    assert upload(b"a\n1\n2\n").status_code == 201
    rejected = upload(b"a\n1\n2\n3\n")

    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "too_many_rows"
    assert client.get("/datasets").json()["total"] == 1


def test_column_limit(
    client: TestClient, upload: Callable[..., Any], set_limits: Callable[..., None]
) -> None:
    set_limits(max_columns=2)

    rejected = upload(b"a,b,c\n1,2,3\n")

    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "too_many_columns"
    assert client.get("/datasets").json()["total"] == 0


@pytest.mark.parametrize(
    ("max_bytes", "expected_status"),
    [pytest.param(4, 201, id="exactly-at-limit"), pytest.param(3, 413, id="one-byte-over")],
)
def test_size_limit_boundary_in_the_route(
    upload: Callable[..., Any],
    set_limits: Callable[..., None],
    max_bytes: int,
    expected_status: int,
) -> None:
    set_limits(max_bytes=max_bytes)

    response = upload(b"a\n1\n")  # exactly 4 bytes

    assert response.status_code == expected_status
    if expected_status == 413:
        assert response.json() == {
            "error": {
                "code": "file_too_large",
                "message": "The file exceeds the limit of 3 bytes.",
            }
        }


# --- Scan failure ----------------------------------------------------------------


def test_scan_failure_returns_500_and_records_failure(
    client: TestClient,
    upload: Callable[..., Any],
    mixed_csv: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom SECRET-VALUE")

    monkeypatch.setattr(scan_graph, "run_detectors", explode)

    response = upload(mixed_csv, "mixed.csv")

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "scan_failed"
    assert "RuntimeError" in error["message"]
    assert "SECRET" not in response.text

    datasets = client.get("/datasets").json()
    assert datasets["total"] == 1
    dataset_id = datasets["items"][0]["id"]
    dataset = client.get(f"/datasets/{dataset_id}").json()
    assert dataset["latest_scan_status"] == "failed"

    scan = client.get(f"/scans/{dataset['latest_scan_id']}").json()
    assert scan["status"] == "failed"
    assert scan["summary"] is None
    assert "RuntimeError" in scan["error_details"]
    assert "SECRET" not in scan["error_details"]
    assert client.get("/findings").json()["total"] == 0
    assert client.get(f"/datasets/{dataset_id}/columns").json()["columns"] == []