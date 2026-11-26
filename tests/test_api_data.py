from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AuditEvent, Dataset

CLEAN_CSV = b"a,b\n1,x\n2,y\n"
RAW_VALUES = ("a@example.com", "call dock", "leave door")
DATASET_KEYS = {
    "id",
    "name",
    "source_filename",
    "row_count",
    "column_count",
    "created_at",
    "updated_at",
}
FINDING_KEYS = {
    "id",
    "dataset_id",
    "scan_run_id",
    "category",
    "severity",
    "column_name",
    "title",
    "description",
    "safe_evidence",
    "detection_method",
    "recommendation",
    "review_status",
    "created_at",
}


def finding_id(scanned: dict[str, Any], category: str, column: str | None = None) -> int:
    return next(
        f["id"]
        for f in scanned["findings"]
        if f["category"] == category and (column is None or f["column_name"] == column)
    )


def review(client: TestClient, finding: int, **body: Any) -> Any:
    return client.post(f"/findings/{finding}/review", json=body)


# --- Datasets --------------------------------------------------------------------


def test_list_datasets_empty(client: TestClient) -> None:
    response = client.get("/datasets")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 50, "offset": 0}


def test_list_datasets_order_and_pagination(client: TestClient, upload: Callable[..., Any]) -> None:
    for name in ("d1", "d2", "d3"):
        assert upload(CLEAN_CSV, dataset_name=name).status_code == 201

    page = client.get("/datasets").json()
    assert page["total"] == 3
    assert [item["name"] for item in page["items"]] == ["d3", "d2", "d1"]  # newest first
    assert (page["limit"], page["offset"]) == (50, 0)
    assert set(page["items"][0]) == DATASET_KEYS

    first = client.get("/datasets", params={"limit": 2}).json()
    assert [item["name"] for item in first["items"]] == ["d3", "d2"]

    second = client.get("/datasets", params={"limit": 2, "offset": 2}).json()
    assert [item["name"] for item in second["items"]] == ["d1"]
    assert (second["total"], second["limit"], second["offset"]) == (3, 2, 2)

    beyond = client.get("/datasets", params={"offset": 10}).json()
    assert beyond["items"] == []
    assert beyond["total"] == 3


def test_get_dataset_detail(client: TestClient, scanned: dict[str, Any]) -> None:
    response = client.get(f"/datasets/{scanned['dataset_id']}")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == DATASET_KEYS | {
        "scan_count",
        "finding_count",
        "pending_review_count",
        "latest_scan_id",
        "latest_scan_status",
    }
    assert body["name"] == "mixed_sample"
    assert body["source_filename"] == "mixed.csv"
    assert (body["row_count"], body["column_count"]) == (4, 5)
    assert (body["scan_count"], body["finding_count"], body["pending_review_count"]) == (1, 3, 3)
    assert body["latest_scan_id"] == scanned["scan_run_id"]
    assert body["latest_scan_status"] == "completed"
    assert body["created_at"].endswith("Z")


def test_dataset_not_found(client: TestClient) -> None:
    response = client.get("/datasets/999")

    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "not_found", "message": "Dataset 999 was not found."}
    }


def test_dataset_columns(client: TestClient, scanned: dict[str, Any]) -> None:
    response = client.get(f"/datasets/{scanned['dataset_id']}/columns")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == scanned["dataset_id"]
    assert body["row_count"] == 4
    assert [c["column_name"] for c in body["columns"]] == ["sku", "contact", "qty", "notes", "memo"]
    assert set(body["columns"][0]) == {
        "id",
        "column_name",
        "inferred_type",
        "null_count",
        "null_percent",
        "distinct_count",
        "classification_label",
        "classification_source",
        "classification_explanation",
    }

    columns = {c["column_name"]: c for c in body["columns"]}
    assert columns["qty"]["inferred_type"] == "integer"
    assert columns["sku"]["inferred_type"] == "string"
    assert columns["sku"]["distinct_count"] == 4
    assert columns["notes"]["inferred_type"] == "empty"
    assert (columns["notes"]["null_count"], columns["notes"]["null_percent"]) == (4, 100.0)
    assert (columns["memo"]["null_count"], columns["memo"]["null_percent"]) == (1, 25.0)
    assert columns["memo"]["distinct_count"] == 3
    assert all(c["classification_label"] is None for c in body["columns"])


def test_columns_not_found(client: TestClient) -> None:
    response = client.get("/datasets/999/columns")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_pending_review_count_follows_reviews(client: TestClient, scanned: dict[str, Any]) -> None:
    review(client, finding_id(scanned, "potential_email"), decision="approved")

    body = client.get(f"/datasets/{scanned['dataset_id']}").json()

    assert body["finding_count"] == 3
    assert body["pending_review_count"] == 2


# --- Findings --------------------------------------------------------------------


def test_list_findings_order_and_shape(client: TestClient, scanned: dict[str, Any]) -> None:
    response = client.get("/findings")

    assert response.status_code == 200
    page = response.json()
    assert page["total"] == 3
    assert [(f["severity"], f["category"], f["column_name"]) for f in page["items"]] == [
        ("high", "missing_values", "notes"),
        ("medium", "missing_values", "memo"),
        ("medium", "potential_email", "contact"),
    ]
    assert set(page["items"][0]) == FINDING_KEYS
    assert {f["review_status"] for f in page["items"]} == {"pending"}
    assert {f["dataset_id"] for f in page["items"]} == {scanned["dataset_id"]}
    assert {f["scan_run_id"] for f in page["items"]} == {scanned["scan_run_id"]}


@pytest.mark.parametrize(
    ("make_params", "expected_total"),
    [
        pytest.param(lambda s: {"severity": "high"}, 1, id="severity-high"),
        pytest.param(lambda s: {"severity": "medium"}, 2, id="severity-medium"),
        pytest.param(lambda s: {"severity": "low"}, 0, id="severity-low"),
        pytest.param(lambda s: {"category": "missing_values"}, 2, id="category-missing"),
        pytest.param(lambda s: {"category": "potential_email"}, 1, id="category-email"),
        pytest.param(lambda s: {"category": "duplicate_rows"}, 0, id="category-absent"),
        pytest.param(lambda s: {"review_status": "pending"}, 3, id="status-pending"),
        pytest.param(lambda s: {"review_status": "approved"}, 0, id="status-approved"),
        pytest.param(lambda s: {"dataset_id": s["dataset_id"]}, 3, id="dataset"),
        pytest.param(lambda s: {"dataset_id": s["dataset_id"] + 1000}, 0, id="other-dataset"),
        pytest.param(lambda s: {"scan_id": s["scan_run_id"]}, 3, id="scan"),
        pytest.param(
            lambda s: {"severity": "medium", "category": "missing_values"}, 1, id="combined"
        ),
    ],
)
def test_finding_filters(
    client: TestClient,
    scanned: dict[str, Any],
    make_params: Callable[[dict[str, Any]], dict[str, Any]],
    expected_total: int,
) -> None:
    response = client.get("/findings", params=make_params(scanned))

    assert response.status_code == 200  # a filter that matches nothing is an empty page, not 404
    body = response.json()
    assert body["total"] == expected_total
    assert len(body["items"]) == expected_total


def test_findings_pagination(client: TestClient, scanned: dict[str, Any]) -> None:
    everything = [item["id"] for item in client.get("/findings").json()["items"]]

    first = client.get("/findings", params={"limit": 2}).json()
    rest = client.get("/findings", params={"limit": 2, "offset": 2}).json()

    assert first["total"] == rest["total"] == 3
    assert len(first["items"]) == 2
    assert len(rest["items"]) == 1
    assert [i["id"] for i in first["items"]] + [i["id"] for i in rest["items"]] == everything


def test_get_finding_detail(client: TestClient, scanned: dict[str, Any]) -> None:
    fid = finding_id(scanned, "potential_email")

    response = client.get(f"/findings/{fid}")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == FINDING_KEYS | {"review_decisions"}
    assert body["category"] == "potential_email"
    assert body["severity"] == "medium"
    assert body["column_name"] == "contact"
    assert body["detection_method"] == "regex_email_pattern"
    assert body["title"] == "Potential email addresses in 'contact'"
    assert "1 of 4" in body["description"]
    assert body["recommendation"]
    assert body["review_status"] == "pending"
    assert body["review_decisions"] == []
    assert body["safe_evidence"] == {
        "non_null_count": 4,
        "match_count": 1,
        "match_percent": 25.0,
        "pattern_family": "email",
    }


def test_finding_not_found(client: TestClient) -> None:
    response = client.get("/findings/999")

    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "not_found", "message": "Finding 999 was not found."}
    }


# --- Reviews ---------------------------------------------------------------------


def test_review_round_trip(client: TestClient, scanned: dict[str, Any]) -> None:
    fid = finding_id(scanned, "potential_email")

    response = review(client, fid, decision="approved", comment="checked", reviewer="tester")

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"decision", "finding"}
    assert set(body["decision"]) == {
        "id",
        "finding_id",
        "decision",
        "comment",
        "reviewer",
        "created_at",
    }
    assert body["decision"]["finding_id"] == fid
    assert body["decision"]["decision"] == "approved"
    assert body["decision"]["comment"] == "checked"
    assert body["decision"]["reviewer"] == "tester"
    assert body["decision"]["created_at"].endswith("Z")
    assert body["finding"]["id"] == fid
    assert body["finding"]["review_status"] == "approved"

    detail = client.get(f"/findings/{fid}").json()
    assert detail["review_status"] == "approved"
    assert [d["decision"] for d in detail["review_decisions"]] == ["approved"]

    approved = client.get("/findings", params={"review_status": "approved"}).json()
    assert approved["total"] == 1
    assert approved["items"][0]["id"] == fid


def test_review_history_latest_wins(client: TestClient, scanned: dict[str, Any]) -> None:
    fid = finding_id(scanned, "potential_email")

    assert review(client, fid, decision="approved", reviewer="alice").status_code == 201
    assert review(client, fid, decision="rejected", reviewer="bob").status_code == 201
    detail = client.get(f"/findings/{fid}").json()

    assert detail["review_status"] == "rejected"
    assert [(d["decision"], d["reviewer"]) for d in detail["review_decisions"]] == [
        ("approved", "alice"),
        ("rejected", "bob"),
    ]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"decision": "pending"}, id="pending-is-not-a-decision"),
        pytest.param({"decision": "deleted"}, id="unknown-decision"),
        pytest.param({}, id="missing-decision"),
        pytest.param({"decision": "approved", "comment": "x" * 1001}, id="comment-too-long"),
        pytest.param({"decision": "approved", "reviewer": "r" * 101}, id="reviewer-too-long"),
        pytest.param({"decision": "approved", "remediate": True}, id="extra-field"),
    ],
)
def test_invalid_review_bodies_are_rejected(
    client: TestClient, scanned: dict[str, Any], session: Session, body: dict[str, Any]
) -> None:
    fid = finding_id(scanned, "potential_email")

    response = review(client, fid, **body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "x" * 50 not in response.text
    detail = client.get(f"/findings/{fid}").json()
    assert detail["review_status"] == "pending"
    assert detail["review_decisions"] == []
    assert session.scalar(select(func.count()).select_from(AuditEvent)) == 0


def test_review_unknown_finding(client: TestClient, session: Session) -> None:
    response = review(client, 999, decision="approved")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert client.get("/metrics/summary").json()["review_decision_count"] == 0
    assert session.scalar(select(func.count()).select_from(AuditEvent)) == 0


def test_review_trims_blank_text(client: TestClient, scanned: dict[str, Any]) -> None:
    fid = finding_id(scanned, "potential_email")

    body = review(client, fid, decision="approved", comment="   ", reviewer="  alice  ").json()

    assert body["decision"]["comment"] is None
    assert body["decision"]["reviewer"] == "alice"


def test_review_writes_audit_event(
    client: TestClient, scanned: dict[str, Any], session: Session
) -> None:
    fid = finding_id(scanned, "potential_email")

    review(client, fid, decision="approved", comment="SECRET-COMMENT-TEXT", reviewer="tester")
    events = session.scalars(select(AuditEvent)).all()

    assert len(events) == 1
    event = events[0]
    assert (event.actor, event.event_type, event.entity_type, event.entity_id) == (
        "tester",
        "review.approved",
        "finding",
        fid,
    )
    assert event.safe_metadata == {
        "dataset_id": scanned["dataset_id"],
        "scan_run_id": scanned["scan_run_id"],
        "category": "potential_email",
        "severity": "medium",
        "previous_status": "pending",
        "new_status": "approved",
        "has_comment": True,
        "actor_self_reported": True,
    }
    assert "SECRET-COMMENT-TEXT" not in repr(event.safe_metadata)


def test_review_changes_only_review_state(client: TestClient, scanned: dict[str, Any]) -> None:
    target = finding_id(scanned, "potential_email")
    ids = [f["id"] for f in scanned["findings"]]
    before = {fid: client.get(f"/findings/{fid}").json() for fid in ids}

    review(client, target, decision="rejected", comment="not an issue")
    after = {fid: client.get(f"/findings/{fid}").json() for fid in ids}

    ignored = {"review_status", "review_decisions"}
    for fid in ids:
        if fid == target:
            assert after[fid]["review_status"] == "rejected"
            assert {k: v for k, v in after[fid].items() if k not in ignored} == {
                k: v for k, v in before[fid].items() if k not in ignored
            }
        else:
            assert after[fid] == before[fid]
    dataset = client.get(f"/datasets/{scanned['dataset_id']}").json()
    assert dataset["finding_count"] == 3
    assert client.get("/datasets").json()["total"] == 1


# --- Metrics ---------------------------------------------------------------------


def test_metrics_empty(client: TestClient) -> None:
    response = client.get("/metrics/summary")

    assert response.status_code == 200
    assert response.json() == {
        "dataset_count": 0,
        "scan_count": 0,
        "scans_by_status": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
        "finding_count": 0,
        "findings_by_severity": {"low": 0, "medium": 0, "high": 0},
        "findings_by_category": {},
        "findings_by_review_status": {"pending": 0, "approved": 0, "rejected": 0},
        "pending_high_severity_count": 0,
        "review_decision_count": 0,
        "reviewed_finding_percent": None,
    }


def test_metrics_after_scan(client: TestClient, scanned: dict[str, Any]) -> None:
    body = client.get("/metrics/summary").json()

    assert body["dataset_count"] == 1
    assert body["scan_count"] == 1
    assert body["scans_by_status"] == {"pending": 0, "running": 0, "completed": 1, "failed": 0}
    assert body["finding_count"] == 3
    assert body["findings_by_severity"] == {"low": 0, "medium": 2, "high": 1}
    assert body["findings_by_category"] == {"missing_values": 2, "potential_email": 1}
    assert body["findings_by_review_status"] == {"pending": 3, "approved": 0, "rejected": 0}
    assert body["pending_high_severity_count"] == 1
    assert body["review_decision_count"] == 0
    assert body["reviewed_finding_percent"] == 0.0


def test_metrics_follow_reviews(client: TestClient, scanned: dict[str, Any]) -> None:
    high = finding_id(scanned, "missing_values", "notes")
    medium = finding_id(scanned, "potential_email")

    review(client, high, decision="approved")
    body = client.get("/metrics/summary").json()
    assert body["findings_by_review_status"] == {"pending": 2, "approved": 1, "rejected": 0}
    assert body["review_decision_count"] == 1
    assert body["pending_high_severity_count"] == 0
    assert body["reviewed_finding_percent"] == pytest.approx(33.33)

    review(client, medium, decision="rejected")
    body = client.get("/metrics/summary").json()
    assert body["findings_by_review_status"] == {"pending": 1, "approved": 1, "rejected": 1}
    assert body["review_decision_count"] == 2
    assert body["reviewed_finding_percent"] == pytest.approx(66.67)

    review(client, high, decision="rejected")  # changing a decision adds history, not findings
    body = client.get("/metrics/summary").json()
    assert body["findings_by_review_status"] == {"pending": 1, "approved": 0, "rejected": 2}
    assert body["review_decision_count"] == 3
    assert body["finding_count"] == 3
    assert body["reviewed_finding_percent"] == pytest.approx(66.67)


def test_metrics_come_from_stored_rows(client: TestClient, session: Session) -> None:
    """A row inserted directly into the database shows up; nothing is pre-filled or estimated."""
    session.add(Dataset(name="direct_row"))
    session.commit()

    body = client.get("/metrics/summary").json()

    assert body["dataset_count"] == 1
    assert body["scan_count"] == 0
    assert body["finding_count"] == 0
    assert body["reviewed_finding_percent"] is None


# --- Privacy across every read endpoint ------------------------------------------


def test_no_raw_values_in_any_response(client: TestClient, scanned: dict[str, Any]) -> None:
    fid = finding_id(scanned, "potential_email")
    review(client, fid, decision="approved", comment="checked", reviewer="tester")
    dataset = scanned["dataset_id"]
    paths = [
        f"/scans/{scanned['scan_run_id']}",
        "/datasets",
        f"/datasets/{dataset}",
        f"/datasets/{dataset}/columns",
        "/findings",
        f"/findings/{fid}",
        "/metrics/summary",
    ]

    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, path
        for raw_value in RAW_VALUES:
            assert raw_value not in response.text, path