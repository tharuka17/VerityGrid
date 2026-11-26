from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.routes.findings import DetectionMethod
from app.schemas.findings import DateRule, DetectionConfig, NumericRule
from app.services import query_service as q
from app.services.detectors import run_detectors

ALL_METHODS = [method.value for method in DetectionMethod]
# The shared `scanned` fixture (tests/conftest.py) produces two missing-value findings and one
# email finding under the default configuration.
EXPECTED_ON_MIXED_SCAN = {"null_percent_threshold": 2, "regex_email_pattern": 1}


def all_methods_frame() -> pd.DataFrame:
    """One frame on which every detector fires once under all_methods_config()."""
    return pd.DataFrame(
        {
            "email": ["a@example.com", "b@example.com", "c@example.com", "a@example.com"],
            "phone": ["503-555-0101", "503-555-0102", "503-555-0103", "503-555-0101"],
            "qty": [5, -2, 7, 5],
            "counted": ["2026-09-01", "2026-02-30", "2026-09-03", "2026-09-01"],
            "notes": [None, None, None, None],
        }
    )


def all_methods_config() -> DetectionConfig:
    return DetectionConfig(
        numeric_rules={"qty": NumericRule(min_value=0)},
        date_rules={"counted": DateRule()},
    )


# --- The enum and the detectors agree ----------------------------------------------------


def test_every_method_the_detectors_produce_is_listed_in_the_api_enum() -> None:
    findings = run_detectors(all_methods_frame(), all_methods_config())

    produced = {finding.detection_method for finding in findings}

    assert produced == set(ALL_METHODS)


# --- Filtering through the API -------------------------------------------------------------


def test_without_the_filter_nothing_changes(client: TestClient, scanned: dict[str, Any]) -> None:
    body = client.get("/findings").json()

    assert body["total"] == 3
    assert {item["detection_method"] for item in body["items"]} == set(EXPECTED_ON_MIXED_SCAN)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_filter_by_each_method(client: TestClient, scanned: dict[str, Any], method: str) -> None:
    expected = EXPECTED_ON_MIXED_SCAN.get(method, 0)

    response = client.get("/findings", params={"detection_method": method})

    assert response.status_code == 200  # a method with no findings is an empty page, not an error
    body = response.json()
    assert body["total"] == expected
    assert len(body["items"]) == expected
    assert {item["detection_method"] for item in body["items"]} <= {method}


@pytest.mark.parametrize(
    ("extra", "expected_total"),
    [
        pytest.param({"severity": "high"}, 1, id="high-severity"),
        pytest.param({"severity": "medium"}, 1, id="medium-severity"),
        pytest.param({"category": "potential_email"}, 0, id="contradicting-category"),
    ],
)
def test_filter_combines_with_other_filters_by_and(
    client: TestClient, scanned: dict[str, Any], extra: dict[str, str], expected_total: int
) -> None:
    params = {"detection_method": "null_percent_threshold", **extra}

    body = client.get("/findings", params=params).json()

    assert body["total"] == expected_total


@pytest.mark.parametrize(
    "value",
    ["bogus", "REGEX_EMAIL_PATTERN", "regex_email_pattern;drop", "SECRET-METHOD"],
    ids=["unknown", "wrong-case", "injection-text", "secret-text"],
)
def test_unknown_methods_are_rejected_without_echo(client: TestClient, value: str) -> None:
    response = client.get("/findings", params={"detection_method": value})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert [issue["location"] for issue in error["issues"]] == ["query.detection_method"]
    assert value not in response.text  # the submitted value is never echoed


def test_filter_total_ignores_paging(client: TestClient, upload: Callable[..., Any], mixed_csv: bytes) -> None:
    for _ in range(3):
        assert upload(mixed_csv, "mixed.csv").status_code == 201  # two missing-value findings each

    params = {"detection_method": "null_percent_threshold", "limit": 2}
    first = client.get("/findings", params={**params, "offset": 0}).json()
    last = client.get("/findings", params={**params, "offset": 4}).json()
    beyond = client.get("/findings", params={**params, "offset": 6}).json()

    assert (first["total"], len(first["items"])) == (6, 2)
    assert (last["total"], len(last["items"])) == (6, 2)
    assert (beyond["total"], beyond["items"]) == (6, [])


def test_openapi_documents_the_filter(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    parameters = spec["paths"]["/findings"]["get"]["parameters"]
    assert "detection_method" in {parameter["name"] for parameter in parameters}
    enums = [set(schema["enum"]) for schema in spec["components"]["schemas"].values() if "enum" in schema]
    assert set(ALL_METHODS) in enums


# --- Filtering in the query service --------------------------------------------------------


def test_query_service_filters_by_method(session: Session, scanned: dict[str, Any]) -> None:
    page = q.list_findings(session, limit=10, offset=0, detection_method="regex_email_pattern")

    assert page.total == 1
    assert page.items[0].category == "potential_email"
    assert page.items[0].detection_method == "regex_email_pattern"