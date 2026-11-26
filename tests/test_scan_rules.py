import ast
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes import scans as scans_route
from app.schemas import rules as rules_module
from app.schemas.findings import DetectionConfig
from app.schemas.rules import (
    MAX_RULES_CHARS,
    InvalidRulesError,
    check_rule_columns,
    parse_rules,
)

ROOT = Path(__file__).resolve().parent.parent
INVENTORY_CSV = b"sku,quantity,counted\nA1,5,2026-01-05\nA2,-12,2026-02-30\nA3,7,2026-03-01\n"
GOOD_RULES = json.dumps(
    {
        "numeric_rules": {"quantity": {"min_value": 0}},
        "date_rules": {"counted": {"date_format": "%Y-%m-%d"}},
    }
)
HOSTILE_KEYS = ["SECRET-KEY-NAME", "x'; DROP TABLE t; --", "<script>alert(1)</script>"]


def rejected(text: str | None) -> InvalidRulesError:
    with pytest.raises(InvalidRulesError) as excinfo:
        parse_rules(text)
    return excinfo.value


def locations(error: InvalidRulesError) -> list[tuple[str, ...]]:
    return [issue.location for issue in error.issues]


def everything(error: InvalidRulesError) -> str:
    return " ".join([*(issue.message for issue in error.issues), *(".".join(i.location) for i in error.issues)])


def post(client: TestClient, csv: bytes = INVENTORY_CSV, rules: str | None = None, name: str = "data.csv") -> Any:
    return client.post(
        "/scans",
        files={"file": (name, csv, "text/csv")},
        data={} if rules is None else {"rules": rules},
    )


def dataset_total(client: TestClient) -> int:
    return int(client.get("/datasets").json()["total"])


# =============================================================================================
# 1. Parsing and validating the rules text
# =============================================================================================


@pytest.mark.parametrize("text", [None, "", "   ", "\n\t"], ids=["none", "empty", "spaces", "whitespace"])
def test_no_rules_means_no_config(text: str | None) -> None:
    assert parse_rules(text) is None


def test_an_empty_object_means_default_rules() -> None:
    assert parse_rules("{}") == DetectionConfig()


def test_a_full_rule_set_is_parsed() -> None:
    config = parse_rules(
        json.dumps(
            {
                "missing_value_threshold_percent": 30,
                "missing_value_overrides": {"notes": 80},
                "missing_value_exempt_columns": ["id"],
                "duplicate_subset": ["sku"],
                "pattern_columns": ["email"],
                "pattern_min_match_count": 2,
                "numeric_rules": {"quantity": {"min_value": 0, "max_value": 1000}},
                "date_rules": {"counted": {"date_format": "%d/%m/%Y"}},
            }
        )
    )

    assert config is not None
    assert config.missing_value_threshold_percent == 30
    assert config.missing_value_overrides == {"notes": 80}
    assert config.duplicate_subset == ["sku"] and config.pattern_columns == ["email"]
    assert config.numeric_rules["quantity"].max_value == 1000
    assert config.date_rules["counted"].date_format == "%d/%m/%Y"


@pytest.mark.parametrize(
    "text",
    ["not json", "{", '{"a": 1,}', "NaN", '{"a": Infinity}', '{"a": -Infinity}', "[]", '"text"', "42", "null", "true"],
)
def test_invalid_json_is_rejected(text: str) -> None:
    error = rejected(text)

    assert locations(error) == [()]
    assert error.issues[0].message


def test_oversized_text_is_rejected() -> None:
    error = rejected("x" * (MAX_RULES_CHARS + 1))

    assert locations(error) == [()]
    assert str(MAX_RULES_CHARS) in error.issues[0].message


def test_deep_nesting_is_rejected() -> None:
    rejected("[" * 9_000 + "]" * 9_000)


@pytest.mark.parametrize("key", HOSTILE_KEYS)
def test_unknown_fields_are_rejected_without_echo(key: str) -> None:
    error = rejected(json.dumps({key: 1}))

    assert ("<unknown field>",) in locations(error)
    assert key not in everything(error)


def test_nested_unknown_fields_are_rejected() -> None:
    error = rejected(json.dumps({"numeric_rules": {"quantity": {"min_value": 0, "SECRET-NESTED": 1}}}))

    assert ("numeric_rules", "<column>", "<unknown field>") in locations(error)
    assert "SECRET" not in everything(error) and "quantity" not in everything(error)


CAP_CASES = [
    ("numeric_rules", lambda n: {f"c{i}": {"min_value": 0} for i in range(n)}),
    ("date_rules", lambda n: {f"c{i}": {} for i in range(n)}),
    ("missing_value_overrides", lambda n: {f"c{i}": 50 for i in range(n)}),
    ("missing_value_exempt_columns", lambda n: [f"c{i}" for i in range(n)]),
    ("duplicate_subset", lambda n: [f"c{i}" for i in range(n)]),
    ("pattern_columns", lambda n: [f"c{i}" for i in range(n)]),
]


@pytest.mark.parametrize(("field", "build"), CAP_CASES, ids=[case[0] for case in CAP_CASES])
def test_entry_caps_are_enforced(field: str, build: Callable[[int], Any]) -> None:
    limit = rules_module.MAX_ENTRIES[field]

    assert parse_rules(json.dumps({field: build(limit)})) is not None
    error = rejected(json.dumps({field: build(limit + 1)}))
    assert locations(error) == [(field,)]


def test_the_cap_values_are_pinned() -> None:
    assert rules_module.MAX_ENTRIES == {
        "numeric_rules": 20,
        "date_rules": 20,
        "missing_value_overrides": 50,
        "missing_value_exempt_columns": 50,
        "duplicate_subset": 50,
        "pattern_columns": 50,
    }


VALUE_CASES = [
    ('{"numeric_rules": {"q": {}}}', ("numeric_rules", "<column>")),
    ('{"numeric_rules": {"q": {"min_value": 5, "max_value": 1}}}', ("numeric_rules", "<column>")),
    ('{"numeric_rules": {"q": {"min_value": "abc"}}}', ("numeric_rules", "<column>", "min_value")),
    ('{"numeric_rules": {"q": {"min_value": 1e999}}}', ("numeric_rules", "<column>")),
    ('{"numeric_rules": {"q": {"max_value": -1e999}}}', ("numeric_rules", "<column>")),
    ('{"numeric_rules": []}', ("numeric_rules",)),
    ('{"missing_value_threshold_percent": 101}', ("missing_value_threshold_percent",)),
    ('{"missing_value_threshold_percent": -1}', ("missing_value_threshold_percent",)),
    ('{"missing_value_overrides": {"q": 150}}', ("missing_value_overrides", "<column>")),
    ('{"duplicate_subset": []}', ("duplicate_subset",)),
    ('{"pattern_min_match_count": 0}', ("pattern_min_match_count",)),
]


@pytest.mark.parametrize(("text", "expected"), VALUE_CASES, ids=[f"case-{i}" for i in range(len(VALUE_CASES))])
def test_value_problems_are_located(text: str, expected: tuple[str, ...]) -> None:
    error = rejected(text)

    assert expected in locations(error)
    assert '"q"' not in everything(error)


VALID_FORMATS = [
    ("%Y-%m-%d", "2026-01-05"),
    ("%d/%m/%Y", "05/01/2026"),
    ("%Y-%m-%d %H:%M:%S", "2026-01-05 10:20:30"),
    ("%b %d, %Y", "Jan 05, 2026"),
]


@pytest.mark.parametrize(("date_format", "sample"), VALID_FORMATS)
def test_valid_date_formats_are_accepted(date_format: str, sample: str) -> None:
    config = parse_rules(json.dumps({"date_rules": {"d": {"date_format": date_format}}}))

    assert config is not None and config.date_rules["d"].date_format == date_format
    datetime.strptime(sample, date_format)  # and the format really works


UNSAFE_FORMATS = [
    "%Y%Y",
    "%Q",
    "%Z",
    "%c",
    "%",
    "%%",
    "%Y-%m-%d\n",
    "dd/mm/yyyy",
    "%Y-%m-%d%d",
    "%Y-%m-%d " + "x" * 40,
    "T",
    "--",
]


@pytest.mark.parametrize("date_format", UNSAFE_FORMATS, ids=[f"format-{i}" for i in range(len(UNSAFE_FORMATS))])
def test_unsafe_date_formats_are_rejected(date_format: str) -> None:
    error = rejected(json.dumps({"date_rules": {"d": {"date_format": date_format}}}))

    assert all(location[:2] == ("date_rules", "<column>") for location in locations(error))


@pytest.mark.parametrize("name", ["", "x" * 256, "a\nb"], ids=["empty", "too-long", "newline"])
def test_bad_column_names_are_rejected(name: str) -> None:
    assert ("numeric_rules", "<column>") in locations(rejected(json.dumps({"numeric_rules": {name: {"min_value": 0}}})))
    assert ("pattern_columns",) in locations(rejected(json.dumps({"pattern_columns": [name]})))


def test_a_255_character_column_name_is_accepted() -> None:
    name = "x" * 255

    assert parse_rules(json.dumps({"numeric_rules": {name: {"min_value": 0}}})) is not None


# =============================================================================================
# 2. Rules that name columns the file does not have
# =============================================================================================


def test_rule_columns_present_in_the_file_are_accepted() -> None:
    config = parse_rules(GOOD_RULES)
    assert config is not None

    check_rule_columns(config, ["sku", "quantity", "counted"])  # no error


ABSENT_CASES = [
    {"numeric_rules": {"SECRET-COLUMN": {"min_value": 0}}},
    {"date_rules": {"SECRET-COLUMN": {}}},
    {"missing_value_overrides": {"SECRET-COLUMN": 50}},
    {"missing_value_exempt_columns": ["SECRET-COLUMN"]},
    {"duplicate_subset": ["SECRET-COLUMN"]},
    {"pattern_columns": ["SECRET-COLUMN"]},
]


@pytest.mark.parametrize("case", ABSENT_CASES, ids=[next(iter(case)) for case in ABSENT_CASES])
def test_rule_columns_absent_from_the_file_are_rejected(case: dict[str, Any]) -> None:
    config = DetectionConfig.model_validate(case)

    with pytest.raises(InvalidRulesError) as excinfo:
        check_rule_columns(config, ["sku", "quantity"])

    assert locations(excinfo.value) == [(next(iter(case)),)]
    assert "SECRET-COLUMN" not in everything(excinfo.value)
    assert "not in the uploaded file" in excinfo.value.issues[0].message


def test_a_config_without_column_rules_passes_any_file() -> None:
    check_rule_columns(DetectionConfig(), ["anything"])  # no error


# =============================================================================================
# 3. Through the real API
# =============================================================================================


def test_a_scan_without_rules_uses_the_defaults(client: TestClient) -> None:
    response = post(client)

    assert response.status_code == 201
    assert response.json()["finding_count"] == 0


def test_valid_rules_activate_the_numeric_and_date_detectors(client: TestClient) -> None:
    response = post(client, rules=GOOD_RULES)

    assert response.status_code == 201
    assert response.json()["findings_by_category"] == {"numeric_rule_violation": 1, "invalid_date": 1}


@pytest.mark.parametrize("text", ["", "   ", "{}"], ids=["empty", "spaces", "empty-object"])
def test_blank_or_empty_rules_are_ignored(client: TestClient, text: str) -> None:
    response = post(client, rules=text)

    assert response.status_code == 201
    assert response.json()["finding_count"] == 0


def test_invalid_json_gets_a_422_and_stores_nothing(client: TestClient) -> None:
    response = post(client, rules="not json")

    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "validation_error"
    assert [issue["location"] for issue in body["issues"]] == ["body.rules"]
    assert dataset_total(client) == 0


def test_an_oversized_rules_field_gets_a_422(client: TestClient) -> None:
    response = post(client, rules=" " * (MAX_RULES_CHARS + 1))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert dataset_total(client) == 0


def test_a_rule_for_an_absent_column_gets_a_422_without_echo(client: TestClient) -> None:
    response = post(client, rules=json.dumps({"numeric_rules": {"SECRET-COLUMN": {"min_value": 0}}}))

    assert response.status_code == 422
    issue = response.json()["error"]["issues"][0]
    assert issue["location"] == "body.rules.numeric_rules"
    assert "not in the uploaded file" in issue["message"]
    assert "SECRET-COLUMN" not in response.text
    assert dataset_total(client) == 0


@pytest.mark.parametrize("key", HOSTILE_KEYS)
def test_hostile_keys_are_never_echoed(client: TestClient, key: str) -> None:
    response = post(client, rules=json.dumps({key: 1}))

    assert response.status_code == 422
    assert key not in response.text
    assert dataset_total(client) == 0


def test_an_unsafe_date_format_gets_a_422_not_a_500(client: TestClient) -> None:
    response = post(client, rules=json.dumps({"date_rules": {"counted": {"date_format": "%Y%Y"}}}))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert dataset_total(client) == 0


def test_the_csv_is_validated_before_the_rule_columns(client: TestClient) -> None:
    response = post(client, csv=b"", rules=GOOD_RULES)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "empty_file"  # the ingestion error wins
    assert dataset_total(client) == 0


def test_the_route_passes_the_parsed_config(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Any] = []
    original = scans_route.run_scan

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs.get("config"))
        return original(*args, **kwargs)

    monkeypatch.setattr(scans_route, "run_scan", spy)

    assert post(client).status_code == 201
    assert post(client, rules=GOOD_RULES).status_code == 201

    assert captured[0] is None
    assert isinstance(captured[1], DetectionConfig) and "quantity" in captured[1].numeric_rules


def test_the_openapi_document_lists_the_rules_field(client: TestClient) -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    assert any("rules" in schema.get("properties", {}) for schema in schemas.values())


def test_rules_are_not_stored(client: TestClient) -> None:
    result = post(client, rules=GOOD_RULES).json()

    scan = client.get(f"/scans/{result['scan_run_id']}").json()
    stored = json.dumps(scan)
    assert "numeric_rules" not in stored and "min_value" not in stored and "date_format" not in stored


# =============================================================================================
# 4. Structure
# =============================================================================================


def test_the_rules_module_imports_only_what_it_needs() -> None:
    tree = ast.parse((ROOT / "app" / "schemas" / "rules.py").read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)

    allowed_top = {"json", "math", "collections", "typing", "pydantic", "app"}
    for module in modules:
        assert module.split(".")[0] in allowed_top, module
        if module.startswith("app"):
            assert module == "app.schemas.findings", module