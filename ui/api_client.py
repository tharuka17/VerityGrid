"""HTTP client for the VerityGrid API, plus the dashboard's own settings.

This module has no Streamlit import, so it can be tested directly. It is the ONLY place the
dashboard talks to the backend: every number or row shown in the UI comes from a call made here.

Failure handling. Every problem becomes an ApiError with a kind the UI can explain:
  UNREACHABLE       the API cannot be contacted (not running, wrong URL, connection refused)
  TIMEOUT           the API accepted the connection but answered too slowly
  HTTP_ERROR        the API answered with an error (carries the API's own code and message)
  INVALID_RESPONSE  the answer was not the JSON shape this client expects

Messages are safe to show: they come from this module or from the API's error shape, which
never contains uploaded content. The client sends no credentials; the API currently has no
authentication, so none are needed (and no secret may be placed in UI code).
"""

from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

import requests
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_API_URL = "http://127.0.0.1:8000"
USER_AGENT = "veritygrid-dashboard"

_METRICS_KEYS = (
    "dataset_count",
    "scan_count",
    "scans_by_status",
    "finding_count",
    "findings_by_severity",
    "findings_by_category",
    "findings_by_review_status",
    "pending_high_severity_count",
    "review_decision_count",
    "reviewed_finding_percent",
)
_DATASET_KEYS = (
    "id",
    "name",
    "row_count",
    "column_count",
    "created_at",
    "scan_count",
    "finding_count",
    "pending_review_count",
    "latest_scan_id",
    "latest_scan_status",
)


class UiSettings(BaseSettings):
    """Dashboard settings from VERITYGRID_* environment variables or the project's .env file.

    Only variables with the VERITYGRID_ prefix are read and everything else in .env is ignored,
    so the dashboard process never loads the Gemini key or any other backend secret.
    Example: VERITYGRID_API_URL=http://127.0.0.1:8000
    """

    api_url: str = DEFAULT_API_URL
    connect_timeout_seconds: float = Field(default=3.0, ge=0.5, le=30.0)
    read_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)

    model_config = SettingsConfigDict(
        env_prefix="VERITYGRID_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("api_url")
    @classmethod
    def _check_api_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        parsed = urlparse(cleaned)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("api_url must be an http or https URL")
        if parsed.username or parsed.password:
            raise ValueError("api_url must not contain credentials")  # it is shown in the UI
        if parsed.query or parsed.fragment:
            raise ValueError("api_url must not contain a query or fragment")
        return cleaned


class ApiErrorKind(StrEnum):
    UNREACHABLE = "unreachable"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    INVALID_RESPONSE = "invalid_response"


class ApiError(Exception):
    def __init__(
        self,
        kind: ApiErrorKind,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        issues: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status
        self.code = code
        self.issues = issues or []  # (location, rule text) pairs from a validation error


def _expect(payload: Any, kind: type, keys: tuple[str, ...], what: str) -> Any:
    if not isinstance(payload, kind) or any(key not in payload for key in keys):
        raise ApiError(
            ApiErrorKind.INVALID_RESPONSE,
            f"The API returned an unexpected response for {what}.",
        )
    return payload


def _page(payload: Any, what: str) -> dict[str, Any]:
    page = _expect(payload, dict, ("items", "total"), what)
    if not isinstance(page["items"], list) or not isinstance(page["total"], int):
        raise ApiError(
            ApiErrorKind.INVALID_RESPONSE,
            f"The API returned an unexpected response for {what}.",
        )
    return page


def _clean(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


class ApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float = 3.0,
        read_timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = (connect_timeout, read_timeout)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._session = session

    @property
    def base_url(self) -> str:
        return self._base_url

    # --- Transport -------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        try:
            response = self._session.request(
                method,
                url,
                params=_clean(params or {}),
                json=json_body,
                files=files,
                data=data,
                timeout=self._timeout,
            )
        except requests.ConnectTimeout:  # a subclass of both ConnectionError and Timeout
            raise ApiError(
                ApiErrorKind.UNREACHABLE, "Could not connect to the API in time."
            ) from None
        except requests.Timeout:
            raise ApiError(
                ApiErrorKind.TIMEOUT, "The API took too long to respond."
            ) from None
        except requests.ConnectionError:
            raise ApiError(ApiErrorKind.UNREACHABLE, "Could not connect to the API.") from None
        except requests.RequestException:
            raise ApiError(ApiErrorKind.UNREACHABLE, "The request to the API failed.") from None

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.status_code >= 400:
            raise self._http_error(response.status_code, payload)
        if not isinstance(payload, (dict, list)):
            raise ApiError(
                ApiErrorKind.INVALID_RESPONSE,
                "The API returned a response that is not valid JSON.",
                status=response.status_code,
            )
        return payload

    @staticmethod
    def _http_error(status: int, payload: Any) -> ApiError:
        """Read the API's shared error shape: {"error": {"code", "message", "issues"?}}."""
        error = payload.get("error") if isinstance(payload, dict) else None
        if (
            isinstance(error, dict)
            and isinstance(error.get("code"), str)
            and isinstance(error.get("message"), str)
        ):
            issues = [
                (str(item.get("location", "")), str(item.get("message", "")))
                for item in error.get("issues") or []
                if isinstance(item, dict)
            ]
            return ApiError(
                ApiErrorKind.HTTP_ERROR,
                error["message"],
                status=status,
                code=error["code"],
                issues=issues,
            )
        return ApiError(
            ApiErrorKind.HTTP_ERROR,
            f"The API returned HTTP {status}.",
            status=status,
            code="unexpected_error_body",
        )

    # --- System, metrics, datasets, scans -----------------------------------------

    def health(self) -> dict[str, Any]:
        return _expect(self._request("GET", "/health"), dict, ("status",), "the health check")

    def metrics_summary(self) -> dict[str, Any]:
        return _expect(
            self._request("GET", "/metrics/summary"), dict, _METRICS_KEYS, "the metrics summary"
        )

    def list_datasets(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        payload = self._request("GET", "/datasets", params={"limit": limit, "offset": offset})
        return _page(payload, "the dataset list")

    def get_dataset(self, dataset_id: int) -> dict[str, Any]:
        payload = self._request("GET", f"/datasets/{int(dataset_id)}")
        return _expect(payload, dict, _DATASET_KEYS, "a dataset")

    def get_columns(self, dataset_id: int) -> dict[str, Any]:
        payload = self._request("GET", f"/datasets/{int(dataset_id)}/columns")
        columns = _expect(payload, dict, ("dataset_id", "row_count", "columns"), "column profiles")
        if not isinstance(columns["columns"], list):
            raise ApiError(
                ApiErrorKind.INVALID_RESPONSE, "The API returned an unexpected response for column profiles."
            )
        return columns

    def get_scan(self, scan_id: int) -> dict[str, Any]:
        payload = self._request("GET", f"/scans/{int(scan_id)}")
        return _expect(payload, dict, ("id", "dataset_id", "status", "started_at", "summary"), "a scan")

    def upload_scan(
        self,
        filename: str,
        content: bytes,
        *,
        dataset_name: str | None = None,
        rules: str | None = None,
    ) -> dict[str, Any]:
        """Upload a CSV and run a scan. The content is sent as is and is never displayed.

        `rules` is optional JSON text with detection rules for this scan only. The API validates it and
        rejects it with a 422 (location body.rules...) if it is invalid.
        """
        form: dict[str, str] = {}
        if dataset_name:
            form["dataset_name"] = dataset_name
        if rules:
            form["rules"] = rules
        payload = self._request(
            "POST",
            "/scans",
            files={"file": (filename, content, "text/csv")},
            data=form or None,
        )
        return _expect(
            payload,
            dict,
            ("dataset_id", "scan_run_id", "status", "finding_count", "findings"),
            "the scan result",
        )

    # --- Findings and reviews --------------------------------------------------------

    def list_findings(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        dataset_id: int | None = None,
        scan_id: int | None = None,
        severity: str | None = None,
        review_status: str | None = None,
        category: str | None = None,
        detection_method: str | None = None,
    ) -> dict[str, Any]:
        payload = self._request(
            "GET",
            "/findings",
            params={
                "limit": limit,
                "offset": offset,
                "dataset_id": dataset_id,
                "scan_id": scan_id,
                "severity": severity,
                "review_status": review_status,
                "category": category,
                "detection_method": detection_method,
            },
        )
        return _page(payload, "the finding list")

    def get_finding(self, finding_id: int) -> dict[str, Any]:
        payload = self._request("GET", f"/findings/{int(finding_id)}")
        return _expect(payload, dict, ("id", "review_status", "review_decisions"), "a finding")

    def review_finding(
        self,
        finding_id: int,
        decision: str,
        *,
        comment: str | None = None,
        reviewer: str | None = None,
    ) -> dict[str, Any]:
        """Record an approve or reject decision. This only changes review records; the API never
        remediates anything."""
        body: dict[str, Any] = {"decision": decision}
        if comment:
            body["comment"] = comment
        if reviewer:
            body["reviewer"] = reviewer
        payload = self._request("POST", f"/findings/{int(finding_id)}/review", json_body=body)
        return _expect(payload, dict, ("decision", "finding"), "the review result")
    

    # --- Evaluation ----------------------------------------------------------------------------

    def evaluation_report(self) -> dict[str, Any]:
        """The latest generated evaluation report, or the reason there is none.

        The API only serves the file written by the evaluation command; it never runs an evaluation.
        """
        payload = self._request("GET", "/evaluation/report")
        answer = _expect(
            payload, dict, ("available", "reason", "file_modified_at", "report"), "the evaluation report"
        )
        available, report = answer["available"], answer["report"]
        if (
            not isinstance(available, bool)
            or (available and not isinstance(report, dict))
            or (not available and report is not None)
        ):
            raise ApiError(
                ApiErrorKind.INVALID_RESPONSE,
                "The API returned an unexpected response for the evaluation report.",
            )
        return answer