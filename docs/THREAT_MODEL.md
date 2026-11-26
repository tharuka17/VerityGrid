# VerityGrid threat model

A concise model of what VerityGrid protects, who might attack it, and what is and is not done about it. It reflects the code as built and tested on 2026-11-22. **It is a design review by the author, not an audit, a penetration test or a compliance statement.** Details of each control and its tests are in [SECURITY.md](../SECURITY.md).

## 1. Assets

| # | Asset | Why it matters | Where it lives |
|---|---|---|---|
| A1 | The Gemini API key | Account abuse and cost | The developer's local `.env` only. Never in an image, a log or a response. |
| A2 | Uploaded file contents | Could be sensitive if someone ignores the synthetic-data rule | In memory while a request is processed. Never written to disk or stored. |
| A3 | Stored free text | Column names, dataset names, review comments and reviewer names can hold sensitive strings | SQLite |
| A4 | Integrity of findings, review decisions and the audit trail | The point of a governance tool | SQLite |
| A5 | Availability and host resources | Memory, disk, CPU | The process or container |
| A6 | Model-provider quota and cost | Limited free quota, possible spend | The Gemini project |
| A7 | Credibility of the evaluation report | Portfolio claims rest on it | `reports/`, fingerprints, the committed snapshot |
| A8 | Detection rules supplied with an upload | Untrusted configuration that controls how much work a scan does | Request memory only; never stored |

## 2. Actors and trust boundaries

| Actor or input | Trust |
|---|---|
| Anonymous visitor to a public read-only demo | Untrusted |
| Anyone who can reach the API while it runs in development mode | Untrusted, and able to write |
| Whoever uploads a CSV (and rules) in development mode | Untrusted file contents and rules |
| The model provider's output | Untrusted data, never instructions |
| Dependencies and the base image | Not verified |
| The developer's machine | Trusted |

Boundaries: browser to the Streamlit server; Streamlit server to the API (server-to-server HTTP); the API to the database; the workflow to the model provider (reachable only from code that calls the workflow directly, **never from an HTTP request**); the reports folder; the build pipeline.

## 3. Threats, controls and what remains

| # | Threat | Controls built | Evidence | Residual risk |
|---|---|---|---|---|
| T1 | Anyone writes data through a public API | Demo mode (the default) rejects every method except GET, HEAD and OPTIONS with 403 `read_only_mode`, in the outermost middleware, before any body is read. A missing or invalid mode means demo. Development mode is refused when `ENVIRONMENT` is production. | `test_security_access.py`, `test_security_hardening.py` | **No authentication exists.** Development mode is fully open and is for local use only. |
| T2 | Resource exhaustion through oversized or malformed uploads | Limits (5 MiB, 50,000 rows, 200 columns); a `Content-Length` check; a counter of the bytes that actually arrive; a bounded read; strict ingestion rules. | `test_security_hardening.py`, Phase 5 tests | No rate limiting, no cap on concurrent uploads, slow-body attacks, and a large body may be spooled to a temporary file by the parser before the limit applies. |
| T3 | SQL injection | ORM access throughout; filters and ids validated as enums or bounded integers; hostile strings sent through every input. | `test_security_access.py` | A passing test is not proof; any future raw SQL would need review. |
| T4 | Markup injection in the dashboard | Untrusted text is shown with `st.text` or inside tables, never as markup; a static test forbids `unsafe_allow_html`, `st.write` and built `st.markdown` text; the API returns JSON only. | `test_ui_api_client.py` static tests | Vulnerabilities in Streamlit or the browser are not covered. |
| T5 | Leaking the API key | `SecretStr`; the dashboard reads only `VERITYGRID_*` variables; `.dockerignore` excludes `.env`; the images set no key; logs hold identifiers and class names; secret patterns are scanned in packaging, documentation and the report snapshot. | Sentinel tests, `test_packaging_docs.py`, `test_demo_snapshot.py`, the image check (no key-like variable) | The key sits in a OneDrive-synced local `.env`. Rotate it before the repository is published. |
| T6 | Sensitive data in logs, findings, responses or model requests | Findings store counts, rule names and safe evidence, never cell values; model requests hold sanitized metadata only; error messages never echo input, including rejected rules. | Sentinel tests in Phases 5 to 9; the seeded database holds no cell values | Column names, dataset names and comments are stored as typed and can themselves be sensitive. `DEBUG=true` writes tracebacks that can contain values (forced off in production). |
| T7 | Prompt injection through column names | Names are sanitized and sent as structured data; output is validated against a strict schema and a reference list; the classifier, not the provider, sets provenance; policy can only raise review requirements; results are labelled unverified; a human reviews. | Phase 6 tests with hostile names | A model can still be nudged to a wrong label. The real-model evaluation used no hostile names, so it is untested against a live model. HTTP cannot reach this path. |
| T8 | Excessive model calls and cost | No HTTP request can request classification; the adapter retries at most 3 times; the workflow classifies at most 50 columns and stops after 3 consecutive failures; the evaluation commands use a real model only on request, with a confirmation, and the paced command waits between calls and stops after repeated failures. | `test_the_upload_route_never_passes_a_classifier`, Phase 6 and 7 tests, `test_llm_run.py` | A first unpaced attempt hit the free-tier rate limit and was stopped. The paced run of 2026-11-19 used about 80 requests (including a trial) against a daily limit of 500. |
| T9 | Invalid review state or a tampered audit trail | The decision is an enum; history is append-only; the status always equals the latest decision; a review writes decision, status and audit event in one commit with rollback on failure; audit metadata holds no comment text. | `test_review_status_always_equals_the_latest_decision` and others | Reviewer names are self-reported. Concurrent reviews are not serialised on PostgreSQL. Nothing signs or protects the audit rows from someone with database access. |
| T10 | Misconfiguration | Fail-closed mode; the `ENVIRONMENT` guard; `DEBUG` forced off in production; the demo refuses to start without its database file; the mode announced at startup; a deployment checklist. | `test_security_hardening.py`, `test_startup_safety.py` | The checks cover the mode, `DEBUG` and the database file only; other settings are not validated against each other. |
| T11 | Information disclosure through public documentation and reports | None beyond the data being synthetic. | none | `/docs`, `/redoc` and `/openapi.json` are public in every mode. The evaluation report includes the generating machine's operating-system string, and review counts when generated without `--skip-review`. |
| T12 | Supply chain (a dependency or the base image) | Official `python:3.14-slim` base; dependencies installed from `requirements.txt`. | none | **No vulnerability scan has been run.** A tracing package (`langsmith`) is installed as a dependency; tracing is opt-in through environment variables and must stay unset. |
| T13 | Container or host compromise | Non-root user; all Linux capabilities dropped; no-new-privileges; ports bound to loopback in Compose. | The Docker checks of 2026-11-19 | No read-only root filesystem, no seccomp or image scanning. |
| T14 | Denial of service against a public demo | Only what the hosting platform provides. | none | A flood, or a sleeping free instance, makes the demo unavailable. |
| T15 | People over-trusting an AI label | Labels are advisory and shown as unverified; the mock is shown as a placeholder; review is mandatory for sensitive labels; the evaluation reports intervals, a baseline, coverage and a paired comparison with the mock. | Classifier and UI tests, `test_packaging_docs.py` | Human over-reliance is a behaviour, not something code removes. The real model's result (0.794, one run) is not significantly better than the mock. |
| T16 | Hostile detection rules (a development-mode upload with crafted rules) | Size limit, strict JSON, entry caps, finite bounds, printable bounded names, a date-format allowlist (each directive once), rejection of columns the file lacks, no echo, never stored. | `test_scan_rules.py` | Many date rules on a large column are slow (estimated, not measured). Rules cannot reach the demo, which accepts no writes. |
| T17 | A slow regular expression on a hostile cell value | The email pattern is anchored; both patterns are tested on 300,000-character values with a time limit. | `test_detector_robustness.py` | Only the two patterns and the sizes in the tests were measured. |
| T18 | A stale or tampered evaluation snapshot | The committed report must pass the service's checks, its fingerprints and its deterministic and mock results must equal what the code produces, and its counts must be internally consistent. | `test_demo_snapshot.py` | The real-model section is a record, not reproducible; the tests cannot prove it was not edited, only that it is consistent. |

## 4. Deployment posture

- **Public:** only the read-only, synthetic-data demo, following the checklist in [SECURITY.md](../SECURITY.md) and [DEPLOYMENT.md](DEPLOYMENT.md). It has no writes, no key and no model calls.
- **Local development:** development mode is acceptable on a machine you control.
- **Not recommended:** any public deployment with uploads or reviews. That needs real access control (users, roles, rate limiting, tests) that does not exist.

## 5. Out of scope

Real personal data, multi-tenant isolation, regulatory compliance, encryption at rest, backup and recovery, secrets management beyond environment variables, and the security of the hosting platform.

## 6. Assumptions

Only synthetic data is uploaded. Nobody other than the developer can reach a development-mode server. The platform provides TLS and rate limiting for any public deployment.

## 7. When to revisit this model

When authentication is added, when classification is exposed over HTTP, when a real database is used instead of SQLite, when real data is ever involved, when a different model is evaluated, or when a dependency scan or penetration test is run.