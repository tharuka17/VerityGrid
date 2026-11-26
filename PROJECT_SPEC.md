# VerityGrid — Project Specification

**Tagline:** AI-Assisted Data Governance & Reliability Workbench
**Document status:** Revised on 2026-11-25 to describe the system **as built**. The Phase 0 draft described a plan; where the build differs from it, this document records the build and section 14 lists the differences. The detailed verification log is IMPLEMENTATION_STATUS.md. Related documents: [README.md](README.md), [SECURITY.md](SECURITY.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md), [docs/EVALUATION.md](docs/EVALUATION.md) and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## 1. Purpose and boundaries

VerityGrid is an independent enterprise-AI portfolio prototype. It demonstrates how deterministic data processing, AI-assisted classification and a controlled workflow can help a data team understand datasets, find sensitive fields and quality problems, and review findings, while preserving evidence, auditability and human oversight.

**Boundaries**

- It is not a Sysco product and is not affiliated with Sysco or Sysco LABS. It must never be presented as an implementation of their internal systems.
- It uses completely synthetic foodservice-distribution data (customers, orders, inventory and similar).
- It is a prototype: it suggests, records and prioritizes. It never deletes records or changes real permissions.
- It is built to be explainable in a technical interview, so readability is preferred over cleverness.
- It has **no authentication**. A public deployment is limited to a read-only demo of synthetic data (section 9). It has not been deployed anywhere.
- Nothing in this project is a security certification or a compliance claim.

**Out of scope:** React, Kubernetes, vector databases, RAG, message brokers, multiple autonomous agents, external data connectors, real customer data, user accounts and roles, and background job queues.

---

## 2. Problem statement

Enterprise data teams must understand the datasets they hold, identify potentially sensitive information, detect data-quality problems, assign classifications and investigate policy violations. Doing this manually is slow and inconsistent.

VerityGrid shows a controlled alternative: deterministic checks produce evidence, an optional language model suggests contextual classifications from metadata only, a bounded workflow consolidates and prioritizes findings, and a human approves or rejects each recommendation with an audit trail.

---

## 3. User journey as built

1. The user opens the dashboard (or the API documentation).
2. The user uploads a CSV, optionally with **detection rules** for that scan, or, in the read-only demo, browses pre-seeded synthetic data.
3. The API validates the file and the rules, then profiles the schema and contents in memory.
4. Deterministic rules detect missing values, duplicate rows, email and phone patterns, and, for the columns named in the rules, values outside numeric bounds and invalid dates.
5. Column classification by a mock or a real model is **optional and not reachable over HTTP**: it runs only from code that calls the workflow with a classifier (the evaluation commands and scripts).
6. A bounded workflow consolidates the findings and assigns column and dataset risk.
7. The user reviews each finding in the Review queue and approves or rejects it, with an optional comment.
8. Every review writes a decision record and an audit event in one transaction.
9. The dashboard's numbers come from persisted records through the API.
10. An evaluation harness compares the system with a fixed labelled synthetic dataset and writes a reproducible report that the API serves and the dashboard shows.

---

## 4. Technology stack

| Layer | Technology | Responsibility |
|---|---|---|
| Language | Python 3.14 (developed and tested on 3.14.2 only) | All backend, dashboard and evaluation code |
| API | FastAPI | Routes, request validation, dependency injection, OpenAPI documentation |
| Validation and settings | Pydantic v2, pydantic-settings | Schemas, model-output validation, environment configuration |
| Persistence | SQLAlchemy 2.x with SQLite | Relational storage through `DATABASE_URL`; PostgreSQL has not been tested |
| Data processing | pandas | CSV parsing, profiling, deterministic detection |
| AI | A provider interface, an offline mock provider and one optional Google Gemini adapter | Advisory column classification from sanitized metadata |
| Orchestration | LangGraph | An explicit, bounded, acyclic scan workflow |
| Dashboard | Streamlit and Requests | A pure HTTP client of the API |
| Testing | Pytest with FastAPI's test client | Unit, API, dashboard-client and evaluation tests |
| Packaging | Docker, Docker Compose, a Render Blueprint | A read-only demo; **no deployment has been performed** |

There are no database migrations: tables are created with `create_all`, and existing tables are never altered.

---

## 5. Repository layout

| Path | Purpose |
|---|---|
| `app/main.py` | Application entry point: middleware, error handlers, routers, `/health`, startup checks |
| `app/core/config.py` | Settings from environment variables and `.env` |
| `app/api/` | HTTP layer: `app/api/routes/scans.py`, `app/api/routes/datasets.py`, `app/api/routes/findings.py`, `app/api/routes/metrics.py`, `app/api/routes/evaluation.py`; `app/api/deps.py`, `app/api/errors.py`, `app/api/middleware.py` (upload size) and `app/api/access.py` (read-only mode) |
| `app/db/` | `app/db/models.py` (six ORM models) and `app/db/session.py` (engine, sessions, `init_db`) |
| `app/schemas/` | `app/schemas/findings.py` (detector output and detection configuration), `app/schemas/api.py` (API shapes) and `app/schemas/rules.py` (validation of the rules supplied with an upload) |
| `app/services/` | `app/services/ingestion.py`, `app/services/profiling.py`, `app/services/detectors.py`, `app/services/scan_service.py`, `app/services/query_service.py`, `app/services/review_service.py` and `app/services/evaluation_service.py` |
| `app/ai/` | Provider interface (`app/ai/provider.py`), the validating classifier (`app/ai/classifier.py`), the mock provider, the Gemini adapter and the factory (`app/ai/factory.py`) |
| `app/workflows/scan_graph.py` | The LangGraph scan workflow |
| `app/evaluation/` | `app/evaluation/dataset.py` (the fixed labelled dataset), `app/evaluation/metrics.py`, `app/evaluation/harness.py` (reproducible report) and `app/evaluation/llm_run.py` (paced real-model run) |
| `ui/` | `ui/streamlit_app.py` (entry point and Overview), `ui/api_client.py`, `ui/common.py` and `ui/views/` (catalog, upload, findings, review, evaluation) |
| `scripts/seed_demo.py` | Builds the synthetic demo database through the application's own services |
| `data/sample/` | Three synthetic CSV files |
| `data/evaluation/llm_run_2026-11-19.json` | A committed snapshot of one real-model evaluation run, served by the demo image |
| `tests/` | Automated tests, with shared fixtures in `tests/conftest.py`; the root `conftest.py` runs the suite in development mode |
| `docs/` | `docs/ARCHITECTURE.md`, `docs/THREAT_MODEL.md`, `docs/EVALUATION.md`, `docs/DEPLOYMENT.md` |
| Project root | `README.md`, `SECURITY.md`, `PROJECT_SPEC.md`, `requirements.txt`, `.env.example`, `.gitignore`, `.dockerignore`, `Dockerfile.api`, `Dockerfile.ui`, `docker-compose.yml`, `render.yaml` |

Generated files (`reports/`, the SQLite databases and `.env`) are not part of the repository content.

---

## 6. Data model

Six ORM tables. No raw data values are stored. Safe evidence means counts, percentages, rule names and column names; it never contains cell values.

| Entity | Fields |
|---|---|
| **Dataset** | id, name, source_filename, row_count, column_count, created_at, updated_at |
| **ColumnProfile** | id, dataset_id, column_name, inferred_type, null_count, distinct_count, classification_label, classification_source, classification_explanation (all three NULL when not classified) |
| **ScanRun** | id, dataset_id, status, started_at, finished_at, summary (counts only, plus a workflow block when classification ran or the workflow was not clean), error_details (a generic reason only) |
| **Finding** | id, dataset_id, scan_run_id, category, severity, column_name (NULL for dataset-level findings), title, description, safe_evidence, detection_method, recommendation, review_status, created_at |
| **ReviewDecision** | id, finding_id, decision, comment, reviewer (self-reported), created_at |
| **AuditEvent** | id, actor, event_type, entity_type, entity_id, safe_metadata |

**Relationships:** a Dataset has many ColumnProfiles and ScanRuns; a ScanRun has many Findings (and a Dataset has many Findings); a Finding has many ReviewDecisions, the latest of which drives its `review_status` and none of which is ever edited or removed. An AuditEvent refers to its entity loosely, by type and id, so it survives independently. Every upload creates a new Dataset and a new ScanRun.

**Enumerations**

- Finding categories: `missing_values`, `duplicate_rows`, `potential_email`, `potential_phone`, `numeric_rule_violation`, `invalid_date`.
- Severity: `low`, `medium`, `high`. Review status and decision: `pending`, `approved`, `rejected` (only `approved` and `rejected` can be submitted).
- Scan status: `pending`, `running`, `completed`, `failed`.
- Classification labels: `public`, `internal`, `personal`, `financial`, `unknown`. Review recommendation: `review_required`, `review_recommended`, `review_optional`. Classification source: `mock`, `llm`, `none` (`none` marks a failed attempt).

---

## 7. Data flow

```text
POST /scans (CSV file, optional dataset name, optional detection rules)
  → read-only mode check (demo mode rejects every write here)
  → upload size check (declared length, then the bytes that actually arrive)
  → bounded read → ingest_csv: UTF-8, headers, ragged rows, row and column limits
  → parse the rules (size, caps, strict JSON, date-format allowlist), then check
    that every column they name exists in the file
  → transaction 1: Dataset and a "running" ScanRun
  → workflow (in memory, no database): validate → profile → run six detectors
      → [only if a classifier was supplied] classify columns, capped, with a circuit breaker
      → consolidate → apply policy (column and dataset risk) → finalize
  → transaction 2: ColumnProfiles, Findings and the "completed" status, committed once
      (on failure: roll back, then a separate transaction marks the scan "failed")
  → review: one transaction writes the ReviewDecision, the new status and the AuditEvent
  → the dashboard reads everything through the API
```

The uploaded file is never stored; raw values exist only in memory during the scan. The evaluation path is separate: the fixed dataset goes through the same workflow, the harness writes a report file, the API serves that file read-only and the dashboard shows it.

---

## 8. Separation of responsibilities

| Concern | Owns | Must not |
|---|---|---|
| **Deterministic processing** (`app/services/`) | Validation, profiling, detection, severity rules, persistence, review | Depend on a language model |
| **AI assistance** (`app/ai/`) | A suggested label and explanation from sanitized metadata; validation of the answer; the policy that can only raise review requirements | See cell values, take actions, override deterministic results, claim its own provenance |
| **Workflow orchestration** (`app/workflows/scan_graph.py`) | Explicit state, bounded execution, optional classification, risk assessment | Contain business rules that belong in services; build a provider or read settings |
| **HTTP layer** (`app/api/`) | Validation of requests, one error shape, access mode, size limits | Contain query or business logic |
| **Dashboard** (`ui/`) | Presentation | Import backend code, open the database or read backend secrets |

Deterministic checks remain authoritative for deterministic tasks. Model output is a suggestion. Policy is enforced in ordinary application code.

---

## 9. Privacy, security and human-review requirements: how they are met

| Requirement | How it is met | Status |
|---|---|---|
| Only synthetic data in fixtures and tests | All samples, the evaluation dataset and the seed are synthetic; the upload view warns users | Met by convention, not enforced technically |
| Never send raw rows or sensitive values to a model | A request holds an opaque column reference, the sanitized column name, type, counts and finding summaries only | Met (column names are sent, sanitized) |
| Treat column names as untrusted | Sanitized, sent as structured data, answers validated against a strict schema and reference list, provenance set by the classifier, policy can only raise review requirements | Mitigated, not prevented |
| The model never executes anything or changes records | A provider returns a mapping only; there are no tools; nothing in the system deletes data | Met |
| Validate all model responses; handle failures safely | Strict validation; any failure becomes a labelled "no suggestion" fallback and never raises | Met |
| Bounded retries, timeouts and calls | The adapter retries at most 3 times with a timeout; the workflow classifies at most 50 columns and stops after 3 consecutive failures; the paced evaluation command adds pauses and a run-wide stop; **no HTTP request can request classification** | Met |
| No secrets or raw values in logs | Logs hold identifiers, counts and exception class names; error messages never echo input; tracebacks only when `DEBUG` is on, which is forced off in production | Met |
| Human approval and an audit trail | Findings start `pending`; each review is append-only and written with its audit event in one transaction; audit metadata never holds the comment text | Met (reviewer names are self-reported) |
| No unauthenticated public write operations | The default access mode is read-only: every method except GET, HEAD and OPTIONS is rejected with 403 before any body is read; development mode is refused when `ENVIRONMENT` says production; the demo refuses to start without its database | Met by a read-only demo. **No authentication exists**, and development mode is open |
| Secrets in the environment | `.env` is git-ignored and excluded from images; `.env.example` holds placeholders; packaging, documentation and the report snapshot are scanned for secret patterns | Met (the developer's real `.env` holds a key and is synchronised by OneDrive) |

The detailed controls, their tests and the remaining risks are in SECURITY.md and docs/THREAT_MODEL.md.

---

## 10. Testing and evaluation

**Automated tests.** More than 1,300 tests (the exact number is in IMPLEMENTATION_STATUS.md) cover ingestion, profiling, each detector, the detection rules, the workflow, the AI layer with fake providers, the API through a test client on a temporary database, the dashboard client and static rules on its source, the evaluation harness, the paced real-model command, the security controls and the packaging and documentation files. **No test needs a key or network access.** Documentation is tested against the code: numbers in documents are compared with generated reports, and stale statements are forbidden.

**Verification discipline.** Code is generated in chat, copied and run locally by the developer, and only locally reported results count as verified.

**Evaluation.** A fixed, deterministic, labelled synthetic dataset (13 cases, 63 labelled columns) is run through the real workflow. Three sources of evidence are reported separately and never combined: deterministic detectors, the mock provider (a keyword placeholder, not a language model) and a real model. Results carry precision, recall, F1, 95% intervals, a confusion matrix, a majority-class baseline, coverage and fingerprints. The reproducible parts carry a results fingerprint that must be identical on every run. The real-model run is a single, non-reproducible run on one model, and it was not distinguishable from the mock on this dataset. The headline results, with their limitations, are in docs/EVALUATION.md.

---

## 11. Development phases as built

Phases ran in order and each started only on the developer's instruction. The numbering below is the one used in the project. Completed items were verified by local test runs reported by the developer; the detailed log is IMPLEMENTATION_STATUS.md.

| Phase | Scope as built | State |
|---|---|---|
| 0 | Specification, directory plan, status tracking | Completed; this document was revised on 2026-11-25 |
| 1 | Skeleton: settings, FastAPI app, health endpoint | Completed |
| 2 | Database session and the six ORM models | Completed |
| 3 | Synthetic samples and column profiling | Completed |
| 4 | Six deterministic detectors with safe evidence | Completed |
| 5 | Safe CSV ingestion (5A), the scan service with persistence (5B), and the API with review and audit (5C) | Completed |
| 6 | AI classification: provider interface, mock provider, validating classifier, optional Gemini adapter | Completed (persisting every suggestion detail, called 6D, is parked) |
| 7 | LangGraph scan workflow, integrated into the scan service | Completed |
| 8 | Streamlit dashboard: Overview, Data catalog, Upload and scan, Findings explorer, Review queue, Evaluation | Completed |
| 9 | Evaluation: fixed dataset, metrics, harness and report (9A); report endpoint and dashboard view (9B); a paced real-model run and one completed run on 2026-11-19 (9C) | Completed |
| 10A | Security and reliability review: read-only demo mode, upload byte counting, startup checks, hostile-input tests, `SECURITY.md` | Completed |
| 10B | Portfolio packaging: README, architecture and threat model, evaluation account, Docker, Compose, Render Blueprint, deployment guide, documentation tests | Completed |
| 12 | Follow-up work: fail-fast demo startup and the `DEBUG` guard, a linear-time email pattern, detection rules over HTTP and in the dashboard, a demo seeded with sample rules and a real-model snapshot, documentation synchronisation and this specification | Completed except the final verification pass |
| 13 | Platform deployment | **Not started.** Instructions and a Blueprint exist; nothing has been deployed |

Git has not been initialised (as of the last report), and no licence has been chosen.

---

## 12. Key design decisions

1. **Deterministic first, AI advisory.** Rules produce findings; a model only suggests a label that a human can ignore. This keeps results explainable and testable.
2. **A thin web layer.** Routes contain no logic and let domain exceptions reach one set of handlers, so errors are uniform and never echo submitted data.
3. **An acyclic, bounded workflow.** A scan is a pipeline, not an agent loop; fatal and non-fatal steps, a column cap and a circuit breaker bound cost and failure.
4. **Classification is not reachable over HTTP.** It would send metadata to a third party, so it exists only for code that opts in. This is a security boundary, not a missing feature.
5. **A read-only public demo instead of access control.** A shared token is not a user identity and would leave a public write API open to abuse, so writes are disabled outright.
6. **The API serves a generated report instead of running an evaluation.** A web request can never start a batch job or cause a paid call.
7. **The dashboard is an HTTP client.** The same API serves the dashboard, scripts and tests, and a dashboard bug cannot touch the database.
8. **Detection rules apply to one upload and are never stored.** They are untrusted configuration, validated at the boundary with caps, an allowlist and a check that every named column exists, because the detectors would otherwise ignore a mistyped name and make bad data look clean.
9. **Three sources of evidence in the evaluation, never combined**, with provenance enforced in code, coverage reported next to accuracy, and a reproducible fingerprint for the parts that can be reproduced.
10. **SQLite and `create_all`.** The prototype stays self-contained; the cost is no migrations and no serialisation of concurrent reviews on other databases.
11. **The Docker image seeds itself at build time.** The read-only demo needs no persistent disk, and every container starts from identical, verifiable data.
12. **Documentation is tested.** Statements in documents are checked against generated reports and the code, so they cannot drift silently.

---

## 13. Known limitations and assumptions

- A prototype: not production-hardened and not validated for real enterprise data.
- **No authentication, authorization, rate limiting or TLS.** Development mode is open to anyone who can reach the server; demo mode is read-only.
- Synthetic data cannot capture the messiness of real data. Evaluation results reflect a small, imbalanced benchmark whose labels, detectors and mock were written by the same author; they describe behaviour on anticipated cases, not real-world performance.
- Regular-expression detection has false positives and false negatives: North American phone formats only, phone numbers stored as numbers are not scanned, a ten-digit identifier can look like a phone number, and placeholder text such as "N/A" is not counted as missing.
- The mock provider is a keyword matcher for reproducibility, not a measure of model quality. The real model was evaluated once, on one dataset without hostile column names; its result is not significantly better than the mock's, and three of 63 columns received no answer.
- Metadata-only context limits classification accuracy by design.
- Scans run synchronously inside the request; there is no job queue.
- SQLite is the only tested database. PostgreSQL compatibility is a design goal and is untested; concurrent reviews are not serialised there.
- Free-text fields (column names, dataset names, comments) are stored as typed and can contain sensitive strings.
- The project has been run on Windows and, for the read-only demo, in a Linux container. It has not been deployed to a platform.
- No dependency vulnerability scan or penetration test has been run.
- Authorship assumption: the developer reviews and understands all generated code.

---

## 14. Differences from the Phase 0 plan

| Phase 0 plan | As built |
|---|---|
| Python 3.11 or later | Python 3.14, tested on 3.14.2 only |
| Samples in data/samples | Samples in `data/sample/` |
| A folder app/workflow | `app/workflows/` |
| One schema file per concern (finding, scan, review, dashboard) | `app/schemas/findings.py`, `app/schemas/api.py` and `app/schemas/rules.py` |
| A base module and an audit service | The ORM base lives in `app/db/models.py`; the review service writes the audit event itself |
| An AI folder with providers, a sanitizer, prompts and a classification module | A provider interface, a validating classifier that also sanitizes and applies policy, a mock provider, a Gemini adapter and a factory, all in `app/ai/` |
| A logging module | Not built; logs go through the standard library and carry identifiers and class names only |
| A reviews route in the review phase and a dashboard endpoint late | Review is `POST /findings/{id}/review`, built with the API; the metrics endpoint was built with the API |
| One real model provider as an optional late phase | The Gemini adapter was built with the AI layer, and one paced real-model evaluation run was completed |
| Pages in a dashboard pages folder | Views in `ui/views/`, called from one entry point |
| Labelled data in data/evaluation | The labelled dataset is code (`app/evaluation/dataset.py`); `data/evaluation/` holds only the committed real-model snapshot |
| Evaluation as Phase 11; Docker, documentation and hardening as Phase 12; an access-controlled or read-only demo as Phase 13 | Evaluation is Phase 9; security review is Phase 10A; packaging and documentation are Phase 10B; a read-only demo was chosen and packaged, and nothing has been deployed |
| Tests with HTTPX | Tests use FastAPI's test client |
| An initial pytest configuration file | Not created; tests run with `python -m pytest` |
| Detection rules configured by an unspecified mechanism | An optional `rules` field on `POST /scans` and a rules box in the dashboard |
| An access-control option for the public demo | Not built; the read-only option was chosen |
| Alembic only if needed | Not used |

Phase numbers elsewhere in the repository follow the "as built" column.