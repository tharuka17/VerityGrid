# Security notes for VerityGrid

VerityGrid is an independent portfolio prototype that uses synthetic data only. It is not affiliated with Sysco or Sysco LABS. This file describes what the code does and does not protect against. **It is not a security certification, an audit report or a compliance statement, and nothing here makes the application enterprise-grade.** Every control listed below is tied to a test; every known gap is listed as such.

## Two ways to run it

| Mode | `VERITYGRID_MODE` | Who it is for | What is allowed |
|---|---|---|---|
| **development** | `development` | One developer on a local machine | Uploads, scans and reviews are OPEN to anyone who can reach the server. No authentication. |
| **demo** (the default) | `demo`, unset or invalid | Any deployment other people can reach | Read-only: every HTTP method other than GET, HEAD and OPTIONS is rejected with 403 `read_only_mode`. |

The mode is read once at startup (restart the API after changing it) and announced in the log. An invalid value means demo. Development is **refused** (demo is used instead, with an error log) when `ENVIRONMENT` is `production` or `prod`, because the two settings are separate and a mismatch must not open public writes. `DEBUG` is likewise forced off when `ENVIRONMENT` says production.

**In demo mode the API refuses to start** when the SQLite file named by `DATABASE_URL` does not exist, so a wrong path cannot silently create an empty database and make a broken demo look healthy. Other databases and development mode are not checked.

**Demo mode is a method filter, not authentication.** It has no users, tokens or roles, so it cannot tell who is asking. It is the smallest public surface that can be explained and tested: nothing can be written, so there is nothing to authenticate. A deployment that needs people to upload or review needs real access control, which does not exist yet.

## Running the read-only demo

From the project folder, with the virtual environment active (a cloud deployment sets the same values as environment variables instead):

```powershell
python -m scripts.seed_demo
$env:DATABASE_URL = "sqlite:///./veritygrid_demo.db"
$env:VERITYGRID_MODE = "demo"
$env:ENVIRONMENT = "production"
uvicorn app.main:app
```

Or `docker compose up --build`, which does the same inside a container (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)). Checklist for anything other people can reach:

1. Do not ship `.env`. It holds the developer's Gemini key. Set configuration through the platform.
2. `VERITYGRID_MODE=demo` (the default) and `ENVIRONMENT=production`.
3. `AI_PROVIDER=mock` and no `GEMINI_*` values. The API cannot cause a model call (see below), but there is no reason to hold a key there.
4. Point `DATABASE_URL` at the seeded demo database (synthetic data only), not at the development database.
5. The evaluation report is a committed snapshot (`data/evaluation/llm_run_2026-11-19.json`) of one real-model run on synthetic data; it contains no secret and the machine's operating-system string.
6. Put TLS and rate limiting in front of it. The application provides neither.
7. Set `VERITYGRID_API_URL` for the dashboard, which has no login of its own.

## Controls, and the tests behind them

| Risk | Control | Tests |
|---|---|---|
| Public unauthenticated writes | The read-only middleware is outermost, so it runs before any body is read, covers routes added later and routes that do not exist, and fails closed. | `test_security_access.py`, `test_security_hardening.py` (mode resolution, every unsafe method, reads still work, check runs before the upload checks) |
| Misconfiguration | Fail-closed mode; development refused in production; `DEBUG` forced off in production; the demo refuses to start without its database file; the mode is announced at startup. | `test_security_hardening.py`, `test_startup_safety.py` |
| Oversized uploads | Defaults: 5 MiB, 50,000 rows, 200 columns. A declared `Content-Length` over the limit gets 413 before any read; the middleware also counts the bytes that actually arrive (a body with no or a false length is stopped as soon as it passes the limit plus a 64 KiB multipart allowance); the route's bounded read and the ingestion limits remain. | `test_security_hardening.py` (nine middleware tests and one through the real app); earlier upload tests |
| Malformed uploads | UTF-8 only, NUL bytes, ragged rows, duplicate or unsafe headers and header-only files are rejected with stable codes. | Phase 5 tests |
| Slow or hostile cell values | The email pattern is anchored so a long run of characters costs linear, not quadratic, time (20,000 characters took 2.6 s with the old pattern, measured). The phone pattern has bounded quantifiers. | `test_detector_robustness.py` (exhaustive equivalence with the old pattern on short strings; 300,000-character values scanned in bounded time) |
| Hostile detection rules | The optional `rules` field is size-limited (20,000 characters), parsed with a strict JSON parser, capped (20 numeric and 20 date rules, 50 entries elsewhere), restricted (finite numeric bounds, printable column names, an allowlist for date formats with each directive at most once, because a repeated directive makes `strptime` raise an exception the detector does not catch), checked against the file's columns, never echoed in an error, and never stored. | `test_scan_rules.py` (86 tests) |
| Unsafe file handling | The application never writes an upload; filenames are reduced to their last path component and stripped of control characters; the report path is fixed in code. | Phase 5 and 9 tests |
| SQL injection | Database access goes through the ORM; filters and ids are validated as enums or bounded integers. | Hostile strings through names, comments, query values and path ids in `test_security_access.py` |
| Improper error handling | One error shape; validation errors carry a location and a rule text, never the submitted value; unexpected errors return a generic message; tracebacks are logged only when `DEBUG` is on. | Phase 5 and 10 tests |
| Credential leakage | `SecretStr` for the key; the dashboard's settings read only `VERITYGRID_*` variables; logs hold identifiers, counts and exception class names. | Sentinel tests in several phases; `test_backend_secrets_never_appear_in_responses` |
| Raw data in logs, findings, requests, responses | Uploaded rows are never stored; findings hold counts and rule names; a model request holds sanitized column metadata only. | Sentinel tests in Phases 5 to 9; `test_the_seeded_database_holds_no_cell_values` |
| Prompt injection through metadata | Column names are sanitized and sent as structured data, outputs are validated against a schema and a reference list, policy only raises review requirements, and results are labeled "unverified". | Phase 6 tests (hostile names, no cell values sent) |
| Unbounded retries and model calls | The provider adapter retries at most 3 times; the workflow classifies at most 50 columns and stops after 3 consecutive failures; **the HTTP API never requests classification.** The paced evaluation command adds a pause between calls and a run-wide stop after repeated failures. | Phase 6 and 7 tests; `test_the_upload_route_never_passes_a_classifier`; `test_the_api_layer_imports_no_provider_code`; `test_llm_run.py` |
| Invalid review state | The decision must be `approved` or `rejected`; history is append-only; the status always equals the latest decision. | `test_review_status_always_equals_the_latest_decision`; invalid decisions change nothing |
| Transactions | A scan commits its results once and records failures separately; a review writes the decision, the status and the audit event in one commit, with rollback on any error; SQLite foreign keys are enabled. | Phase 5 tests (rollback, audit consistency) |
| Secrets in packaging | `.dockerignore` excludes `.env`, databases and reports; no key-like names in the Docker and Render files; secret patterns are scanned in every packaging and documentation file and in the committed report snapshot. | `test_packaging_docs.py`, `test_demo_snapshot.py` |

## Known gaps and remaining risks

- **No authentication, authorization or user accounts.** Development mode is fully open. Reviewer names are self-reported and stored as typed.
- **No rate limiting, TLS or request throttling.** The deployment platform must supply them. Slow-body attacks and many concurrent uploads (each held in memory while parsed) are not addressed.
- **Starlette may spool a large multipart body to a temporary file** before the route runs. The byte counter bounds that to the configured limit, but the temporary file lives on the host's disk.
- **The API documentation pages (`/docs`, `/redoc`, `/openapi.json`) are public in every mode.**
- **Rules cost work.** Each date rule runs one `strptime` call per distinct value, so a development-mode upload with 20 date rules on a large column is slow (estimated at several seconds, not measured). Demo mode never reaches this code.
- **The upload limit is read in two places** (the middleware reads the settings; the route reads a dependency built from the same settings). They agree in normal operation; a test override of only the dependency makes them differ.
- **Concurrent reviews are not serialized on PostgreSQL.** On SQLite writes are serialized, so the status equals the latest decision. On PostgreSQL a race could break that, and a row lock would be needed first.
- **Free-text fields** (column names, dataset names, review comments, rule column names) can hold sensitive strings. They are stored as typed, except that rules are never stored. Column names would be sent, sanitized, to a third party if classification with a real provider were ever enabled over HTTP.
- **Prompt injection is mitigated, not prevented.** A hostile column name can still nudge a model toward a wrong label. Human review and the "unverified" labeling are the control. The real-model evaluation used no hostile column names, so it has not been tried against a live model.
- **A read timeout from the provider is recorded as an error, not as "provider unavailable".** It still leaves the column unanswered and counts toward the stop rule.
- **The dashboard has no login.** In a public deployment it must only be pointed at a read-only API with synthetic data. Streamlit's own server surface has not been reviewed.
- **The evaluation report is served as written**, including the operating system string of the machine that generated it, and review counts if it was generated without `--skip-review`.
- **No dependency vulnerability scan, secret scan of the Git history or penetration test has been run.** A `langsmith` package is installed as a dependency of an orchestration library; tracing integrations are opt-in through environment variables and should stay unset.
- **The developer's `.env` holds a real API key in a OneDrive-synced folder.** Rotate the key before the project is ever published, and make sure `.env` is git-ignored.
- **The containers** run as an unprivileged user with all capabilities dropped, but have no read-only root filesystem, seccomp profile or image scan.

## What this review did not cover

The review read the files named in its history: the API application, middleware, error handlers, settings, the session module, the review service, the scans route, the schemas, the dependencies, the detectors, the detection-config schema, the scan service, the dashboard client and its upload view, and the evaluation and packaging files through their tests. It did not read the ORM models, the query service, the ingestion module, the profiler, the AI layer's source (it was reviewed through its tests), the workflow's source or the remaining dashboard views.