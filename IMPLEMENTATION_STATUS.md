# VerityGrid — Implementation Status

**Overall status:** All planned work is implemented, tested and committed, and the read-only demo is deployed. The last full-suite result reported is **1396 passed, 1 warning** (2026-11-26, after commit `af688be`). The code is in a GitHub repository (private at the time of writing). On 2026-11-26 the demo was deployed to Render's free plan as two Docker web services (dashboard <https://veritygrid-ui.onrender.com>, API <https://veritygrid-api.onrender.com>). Both services are configured with `autoDeployTrigger: commit` and a build filter, and the dashboard's automatic deploy was observed working. This file is committed in the fifth commit; it replaces the longer earlier version, which had stale sentences.

## Environment

- Development uses Claude chat plus local VS Code. Files are created manually from code in chat; commands and tests run in Windows PowerShell, one at a time; only locally reported results count as verified. Two test runs were made against unsaved edits (2026-11-14 and 2026-11-25), so edits are saved before tests run.
- Python 3.14.2, FastAPI 0.143.2, LangGraph 1.2.20, google-genai 1.75.0, Streamlit 1.67.0, pandas 3.0.6 (reported 2026-11-25); `pip check` reported no broken requirements; no LangSmith or LangChain environment variables are set. FastAPI 0.137.0 and later keep included routers as `_IncludedRouter` objects, so route paths cannot be found in `app.routes`; the OpenAPI document is used instead.
- Docker works locally (images `veritygrid-api` and `veritygrid-ui` on `python:3.14-slim`; Compose publishes ports 8000 and 8501 on `127.0.0.1`).
- The project folder is under OneDrive. The developer's `.env` holds a working Gemini key and `VERITYGRID_MODE=development`; it is git-ignored. The Gemini project is on the Free tier (limits seen for Gemini 3.5 Flash Lite: 15 requests per minute, 250K tokens per minute, 500 requests per day).

## What exists

- **Backend:** FastAPI application with safe CSV ingestion, profiling, six deterministic detectors, a scan service with persistence, review and audit, one error shape, and read-only mode (the default) that rejects every method except GET, HEAD and OPTIONS before any body is read. Detection rules are accepted per upload (optional `rules` field) after strict validation.
- **AI layer:** provider interface, offline mock provider, validating classifier with a policy that can only raise review requirements, and an optional Gemini adapter. Classification is not reachable over HTTP by design.
- **Workflow:** a bounded, acyclic LangGraph scan workflow.
- **Dashboard:** six Streamlit views, an HTTP client only.
- **Evaluation:** a fixed, labelled, synthetic dataset (13 cases, 63 columns), a reproducible harness whose results fingerprint is `67077ce4bc4e8fd3324cbf163f6be21aeb698feb21ae84c85117c50c63bd8d2e` on every run (Windows, Linux container, tests), and a paced command for a real-model run.
- **Packaging and documents:** Dockerfiles, Compose, `render.yaml`, `README.md`, `SECURITY.md`, `PROJECT_SPEC.md` (revised to describe the system as built), `docs/` (architecture with six Mermaid diagrams, threat model T1 to T18, evaluation, deployment) and an MIT `LICENSE`. Tests check the documents against the code and the generated reports.

## Real-model result (one run, 2026-11-19, single and non-reproducible)

`gemini:gemini-3.5-flash-lite`: accuracy 0.794 (50 of 63; 95% interval [0.68, 0.88]); 60 of 63 columns answered (0.833 over the answered ones); macro F1 0.800. The keyword mock scored 0.698 ([0.58, 0.80]) and always answering `internal` 0.556. **Not distinguishable from the mock:** the intervals overlap; 12 columns were wrong only for the mock, 6 only for the model, 7 for both (an exact paired test gives a two-sided p of about 0.24, the assistant's calculation, optimistic because columns are not independent). Of the 10 answered errors, 8 labelled an internal column `personal` (4) or `unknown` (4), 1 labelled a public column `internal`, 1 labelled an internal column `public`. Three columns got no answer (two HTTP 503 responses on both attempts, one read timeout recorded as a provider error). Cost and tokens are not measured by the harness; the provider's usage page showed roughly 80 requests, 30 thousand input and 5 thousand output tokens for that day (approximate chart readings). A committed snapshot of the run (`data/evaluation/llm_run_2026-11-19.json`) is the evaluation report the API image serves.

## Git and deployment record

| Commit | Content | Render effect |
|---|---|---|
| `014506f` | Initial commit, 101 files | First deploy of both services (12:03 to 12:05 in the Render dashboard's times, the author's local time) |
| `2b9a025` | Deployment record, MIT licence | None (autoDeploy was off) |
| `74b2af3` | `render.yaml`: `autoDeployTrigger: commit` with a `buildFilter` per service | Blueprint sync succeeded (green); no deploy, as expected: the commit changed only files outside both filters |
| `af688be` | One comment line in `Dockerfile.ui` | Dashboard deployed automatically ("New commit via Auto-Deploy", started 4:14 PM, live 4:15 PM); API showed no new deploy |
| fifth | This file, a deployment-guide paragraph, two tests | Documents only; no deploy expected |

- **Git:** initialised 2026-11-26 (`git init -b main`); project-local author name and a GitHub no-reply address (the global address is unchanged); `core.autocrlf` false. Dry run listed 101 files, none over 200 KB; a key-pattern search found nothing; the real key's value was found in no file to be committed and in no commit.
- **First deploy:** the dashboard showed `API: http://api:8000` (the Compose default in `Dockerfile.ui`) and "API unreachable" until `VERITYGRID_API_URL` was set by hand to the API's public address and the dashboard was deployed again; each service was also deployed by hand once (API 12:28, dashboard 12:34). A changed environment variable reaches a service only on a new deploy.
- **After the first deploy** (reported 2026-11-26): the API's `/health` and `/metrics/summary` answered (3 datasets, 12 findings: medium 8, high 4; pending 10, approved 1, rejected 1; 16.67%); `POST /scans` returned HTTP 403 `read_only_mode` and the API log showed `Write request rejected in read-only mode: method=POST`; the dashboard showed API reachable, `VerityGrid (production)`, and the Overview; the developer checked the Evaluation page and the blue read-only message on Upload and Review; no out-of-memory line was seen; the dashboard log showed two clean stop-and-restart cycles of unknown cause; a hand-set `VERITYGRID_API_URL` survived the Blueprint sync. The developer reported no `GEMINI_*` variable in Render's environment tab (without saying which service).
- **Cold start:** after about 2.5 idle hours the dashboard showed "HTTP 502, unexpected_error_body" until the API woke; waking it with `/health` and refreshing fixed it. The documents tell visitors to open `/health` first.
- **Not tested:** an API-file commit deploying the API; a documents-only commit deploying nothing; memory use, uptime and load. The running demo lags behind documents-only commits by design.

## Phase tracker

| Phase | Scope | State |
|---|---|---|
| 0 to 4 | Specification, skeleton, database, profiling, detectors | Locally verified |
| 5 | Ingestion, scan service, API with review and audit | Locally verified |
| 6, 7 | AI classification; LangGraph workflow | Locally verified (6D, persisting every suggestion detail, parked) |
| 8 | Streamlit dashboard | Completed |
| 9 | Evaluation harness, report endpoint and view, paced real-model run | Completed |
| 10A | Security and reliability review | Locally verified |
| 10B | Packaging and documentation | Locally verified |
| 12 | Follow-up: fail-fast demo startup and `DEBUG` guard, linear-time email pattern, detection rules over HTTP and in the dashboard, seeded demo and snapshot, documentation sync, specification revision | Locally verified |
| 13 | Render deployment | Completed 2026-11-26 for the read-only demo |
| Git | Repository, remote, licence | Completed; making the repository public is pending (see open items) |

## Verification log (condensed; full counts)

| Date | Phase | Result |
|---|---|---|
| 2026-10-16 to 24 | 1 to 5 | Developer-reported working; 226 passed, then 329 after the API (103 tests); real-server walkthrough of all nine endpoints consistent |
| 2026-10-27 to 11-11 | 6, 7 | One live Gemini call on one synthetic column (`personal`); 733 passed |
| 2026-11-12 to 13 | 8 | Dashboard checked in the browser; 880 passed |
| 2026-11-14 | 9 | First `--llm` attempt (unpaced) hit HTTP 429 and 503 and was stopped. 9A: 945 passed after fixing the assistant's hand-count of a tie. 9B: three failures from the assistant's mistakes (a wrong text fragment; the router include line inside a loop giving three duplicate operations; a test counting `app.routes`); then 978 passed |
| 2026-11-14 to 19 | 10A | Read-only mode, demo seed, byte-counting upload limit; one wrong test replaced; 1056, then 1084 passed |
| 2026-11-19 | 10B | Docker demo healthy, 403 on upload, no `.env` or key in the image; 1164 passed |
| 2026-11-19 to 25 | 12 | Fail-fast start (1201 with the regex fix: 20,000 characters took 2.565 s with the old pattern); rules over HTTP (`%Y%Y` raises `re.PatternError`; 1287); paced command (1313); full real-model run 0.794 and 60 of 63 answered; dashboard rules (1329); seeded demo (1340); documents (1343, 1361); specification (1384) |
| 2026-11-25 | final pass | Docker rebuild healthy; the harness inside the image printed the same fingerprint as Windows; the demo database check refused a missing file without creating one; all Mermaid diagrams and the browser check passed (reported 2026-11-26) |
| 2026-11-26 | Git and Render | Commit `014506f`, then `2b9a025` (1393 passed before it), `74b2af3` (1395), `af688be` (1396); see the record above |

## Open items and notes

- **Making the repository public** is pending the final history checks listed in the deployment steps (key pattern search, authors, commit messages, file count).
- **The Gemini key was not rotated** (a deliberate decision on 2026-11-26: the key value is in no commit and in no Render service; OneDrive keeps copies of the local `.env`, so rotation remains advisable if that file's history could ever be exposed).
- **Remaining security risks** are in `SECURITY.md` and `docs/THREAT_MODEL.md`: no authentication, rate limiting or TLS; public API documentation pages; spooling of large bodies; slow date rules (estimated, not measured); concurrent reviews not serialised on PostgreSQL; sensitive free text; prompt injection mitigated but untested with hostile names against a live model; no dependency or penetration scan.
- **Evaluation caveats:** one run of one model; a small, imbalanced, same-author dataset; no hostile column names; the adapter records a read timeout as a provider error. The older `python -m app.evaluation.harness --llm` path sends calls back to back and still says "paid provider calls"; the paced `python -m app.evaluation.llm_run` is the supported route. The snapshot in the image must be refreshed by hand if the dataset or detectors change (tests fail until it is).
- **Parked:** 6D persistence; an audit-event read endpoint; a PostgreSQL row lock.
- **Housekeeping, not required:** duplicated finding-detail panel; `EvaluationReportOut` placement; a leaner `requirements-ui.txt` (both images install the whole `requirements.txt`); `pytest.ini`; duplicated fixtures in older tests; several folders lack `__init__.py`; scratch folders `reports\run1` and `reports\run2` can be deleted (ignored by Git).
- **Where decisions live:** `PROJECT_SPEC.md` (sections 12 and 14), `docs/ARCHITECTURE.md`, `SECURITY.md`, `docs/THREAT_MODEL.md`, `docs/EVALUATION.md`, `docs/DEPLOYMENT.md`.