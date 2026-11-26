# Deployment

> **VerityGrid has not been deployed anywhere.** This document says how to run the read-only demo on your own machine and how it could be deployed to Render. Nothing below has been done by the author unless a result is reported in the project's status file.

## 1. What can be deployed

**Only the read-only demo with pre-seeded synthetic data.** It has no uploads, no scans, no reviews, no API key and no model calls. The API answers every write with HTTP 403 `read_only_mode`.

**A public deployment with uploads or reviews is not recommended.** No authentication, authorization or rate limiting exists. Development mode, which allows writes, is for a machine you control. See [SECURITY.md](../SECURITY.md) and [THREAT_MODEL.md](THREAT_MODEL.md).

## 2. Run the demo locally (no external services)

With Docker installed and running:

```powershell
docker compose up --build
```

Open the dashboard at <http://127.0.0.1:8501> and the API at <http://127.0.0.1:8000/docs>. Both ports are bound to your machine only. Stop with `Ctrl+C` and `docker compose down`. The image seeds its own database (3 datasets, 12 findings, 2 example reviews) and carries a committed snapshot of the real-model evaluation run as its report.

Without Docker, use the steps in the [README](../README.md) (`python -m scripts.seed_demo` and the `DATABASE_URL` and `VERITYGRID_MODE` variables).

For a portfolio, **a short screen recording of the local demo is a perfectly good alternative to a public deployment.**

### What the demo contains (as of 2026-11-22)

- **Data:** the three synthetic sample files, scanned during the image build with sample detection rules on the inventory file, so all six detectors appear: 3 datasets, 12 findings (medium 8, high 4) and 2 example review decisions.
- **The Evaluation page** shows a committed snapshot of one real-model run (`data/evaluation/llm_run_2026-11-19.json`), labelled a single, non-reproducible run, next to the reproducible deterministic and mock results. The image carries no key and makes no model call.
- **Startup:** in demo mode the API refuses to start if its seeded database file is missing, so a wrong `DATABASE_URL` shows up as an error immediately.

## 3. Deploy the demo to Render (optional)

### What you need first (external)

| Need | Cost | Notes |
|---|---|---|
| A Git repository host (GitHub, GitLab or Bitbucket) with this project pushed | Free | The project is not a Git repository yet. Before the first push confirm `.env` is **not** tracked (`git check-ignore -v .env` should name an ignore rule; `git status` must not list it), run `git grep -n "AIza"` to look for a leaked Google key, and rotate your Gemini key if it was ever synced or exposed. |
| A Render account | Free plan available | Whether a payment method is required even for free instances is **not verified**: check during sign-up. |
| `render.yaml` in the repository root | Included | Defines two free Docker web services. |

You do **not** need a Gemini key, a database service or a persistent disk. The seeded database and the report are baked into the image.

### Steps

1. Push the repository. Check the files above are included and `.env` is not.
2. In the Render dashboard choose **New**, then **Blueprint**, connect the repository and apply. Render reads `render.yaml` and creates two web services, `veritygrid-api` and `veritygrid-ui`.
3. When Render asks for `VERITYGRID_API_URL` (the dashboard's service), enter the API's public address. It normally has the form `https://veritygrid-api.onrender.com`, but Render may add a suffix if the name is taken, so check the real address on the API service's page afterwards.
4. Wait for both builds. Each build installs the dependencies, seeds the database and generates the report.
5. If the API's real address differs from what you entered, open the UI service's **Environment** settings, correct `VERITYGRID_API_URL` and let it redeploy.

`autoDeploy` is off, so a push does not deploy anything until you ask.

### Verify after deploying (and tell the author what you see)

```powershell
curl.exe -sS https://<your-api>.onrender.com/health
curl.exe -sS https://<your-api>.onrender.com/metrics/summary
curl.exe -sS -X POST https://<your-api>.onrender.com/scans -F "file=@data/sample/customers.csv"
```

Expected: `"environment":"production"`; 3 datasets, 12 findings (medium 8, high 4), pending 10, approved 1, rejected 1; and HTTP 403 `read_only_mode` for the upload. Open the dashboard and check the Overview, the Evaluation page and the blue read-only message on *Upload and scan* and *Review queue*. In each service's environment settings confirm there is no `GEMINI_*` variable.

### How the free plan behaves (per Render's documentation at the time of writing; check the current pages)

- **Spin-down:** a free web service stops after 15 minutes without traffic and takes about a minute to start again.
- **First visit:** the dashboard may wake up before the API. Open `https://<your-api>.onrender.com/health` first, wait for it to answer, then open the dashboard and click *Refresh data*. Otherwise the dashboard may show "API unreachable" or a timeout until the API is awake.
- **Ephemeral filesystem:** local files are lost on redeploy, restart or spin-down. This is harmless here because the database is read-only and rebuilt into every image.
- **Instance hours:** 750 free hours per month are shared across the workspace.
- **No private network:** free services cannot receive private traffic, which is why the dashboard uses the API's public address.
- **Resources:** third-party pages report 512 MB of RAM and a fraction of a CPU for free instances; check Render's pricing page. The images load pandas, LangGraph and Streamlit, so a service may run out of memory. If a service fails with an out-of-memory message, a paid plan or a leaner image is the fix.
- **WebSockets** (which Streamlit uses) are supported on Render. Streamlit behind Render's proxy is **not verified** by the author.

### Tear down

Delete both services in the Render dashboard. Nothing else was created.

## 4. Which steps need external configuration or a paid service

| Step | External account or configuration | Paid? |
|---|---|---|
| Run locally with Docker | Docker installed (licensing terms are not verified here) | No |
| Push the code | A Git host account; a clean repository (no `.env`, no key) | Free tier available |
| Deploy to Render | A Render account; a connected repository; entering `VERITYGRID_API_URL` | The free plan is available; limits above |
| Keep the services always awake | A paid Render plan, or an external pinger | **Yes** (price not verified here) |
| More memory or CPU | A paid Render plan | **Yes** |
| Persistent data | A paid disk or a database | Yes, and **not recommended** (see section 1) |
| Custom domain and TLS | Render supports both on free services | No (domain registration costs money) |
| Rate limiting, a web application firewall, monitoring and alerts | A platform feature or another service | Usually |
| A real language model | A Gemini key (free quota with data-use terms, or billing) | Optional; never needed for the demo |
| Access control for uploads and reviews | **Does not exist.** Would have to be built first. | n/a |

## 5. Checklist for anything other people can reach

1. Do not ship `.env`; set configuration in the platform.
2. `VERITYGRID_MODE=demo` and `ENVIRONMENT=production`.
3. `AI_PROVIDER=mock` and no `GEMINI_*` values.
4. The seeded demo database only.
5. TLS and rate limiting from the platform.
6. The dashboard points at the read-only API.
7. Read [SECURITY.md](../SECURITY.md).