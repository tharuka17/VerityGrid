> **VerityGrid's read-only demo was deployed to Render's free plan on 2026-11-26** (see [What was done](#what-was-done-2026-11-26)). Only the read-only demo of synthetic data is deployed: uploads, scans and reviews are disabled. A public deployment with uploads or reviews is **not recommended**, because no access control exists.

## What was done (2026-11-26)

The read-only demo was deployed to Render's free plan on 2026-11-26. This section is the record; the sections after it explain how to repeat it.

| Item | Value |
|---|---|
| Source | A GitHub repository that was **private** when deployed, branch `main`, commit `014506f` (Render shows this commit on both services) |
| Services | Two Docker web services created from `render.yaml`: `veritygrid-api` and `veritygrid-ui`, both on the Free plan; deployed by hand at first (`autoDeploy` off), then configured to deploy automatically on a push that changes a file in their image (`autoDeployTrigger: commit` with a build filter per service, set on 2026-11-26) |
| Dashboard | <https://veritygrid-ui.onrender.com> |
| API | <https://veritygrid-api.onrender.com> (interactive documentation at `/docs`) |
| Configuration | The Blueprint sets the demo mode, the production environment label, the mock provider and the seeded database; it sets no secret and no API key |

**What the author observed on 2026-11-26** (terminal output and screenshots):

- The API's `/health` answered `{"status":"ok","app":"VerityGrid","environment":"production"}`.
- `/metrics/summary` returned 3 datasets, 3 completed scans and 12 findings (medium 8, high 4), with 10 pending, 1 approved and 1 rejected (16.67% reviewed): the same numbers as the Docker demo.
- `POST /scans` returned HTTP 403 with code `read_only_mode`, and the API log showed `Write request rejected in read-only mode: method=POST`.
- The dashboard's sidebar showed **API reachable** with `VerityGrid (production)`, and its Overview showed 12 findings. The author also checked the Evaluation page and the blue read-only message on *Upload and scan* and *Review queue*.
- The API log showed health checks answered `200 OK` every five seconds. Neither service's log showed an out-of-memory message in the stretches the author inspected. The dashboard's log showed two clean stop-and-restart cycles whose cause was not established (redeploys and idle recycling are the likely explanation). Memory use was not measured.

**The first problem, and its fix.** After the first deploy the dashboard showed "API unreachable" with `API: http://api:8000` in its sidebar. That address is the default in `Dockerfile.ui`, which works only inside Docker Compose, where a service is named `api`. Render does not provide it, and `VERITYGRID_API_URL` is marked `sync: false` in `render.yaml`, so it has to be entered by hand with the API's **public** address (free services cannot use Render's private network). The fix was to set `VERITYGRID_API_URL` to `https://veritygrid-api.onrender.com` on the dashboard service and deploy it again, because a service receives a changed environment variable only on a new deploy. The dashboard connected after that.

**Automatic deploys, tested once (2026-11-26).** The Blueprint sync for commit `74b2af3` succeeded and no service deployed, as intended, because that commit changed only files outside both build filters. Commit `af688be` changed only `Dockerfile.ui`: Render started a deploy of the dashboard service labelled "New commit via Auto-Deploy", it went live about a minute later, and the API service showed no new deploy. Not tested: an API-file commit deploying the API, and a documents-only commit deploying nothing.

**What this does not show.** No load testing, no uptime history and no memory measurement. Until automatic deploys were configured (2026-11-26, after the first deployment) both services stayed on commit `014506f` until deployed by hand. Since then a push to `main` redeploys a service only if it changes a file that goes into that service's image, so a documents-only commit leaves the running demo unchanged. The two addresses are public, so anyone with the link can open the demo. Free services can be changed or removed by the platform, so availability is not guaranteed.

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

`render.yaml` sets `autoDeployTrigger: commit` with a build filter per service, so after the first deployment a push to `main` deploys a service when it changes a file that goes into that service's image (documents and tests do not). To deploy by hand, use **Manual Deploy** on the service. Change this behaviour in `render.yaml`, not in the dashboard: the services are Blueprint-managed, and the next Blueprint sync overwrites dashboard changes.

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