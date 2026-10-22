# Deploying Dr. Aishu on DOM Cloud

DOM Cloud (domcloud.co) can host Dr. Aishu on its free plan. This is a walkthrough
of the two things that matter: picking the right framework preset, and pointing the
deploy at the right branch.

## Which preset to choose

In the create-website wizard, choose **Uvicorn (ASGI)**.

Dr. Aishu is a FastAPI app, and FastAPI speaks ASGI. The presets differ in how they
start your process:

| Preset | Speaks | Works for Dr. Aishu? |
| --- | --- | --- |
| Python `app.py` | WSGI | No — needs a hand-written `application` callable |
| Python `wsgi.py` | WSGI | No — same problem |
| Streamlit `app.py` | its own runtime | No — different framework |
| Flask / Flask (MySQL) / Flask (PostgreSQL) | WSGI | No |
| Django | WSGI | No |
| **Uvicorn (ASGI)** | **ASGI** | **Yes — use this one** |

Choosing a WSGI preset means you must add a translation layer (`a2wsgi`) and a
`passenger_wsgi.py` shim. Choosing Uvicorn (ASGI) runs the app natively, with the
same command the Dockerfile uses. `passenger_wsgi.py` is kept in the repository as
a fallback for WSGI-only hosts, but you do not need it here.

If the wizard asks for a startup file or command, use:

```
python -m uvicorn app.main:app --host 127.0.0.1 --port $PORT
```

Bind `127.0.0.1`, not `0.0.0.0`. Passenger proxies from loopback and expects the
app to listen there.

## If the wizard generates the start command itself

DOM Cloud's Uvicorn preset writes `uvicorn main:app --port $PORT`. That fails
against this repository for two reasons, both of which the repository now works
around, but it is worth knowing why:

1. **`main:app` only resolves from the checkout root.** The app lives in
   `app/main.py`, so the correct target is `app.main:app`. A root-level
   `main.py` shim re-exports the app, so `main:app` works too, but only when the
   process starts in the repository root (`app_root: public_html` ensures that).
2. **Bare `uvicorn` may not be on PATH.** `pip install --user` puts the console
   script in `~/.local/bin`, which Passenger does not always include. Prefer
   `python -m uvicorn`.

The correct command is:

```
env PORT=$PORT python -m uvicorn app.main:app --host 127.0.0.1 --port $PORT
```

If you keep the wizard's generated config, edit the last line to match. Symptom
of getting this wrong: the deploy log ends with "Execution Finished" but the
site returns 503, and the error log contains
`Error loading ASGI app. Could not import module "main"`.

## Point the deploy at the right branch

The repository's default branch (`main`) is empty. The application lives on
`feat/external-dataset-sources`. If the wizard clones the default branch you will
deploy nothing, so either:

- switch the repository's default branch to `feat/external-dataset-sources` on
  GitHub (Settings → Branches → change default), or
- use a `source` block that names the branch explicitly:

```yaml
source:
  url: https://github.com/Geetesh3/dr-aishu.git
  branch: feat/external-dataset-sources
```

## Deploy script

`domcloud.yml` in the repository root is ready to paste into
**Setup → Deploy → Add a deployment task**. It installs dependencies, copies the
app into `public_html`, and configures Passenger.

## Free plan limits, and what they mean here

| Limit | Value | Impact |
| --- | --- | --- |
| Storage | 1.5 GB | Fine. App is 3.5 MB; dependencies are about 102 MB (numpy and scikit-learn dominate). |
| Network | 2 GB/month | Fine for testing. Roughly 2,000 page views of 1 MB. |
| CPU / RAM | unlimited (shared) | Fine. The app idles near 300 MB. |
| Instances | 3 | Fine. |
| Login | monthly | **You must sign in each month** or the site is removed. |

An invitation code is required for the free plan, unless your GitHub account is
older than six months and has at least one follower.

## What will not work on the free plan

**Image OCR of printed text.** The app shells out to the `tesseract` binary, which
needs a system package install. The free plan has no sudo and no `apt`. The app
handles this gracefully — `app/vision.py` reports `tesseract binary not found on
PATH` and the endpoint returns a clear error instead of crashing. Everything else
(typed symptoms, severity, duration, medicines, timetable, routines) is unaffected.

If you need OCR, options are: install Tesseract on a paid plan, or run the OCR call
against an external service.

## Model files are not in the repository

`data/models/*.joblib` is gitignored, so a fresh deploy starts with **no trained
model**. The app still works — it falls back to the knowledge base. To train on the
server, call `POST /api/model/train` with an operator key, or copy the model files
into `data/models/` over SSH.

## Persistent data

`DR_AISHU_DATA_DIR` points at `public_html/data`, inside the site, so the SQLite
case database and API keys survive redeploys (unlike Render's free tier). The
directory sits outside the served document root, so it is not reachable over HTTP.
