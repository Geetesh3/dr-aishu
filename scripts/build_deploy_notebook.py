"""Generates notebooks/Dr_Aishu_Deploy.ipynb.

The deployment notebook is built from cell definitions here so the content stays
reviewable as Python source. It covers the three free paths to a hosted Dr. Aishu:

1. Google Colab + a public tunnel (fastest, no account needed beyond Google).
2. Render / Fly.io (a permanent URL, free tier).
3. Hugging Face Spaces (a permanent URL, no credit card).

Run:  python scripts/build_deploy_notebook.py
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
CELLS: list[tuple[str, str]] = []


def md(text):
    CELLS.append((MD, text.strip("\n")))


def code(text):
    CELLS.append((CODE, text.strip("\n")))


md("""
# Dr. Aishu — Host It For Free

This notebook gives you a **public URL** for Dr. Aishu, at no cost. Three routes, cheapest first:

| Route | Cost | URL lasts | Best for |
|---|---|---|---|
| Colab + tunnel | free | while the tab is open | testing, sharing a demo right now |
| Render free web service | free | permanent | a real app with users |
| Fly.io free allowance | free | permanent | a real app, no cold-start limit |

Every route runs the **same app**, with the same API-key system: users get their own key from `/developers`, and the browser UI works with no signup at all.

> Dr. Aishu is an educational triage aid. She does not replace a qualified clinician, and emergency cases deliberately return no medicine suggestions.
""")

md("""
## Step 1 — Get the project into this runtime

Clone the repository, or upload a zip of the project and unzip it.
""")

code('''
REPO_URL = "https://github.com/YOUR_USERNAME/dr-aishu.git"  # <-- change me, or leave blank to upload

import os, pathlib, shutil, subprocess, zipfile

if REPO_URL and "YOUR_USERNAME" not in REPO_URL:
    target = pathlib.Path("/content/dr-aishu")
    if target.exists():
        shutil.rmtree(target)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)], check=True)
    os.chdir(target)
else:
    # Upload route: choose a zip of the project folder.
    try:
        from google.colab import files
        up = files.upload()
        name = next(iter(up))
        with zipfile.ZipFile(name) as zf:
            zf.extractall("/content")
        for candidate in pathlib.Path("/content").iterdir():
            if (candidate / "app" / "main.py").exists():
                os.chdir(candidate)
                break
    except Exception as exc:
        print("Upload a zip of the project, or set REPO_URL above.", exc)

print("working directory:", os.getcwd())
print("looks like the project:", pathlib.Path("app/main.py").exists())
''')

md("""
## Step 2 — Install dependencies

FastAPI and friends for the API, plus Tesseract for reading lab reports and prescriptions out of photos.
""")

code('''
!pip install -q fastapi "uvicorn[standard]" jinja2 python-multipart pillow scikit-learn joblib pyngrok
!apt-get install -y -qq tesseract-ocr > /dev/null 2>&1

import subprocess
print("tesseract:", subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.splitlines()[0])
''')

md("""
## Step 3 — Create a permanent admin key (optional but recommended)

A pinned key survives restarts, so you always have admin access. On Colab's ephemeral disk this matters; on a paid volume it matters less.
""")

code('''
import secrets, os

BOOTSTRAP_KEY = "dra_" + secrets.token_urlsafe(32)
os.environ["DR_AISHU_BOOTSTRAP_KEY"] = BOOTSTRAP_KEY
print("Your permanent admin key (keep it secret, save it now):")
print()
print("   ", BOOTSTRAP_KEY)
print()
print("This key has the Pro tier: every scope, including training and export.")
''')

md("""
## Step 4 — Start the server

Runs the app in the background and waits until `/api/health` answers.
""")

code('''
import os, subprocess, time, json, urllib.request, pathlib

# Fresh database in this runtime; harmless if it already exists.
pathlib.Path("data").mkdir(exist_ok=True)

log = open("/content/dr-aishu-server.log", "w")
server = subprocess.Popen(
    ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=log, stderr=subprocess.STDOUT,
)

for attempt in range(40):
    time.sleep(0.5)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=2) as r:
            health = json.load(r)
        print("Dr. Aishu is up.")
        print("  assistant:", health["assistant"], "| version:", health["version"])
        print("  knowledge base:", health["knowledge_base"])
        print("  model:", health["model"].get("backend") or "rules only")
        break
    except Exception:
        if server.poll() is not None:
            print("server exited; last log lines:")
            print(open("/content/dr-aishu-server.log").read()[-2000:])
            break
else:
    print("server did not become healthy in time; see /content/dr-aishu-server.log")
''')

md("""
## Step 5 — Publish it with a free public URL

Colab runs behind a firewall, so we tunnel out. **ngrok** is the simplest; it needs a free token from [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).

Free ngrok URLs change each restart. That is fine for a demo; for a stable address use Render or Fly.io further down.
""")

code('''
NGROK_AUTHTOKEN = ""   # <-- paste your free token, or leave blank to use a Cloudflare quick tunnel
PORT = 8000

public_url = None

if NGROK_AUTHTOKEN:
    from pyngrok import ngrok, conf
    conf.get_default().auth_token = NGROK_AUTHTOKEN
    tunnel = ngrok.connect(PORT, "http")
    public_url = tunnel.public_url
    print("ngrok tunnel:", public_url)
else:
    # Cloudflare's quick tunnel needs no account at all.
    import subprocess, re, time
    if not pathlib.Path("/usr/local/bin/cloudflared").exists():
        subprocess.run(
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True, check=True,
        )
    cf = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in cf.stdout:
        m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
        if m:
            public_url = m.group(0)
            break

print()
print("=" * 66)
print("  Dr. Aishu is live at:", public_url)
print("  Web UI   :", public_url)
print("  API docs :", public_url + "/docs")
print("  Get a key:", public_url + "/developers")
print("=" * 66)
''')

md("""
## Step 6 — Prove the public URL works

Calls the live server over the public internet, exactly as your own app would: sign up for a key, then analyse symptoms with it.
""")

code('''
import json, urllib.request

def call(url, payload=None, headers=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

if public_url:
    signup = call(f"{public_url}/api/keys/signup", {"email": "demo@example.com", "label": "colab"})
    key = signup["secret"]
    print("issued key:", key[:16] + "...", "| tier:", signup["tier_label"])

    out = call(
        f"{public_url}/api/analyze",
        {"text": "high fever, chills and body aches for three days with a dry cough"},
        headers={"X-API-Key": key},
    )
    print()
    print("top diagnosis :", out["diagnoses"][0]["name"], f"({out['diagnoses'][0]['confidence']}%)")
    print("severity      :", out["severity"]["score"], "/10 ->", out["severity"]["triage"])
    meds = out["medicines"]["recommended"]
    print("medicines     :", ", ".join(m["name"] for m in meds) or "none (see the triage advice)")
    if meds:
        print("first dose at :", meds[0].get("times") or meds[0].get("frequency"))
    print("routine       :", out["routine"]["name"])
    for block in out["routine"]["blocks"][:3]:
        print("   ", block["time"], "-", block["title"])

    sol = out["solutions"]
    print("action plan   :", sol["condition"]["name"] if sol["condition"] else "n/a")
    for problem in sol["problems"][:3]:
        covered = ", ".join(problem["covered_by"]) or "no medicine"
        print("   ", problem["label"], "->", problem["steps"][0] if problem["steps"] else "ask a clinician")
        print("        covered by:", covered)

    print()
    print("Emergency check (no medicines should be returned):")
    emg = call(
        f"{public_url}/api/analyze",
        {"text": "crushing chest pain going to my left arm with sweating and breathlessness"},
        headers={"X-API-Key": key},
    )
    print("  triage:", emg["severity"]["triage"], "| recommended medicines:", emg["medicines"]["recommended"])
else:
    print("Run Step 5 first to get a public URL.")
''')

md("""
## Step 7 — Use it from your own app

The key from Step 6 is all your app needs. This is the entire integration.
""")

code('''
# Your app, anywhere in the world, can now do this.
import os, requests

DR_AISHU_URL = public_url or "http://127.0.0.1:8000"
DR_AISHU_KEY = key if public_url else BOOTSTRAP_KEY

def diagnose(symptoms, **context):
    """Return Dr. Aishu's analysis of a free-text description."""
    r = requests.post(
        f"{DR_AISHU_URL}/api/analyze",
        headers={"X-API-Key": DR_AISHU_KEY},
        json={"text": symptoms, **context},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def diagnose_image(path, symptoms="", **context):
    """Analyse a photo of a lab report or prescription, with optional notes."""
    with open(path, "rb") as fh:
        r = requests.post(
            f"{DR_AISHU_URL}/api/analyze-image",
            headers={"X-API-Key": DR_AISHU_KEY},   # let requests set the multipart boundary
            files={"image": fh},
            data={"text": symptoms, **context},
            timeout=60,
        )
    r.raise_for_status()
    return r.json()

print("Helper ready. Example:")
out = diagnose("burning when I pee and going very often, lower abdominal pain for two days")
print(" ", out["diagnoses"][0]["name"], "|", out["severity"]["triage"])
''')

md("""
## Route B — Render, for a permanent free URL

Colab stops when the tab closes. For a URL that stays up, Render's free tier runs this repository directly.

1. Push this project to a GitHub repository.
2. In Render: **New +** → **Blueprint**, and pick the repository.
3. Render reads `render.yaml`, builds the `Dockerfile`, and gives you a URL like `https://dr-aishu.onrender.com`.

**Know the two free-tier limits.** The instance sleeps after about 15 minutes idle and takes about 50 seconds to wake, and its disk is ephemeral, so issued API keys reset on each deploy. `render.yaml` sets `DR_AISHU_BOOTSTRAP_KEY` to a generated value for exactly that reason: that key always works.

Check it once it is deployed:
""")

code('''
RENDER_URL = ""   # e.g. "https://dr-aishu.onrender.com"

if RENDER_URL:
    import urllib.request, json
    with urllib.request.urlopen(f"{RENDER_URL}/api/health", timeout=60) as r:
        h = json.load(r)
    print("deployed:", h["assistant"], h["version"], "| model:", h["model"].get("backend"))
    print("open the UI at:", RENDER_URL)
    print("get a key at  :", RENDER_URL + "/developers")
else:
    print("Set RENDER_URL after the first deploy to check it from here.")
''')

md("""
## Route C — Fly.io, for a permanent free URL with a disk

Fly's free allowance covers one small machine, and a volume keeps your keys and cases across restarts.

```bash
fly launch --no-deploy --copy-config
fly volumes create dr_aishu_data --size 1
fly secrets set DR_AISHU_BOOTSTRAP_KEY="dra_$(openssl rand -hex 24)"
fly deploy
```

Then set `FLY_URL` below to `https://<your-app>.fly.dev`.

## Route D — Hugging Face Spaces, no credit card

1. Create a new **Docker** Space.
2. Copy the repository into it. The `Dockerfile` works unchanged.
3. Set `DR_AISHU_BOOTSTRAP_KEY` under **Settings → Variables and secrets**.
4. The Space serves at `https://<user>-<space>.hf.space`.

## Keep the model trained

Any of these hosts can retrain. Hit **Train** in the web UI, or call the API with a Pro key:

```bash
curl -X POST "$BASE/api/model/train" \\
  -H "X-API-Key: $ADMIN_KEY" -H "Content-Type: application/json" \\
  -d '{"backend":"tfidf","use_seed":true}'
```

For the transformer backend, use `notebooks/Dr_Aishu_Training.ipynb`, which exports a model bundle you drop into `data/models/`.
""")

md("""
## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `429` with `Retry-After` | rate limit hit | wait, or use a higher-tier key |
| `401` | missing or invalid key | send `X-API-Key`, get one from `/developers` |
| `403` | tier lacks the scope | training and export need a Pro key |
| Render takes ~50s to answer | free instance was asleep | expected; upgrade or use Fly |
| Keys vanish after redeploy | ephemeral disk | use `DR_AISHU_BOOTSTRAP_KEY`, or attach a volume |
| OCR returns nothing | Tesseract missing | the `Dockerfile` installs it; check the build log |
| Tunnel URL is unreachable | Colab tab closed | keep it open, or use Route B/C/D |

## Safety rules that never change

* Emergency presentations return **no** medicines, only escalation advice.
* Medicine suggestions are limited to the leading diagnosis and always carry a disclaimer.
* Stored cases require explicit consent and are scrubbed of direct identifiers first.
* Nothing here diagnoses or prescribes. Show the `disclaimer` field to your users.
""")


def main() -> None:
    nb = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": [line + "\n" for line in text.split("\n")][:-1]
                + ([text.split("\n")[-1]] if text else []),
            }
            for kind, text in CELLS
        ],
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    for cell in nb["cells"]:
        src = cell["source"]
        cell["source"] = [line if line.endswith("\n") else line for line in src]
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    out = Path(__file__).resolve().parent.parent / "notebooks" / "Dr_Aishu_Deploy.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
