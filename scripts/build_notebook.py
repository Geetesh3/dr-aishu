"""Generates notebooks/Dr_Aishu_Training.ipynb.

Building the notebook from cell definitions keeps the content reviewable as
Python source rather than a large hand-written JSON blob.
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
# Dr. Aishu — Train Your Medical AI in Google Colab

This notebook trains **Dr. Aishu**, the medical assistant in your application. Everything here uses free, open-source tooling and runs on Colab's free CPU runtime, or a free GPU if you enable one.

**What you get at the end**

1. A trained diagnosis classifier — scikit-learn by default, or a fine-tuned HuggingFace transformer.
2. An accuracy and macro-F1 report, plus a confusion analysis showing where she is weak.
3. A downloadable model bundle to drop into your app's `data/models/` folder.
4. A loop that gets stronger every time you feed it real consented cases from your app.

**Two ways to work**

* **Offline (recommended).** Export training data from your app with `GET /api/export/training-data`, upload the JSON here, train, then copy the model file back.
* **Direct.** Point this notebook at a running copy of your app (for example through ngrok) and pull data straight from it.

> Dr. Aishu is an educational triage aid. She does not replace a qualified clinician.
""")

md("""
## Step 0 — Choose a runtime

`Runtime → Change runtime type`

* **CPU** is enough for the scikit-learn backend. It trains in seconds.
* **T4 GPU** makes the transformer backend practical, a few minutes rather than an hour.

## Step 1 — Get the project

Clone your repository, or upload a zip of the project folder.
""")

code('''
# Option A: clone your repository (edit the URL)
REPO_URL = "https://github.com/YOUR_USERNAME/dr-aishu.git"  # <-- change me

import os, subprocess, pathlib

if REPO_URL and "YOUR_USERNAME" not in REPO_URL:
    if not os.path.isdir("dr-aishu"):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "dr-aishu"], check=True)
    os.chdir("dr-aishu")
    subprocess.run(["git", "pull", "--ff-only"], check=False)
    print("project ready at", os.getcwd())
else:
    # Option B: upload dr-aishu.zip with the file browser on the left, then run:
    #   !unzip -q dr-aishu.zip && cd dr-aishu
    # and re-run this cell with REPO_URL = ""
    print("No repo configured. Upload the project zip, unzip it, then re-run this cell.")
''')

code('''
# Confirm we are in the project root
import os, pathlib

assert pathlib.Path("app/kb.py").exists(), "Run this from the project root (the folder containing app/)"
print("project root:", os.getcwd())
print("contents:", sorted(os.listdir(".")))
''')

md("""
## Step 2 — Install dependencies

Only free, open-source packages. The transformer line is commented out because the download is large. Enable it only if you want the deep-learning backend.
""")

code('''
!pip install -q scikit-learn joblib pandas

# Uncomment for the HuggingFace transformer backend (free, open source)
# !pip install -q transformers datasets accelerate torch

import sklearn, joblib, pandas as pd
print("scikit-learn", sklearn.__version__)
print("pandas", pd.__version__)
''')

md("""
## Step 3 — Load the medical knowledge base

Dr. Aishu starts with a curated knowledge base of conditions, symptoms, medicines and routines. This is her medical grounding. The model learns to recognise the language of these conditions, and the knowledge base keeps every result explainable.
""")

code('''
import sys, json
sys.path.insert(0, os.getcwd())

from app.kb import KB

print(f"conditions : {len(KB.diseases)}")
print(f"symptoms   : {len(KB.symptoms)}")
print(f"medicines  : {len(KB.medicines)}")
print(f"routines   : {len(KB.routines)}")
print()
for d in list(KB.diseases.values())[:5]:
    print(f"  {d.name:38s} {d.category:16s} severity {d.severity_score}/10")
''')

md("""
## Step 4 — Build the training set

Two sources are merged:

1. **Seed data** generated from the knowledge base — thousands of varied symptom descriptions, so training works on day one.
2. **Real consented cases** from your app, with user corrections up-weighted so they carry more influence.

To include real data, export it from your app with `GET /api/export/training-data`, upload `dr_aishu_training_data.json` using the file browser, then set `REAL_DATA_PATH` below.
""")

code('''
from scripts.seed_training_data import generate

PER_DISEASE = 60          # synthetic examples per condition
REAL_DATA_PATH = ""       # e.g. "dr_aishu_training_data.json" after uploading

seed = generate(per_disease=PER_DISEASE, seed=7)
print(f"seed examples: {len(seed)}")

real = []
if REAL_DATA_PATH and pathlib.Path(REAL_DATA_PATH).exists():
    payload = json.loads(pathlib.Path(REAL_DATA_PATH).read_text())
    real = payload.get("examples", payload if isinstance(payload, list) else [])
    print(f"real consented examples: {len(real)}")
else:
    print("no real data file supplied (seed data alone is enough to start)")

examples = seed + real
df = pd.DataFrame(examples)
print()
print(df["label"].value_counts().head(10))
print()
print("total examples:", len(df), "| labels:", df["label"].nunique())
''')

code('''
# Eyeball a few generated descriptions
for row in df.sample(5, random_state=1).to_dict(orient="records"):
    print(f"[{row['label']}] {row['text']}\\n")
''')

md("""
## Step 5 — Train

### Option A — TF-IDF with logistic regression

Fast, always available, and gives calibrated probabilities that are easy to explain. Trains in seconds.
""")

code('''
from app import learning

report = learning.train_tfidf(examples, min_examples_per_label=3)
print(json.dumps({
    "backend": report.backend,
    "examples": report.n_examples,
    "labels": report.n_labels,
    "accuracy": report.accuracy,
    "macro_f1": report.macro_f1,
    "seconds": report.seconds,
    "notes": report.notes,
}, indent=2))
''')

md("""
### Option B — Fine-tune a HuggingFace transformer

Better accuracy, needs a GPU. Free and open-source base models you can try:

* `distilbert-base-uncased` — small and fast
* `emilyalsentzer/Bio_ClinicalBERT` — pretrained on clinical notes
* `dmis-lab/biobert-base-cased-v1.1` — biomedical literature

Skip this cell if Option A is good enough.
""")

code('''
# Skip unless you installed the transformer packages and enabled a GPU.
RUN_TRANSFORMER = False
BASE_MODEL = "distilbert-base-uncased"
EPOCHS = 4

if RUN_TRANSFORMER:
    report_t = learning.train_transformer(examples, base_model=BASE_MODEL, epochs=EPOCHS)
    print(json.dumps({
        "backend": report_t.backend,
        "base_model": BASE_MODEL,
        "examples": report_t.n_examples,
        "labels": report_t.n_labels,
        "accuracy": report_t.accuracy,
        "seconds": report_t.seconds,
        "device": report_t.detail.get("device"),
    }, indent=2))
else:
    print("Transformer training skipped. Set RUN_TRANSFORMER = True to enable.")
''')

md("""
## Step 6 — Evaluate properly

A held-out split plus a confusion analysis shows exactly where Dr. Aishu is weak, so you know which conditions need more data.
""")

code('''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

docs = [learning.build_document(e.get("text", ""), e.get("symptoms")) for e in examples]
labels = [e["label"] for e in examples]

X_tr, X_te, y_tr, y_te = train_test_split(
    docs, labels, test_size=0.2, random_state=42, stratify=labels
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")),
])
pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)

print(f"held-out accuracy: {accuracy_score(y_te, pred):.3f}\\n")
print(classification_report(y_te, pred, zero_division=0))
''')

code('''
# Which conditions get confused with each other?
names = sorted(set(y_te))
cm = confusion_matrix(y_te, pred, labels=names)

confusions = []
for i, actual in enumerate(names):
    for j, predicted in enumerate(names):
        if i != j and cm[i, j] > 0:
            confusions.append((int(cm[i, j]), actual, predicted))

confusions.sort(reverse=True)
print("Top confusions (actual -> predicted):")
for n, actual, predicted in confusions[:12]:
    print(f"  {n:3d}  {actual:28s} -> {predicted}")
if not confusions:
    print("  none — every held-out case was classified correctly")
''')

md("""
## Step 7 — Try Dr. Aishu on new sentences

Edit the list below and re-run.

Two numbers are shown for each sentence:

* **model** — the raw classifier output. Probabilities are spread thinly across all 42 conditions, so the absolute values look small. Only the ranking matters here.
* **Dr. Aishu** — the blended result the app actually shows, combining the model with the knowledge base. This is the number to trust.
""")

code('''
from app.nlp import extract
from app.analyzer import diagnose

learning.load_model()  # activate the freshly trained model

samples = [
    "high fever with severe body aches and chills for three days, dry cough",
    "burning when I pee and needing to go very often for two days",
    "throbbing one sided headache with nausea and light sensitivity",
    "watery diarrhoea and vomiting since yesterday with stomach cramps",
    "very thirsty all the time and peeing a lot at night, losing weight",
    "itchy red circular rash on my arm spreading slowly for two weeks",
]

for s in samples:
    preds = learning.predict(s, top_k=3)
    facts = extract(s)
    blended = diagnose(facts, top_n=3, ml_predictions=preds)

    raw = ", ".join(f"{p['label']} ({p['probability']*100:.0f}%)" for p in preds)
    final = ", ".join(f"{c.name} ({c.confidence:.0f}%)" for c in blended)
    print(f"\\n{s}")
    print(f"  model    : {raw}")
    print(f"  Dr. Aishu: {final}")
''')

md("""
## Step 8 — Save and download the model

Copy the bundle into your app's `data/models/` folder and restart the app. Dr. Aishu loads it automatically.
""")

code('''
import shutil, pathlib

BUNDLE = pathlib.Path("dr_aishu_model")
if BUNDLE.exists():
    shutil.rmtree(BUNDLE)
(BUNDLE / "models").mkdir(parents=True)

src = pathlib.Path("data/models")
if src.exists():
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, BUNDLE / "models" / f.name)
        elif f.is_dir():
            shutil.copytree(f, BUNDLE / "models" / f.name, dirs_exist_ok=True)

(BUNDLE / "README.txt").write_text(
    "Dr. Aishu model bundle\\n"
    "Copy the contents of models/ into your app's data/models/ directory, then restart the app.\\n"
)

shutil.make_archive("dr_aishu_model", "zip", BUNDLE)
print("created dr_aishu_model.zip")
for p in sorted(BUNDLE.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(BUNDLE)}  ({p.stat().st_size/1024:.1f} KB)")

try:
    from google.colab import files
    files.download("dr_aishu_model.zip")
except Exception as exc:
    print("Not running in Colab, so download dr_aishu_model.zip manually:", exc)
''')

md("""
## Step 9 — Optional: pull live data from your running app

If your app is reachable from Colab, for example through ngrok, fetch its consented cases and retrain. This is the continuous-improvement loop.
""")

code('''
APP_URL = ""   # e.g. "https://abc123.ngrok-free.app"
RETRAIN_WITH_LIVE_DATA = False

if RETRAIN_WITH_LIVE_DATA and APP_URL:
    import urllib.request

    with urllib.request.urlopen(f"{APP_URL}/api/export/training-data", timeout=60) as r:
        live = json.loads(r.read())
    live_examples = live["examples"]
    print(f"pulled {len(live_examples)} consented examples from the app")

    rep = learning.train_tfidf(seed + live_examples, min_examples_per_label=3)
    print(json.dumps({
        "examples": rep.n_examples,
        "labels": rep.n_labels,
        "accuracy": rep.accuracy,
        "macro_f1": rep.macro_f1,
    }, indent=2))
else:
    print("Set RETRAIN_WITH_LIVE_DATA = True and APP_URL to enable.")
''')

md("""
## Step 10 — Optional: serve the app from Colab

Runs Dr. Aishu in this runtime and exposes it with ngrok. Colab runtimes stop when the tab closes, so use this for testing only.
""")

code('''
RUN_APP = False

if RUN_APP:
    !pip install -q fastapi "uvicorn[standard]" jinja2 python-multipart pillow pyngrok
    !apt-get install -y -qq tesseract-ocr > /dev/null

    import subprocess, time
    proc = subprocess.Popen([
        "python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"
    ])
    time.sleep(6)

    from pyngrok import ngrok
    print("Dr. Aishu is live at:", ngrok.connect(8000))
else:
    print("Set RUN_APP = True to serve the app from this runtime.")
''')

md("""
## How the improvement loop works

```
 user describes symptoms -> Dr. Aishu analyses (knowledge base + model)
          |                             |
          |                   case stored, consent required
          |                   direct identifiers scrubbed
          v                             v
 user corrects the diagnosis -> correction stored with double weight
                                        |
                                        v
                  retrain (app button, CLI, or this notebook)
                                        |
                                        v
                  accuracy improves on real cases
```

**Safety rules that never change**

* Nothing is stored without explicit consent.
* Red-flag and emergency triage is rule-based and cannot be overridden by the model.
* Medicine filtering for allergy, age, pregnancy and contraindication always applies.
* The model's opinion is blended with, never replaces, the auditable knowledge base.

**Where to add more medical depth**

| File | What to add |
| --- | --- |
| `app/data/diseases.json` | New conditions, symptom weights, red flags, self-care |
| `app/data/symptoms.json` | New synonyms so more phrasings are understood |
| `app/data/medicines.json` | New medicines with contraindications and interactions |
| `app/data/routines.json` | New daily routine templates |

After editing any of those, re-run Step 4 and Step 5 to retrain.
""")

nb = {
    "cells": [
        {
            "cell_type": t,
            "metadata": {},
            "source": [line + "\n" for line in body.split("\n")][:-1] + [body.split("\n")[-1]],
            **({"execution_count": None, "outputs": []} if t == CODE else {}),
        }
        for t, body in CELLS
    ],
    "metadata": {
        "accelerator": "CPU",
        "colab": {"name": "Dr_Aishu_Training.ipynb", "provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "Dr_Aishu_Training.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))

# Validate it parses back and nbformat accepts it.
reloaded = json.loads(out.read_text())
print(f"wrote {out} with {len(reloaded['cells'])} cells")
print("cell types:", [c["cell_type"][0] for c in reloaded["cells"]])
