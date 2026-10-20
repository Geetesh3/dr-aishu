# Dr. Aishu

An educational health assistant that turns a plain-language symptom description,
or a photo of a lab report, into a structured triage result: differential
diagnosis, severity level, medicine suggestions with a daily timetable, and a
personalised routine.

> **Dr. Aishu is an educational triage aid, not a medical diagnosis or a
> prescription.** It cannot examine you and it does not replace a clinician. In
> an emergency, contact your local emergency number immediately.

---

## What it does

| Capability | How it works |
| --- | --- |
| Symptom extraction | Rule-based NLP with negation, hedging, list distribution and duration handling |
| Differential diagnosis | Knowledge-base scoring plus a trained TF-IDF classifier, blended and capped by evidence |
| Severity and triage | Red-flag rules, vitals thresholds, age and pregnancy modifiers |
| Medicine suggestions | Curated formulary, filtered by age, pregnancy, allergies and interactions |
| Medicine timetable | Per-drug dosing times merged into a single daily schedule |
| Daily routine | Condition-specific routine blocks (meals, activity, sleep, monitoring) |
| Action plan | Per-problem steps for each reported symptom, ordered with warning signs first |
| Image analysis | Any format Pillow reads, plus HEIC/HEIF and PDF; Tesseract OCR of reports plus colour and texture heuristics |
| Search insights | Aggregate analytics dashboard over recorded analyses (charts, brands) |
| Learning loop | Consented feedback becomes weighted training data; the model can be retrained |
| Dataset import | Turns a large disease/symptom dataset into coverage reports and optional training rows |
| Consolidated dataset | Downloads every external source once, hashes it, and merges it into one Excel workbook browsable at `/dataset` |
| Trained image model | Eye and skin photo classifiers, pinned to a modality, refusing to name a class below 0.55 confidence |

Severity is reported on four triage levels: `self_care`, `see_doctor`, `urgent`
and `emergency`.

### Supported image formats

Uploads are sniffed by their leading bytes, not trusted by content type, then
decoded and re-encoded to a single RGB PNG so every downstream step sees one
format. Multi-page inputs (TIFF, animated GIF, PDF) are stacked so no page is
silently dropped.

| Group | Formats |
| --- | --- |
| Always | JPEG, PNG, GIF, BMP, TIFF, WebP, ICO, PPM/PGM/PBM, TGA, PCX, and anything else Pillow is built with |
| With `pillow-heif` | HEIC, HEIF, AVIF (iPhone and modern camera photos) |
| With `pypdfium2` | PDF (scanned or exported reports), rendered page by page |

Both optional decoders are listed in `requirements.txt`. If one is absent the
format is reported as unsupported rather than crashing.

### Safety rules built into the pipeline

- **No medicines in an emergency.** If triage is `emergency`, the medicine list
  is emptied and the advice is to seek immediate care.
- **Abstain rather than guess.** Thin evidence produces no diagnosis instead of a
  confident wrong one, and a confidence cap prevents high scores from weak input.
- **Red flags win.** Emergency vitals and red-flag symptoms raise triage even
  when the average symptom picture looks mild.
- **Free text is only stored with consent.** Cases are kept for learning only
  when the caller opts in, and a case can be deleted on request. An uploaded
  image is kept under `data/uploads/` only when the same consent is given; the
  original filename is discarded and the text read from the image is scrubbed of
  identifiers before it is saved. Deleting a case deletes its image too.
- **Retention is a deployment choice, and provenance is always recorded.** Set
  `DR_AISHU_RETENTION=all` to keep narrative data (free text and images) for
  every request instead of only consented ones. Identifier scrubbing still
  applies, and every row records whether the user consented separately from
  whether the policy retained it, so a policy-kept row is never counted as
  consented. An unrecognised value falls back to the stricter `consented`.

### Data retention policies

| `DR_AISHU_RETENTION` | Free text and images | Structured clinical signal |
| --- | --- | --- |
| `consented` (default) | Stored only when the request opts in | Always stored |
| `all` | Stored for every request, scrubbed of identifiers | Always stored |

Structured signal means the canonical symptom ids, numeric observations, triage
level and reported diagnosis -- never the user's own words. That is why searches
made without consent can still improve the classifier without storing anything
the user typed.

---

## Quick start

```bash
cd /workspace/project
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000> for the web interface, or
<http://127.0.0.1:8000/docs> for the interactive API reference.

The repository ships with a trained model, so the app works immediately. To
rebuild it from scratch:

```bash
python scripts/seed_training_data.py --per-disease 60
python scripts/train_model.py
```

OCR needs the Tesseract binary. If it is missing, the app still works and the
image endpoint simply reports that OCR is unavailable.

```bash
sudo apt-get install -y tesseract-ocr   # Debian/Ubuntu
brew install tesseract                  # macOS
```

### Optional: building the consolidated dataset workbook

`/dataset` reads a single Excel workbook built from every external source. The
raw downloads are never committed; the workbook is a build artefact, so a fresh
clone runs fine without it and simply reports the dataset as unavailable.

```bash
pip install openpyxl
python scripts/fetch_sources.py            # download + hash every source
python scripts/build_dataset_workbook.py   # merge into one workbook
python scripts/train_image_model.py        # train the eye/skin classifiers
python scripts/import_workbook_training.py # narrative training rows (optional)
```

The workbook lands at `data/external/dr_aishu_dataset.xlsx` with a sheet per
source family (`Diseases`, `Symptoms`, `Disease_Symptoms`, `Precautions`,
`Images`, `MedicineNet`, `Sources`). `scripts/import_workbook_training.py`
phrases the in-knowledge-base conditions the way a patient would and writes
`data/training/workbook_2024.jsonl`; `scripts/train_model.py` loads that file by
default because it measurably raises held-out accuracy.

Licences differ per source and three declare none at all (all rights reserved by
default). `/dataset` shows the licence and the SHA-256 of every source so a row
can always be traced back to the exact bytes it came from. SkinDisNet is
CC BY-NC 4.0: non-commercial use only. MedicineNet is copyright WebMD and its
robots.txt disallows the AI crawlers, so only its public condition index (name,
slug, link) is stored - no article text or image is copied.

### Optional: importing an external symptom dataset

`scripts/import_symptom_dataset.py` reads a wide disease-by-symptom CSV (by
default the Mendeley "Disease and symptoms dataset 2023", DOI
`10.17632/2cxccsxydc.1`, downloaded to `data/external/`) and produces:

- `data/external/import_report.json` -- what mapped, what did not, and why;
- `data/external/disease_symptom_rates.json` -- per-disease symptom rates, as a
  descriptive reference (the dataset's rates cluster near 0.5, so they are not
  calibrated enough to replace the knowledge base's hand-set weights);
- `data/training/mendeley_2023.jsonl` -- labelled examples.

```bash
python scripts/import_symptom_dataset.py
```

The importer only maps what it can justify. Names are matched by id, by display
name, and by an explicit alias table; anything else is reported as a coverage
gap rather than guessed at. Only rows whose disease maps to a knowledge-base
condition become examples, so the classifier's label space stays the knowledge
base.

The examples are **not** mixed into training by default. Measured against a
held-out fifth of the seed set, adding them lowered top-1 accuracy from 0.601 to
0.523 (restricting to rows the knowledge base already agrees with still left it
at 0.572): the imported rows are symptom lists rather than narrative, and they
pull the TF-IDF vocabulary away from the phrasing real users type. Pass
`--use-imports` to `scripts/train_model.py` to opt in, and re-measure first.

---

## Using it from your own app

### HTTP API

```bash
curl -s -X POST http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"high fever and body aches and chills for 3 days with a dry cough"}'
```

```python
import requests

r = requests.post(
    "http://127.0.0.1:8000/api/analyze",
    json={"text": "burning when I pee and going very often, lower abdominal pain"},
)
data = r.json()

print(data["severity"]["triage"])          # 'self_care'
print(data["diagnoses"][0]["name"])        # leading diagnosis
print(data["medicines"]["recommended"])    # safe suggestions (empty if emergency)
print(data["routine"]["blocks"])           # daily timetable
print(data["solutions"]["problems"])       # what to do about each symptom
```

An image can be sent on its own or alongside text:

```python
with open("lab_report.png", "rb") as fh:
    r = requests.post(
        "http://127.0.0.1:8000/api/analyze-image",
        files={"image": fh},
        data={"text": "reports fever for four days"},
    )
```

### In Google Colab

The notebook `notebooks/Dr_Aishu_Training.ipynb` runs the whole project in Colab:
install dependencies, load the knowledge base, generate training data, train,
evaluate, and try the assistant on new sentences. It ends with two optional
cells for pulling live data from a running app and for serving the app itself.

Upload the notebook to Colab, set the runtime to CPU (a GPU only helps the
optional transformer path), and run the cells in order.

### Host it for free from Colab

`notebooks/Dr_Aishu_Deploy.ipynb` turns a Colab runtime into a public HTTPS URL
for the full web UI, with no hosting account and no credit card. It clones the
repository, installs dependencies, retrains if you ask it to, starts the server
and opens a tunnel:

1. Open the notebook in Colab (upload it, or open it from GitHub).
2. Edit the first cell and set `REPO_URL` to your own clone.
3. Run the cells in order. The last cell prints a `https://…` URL that serves
   Dr. Aishu to any browser, including a phone.

The tunnel URL lives only while the Colab cell keeps running. For a permanent
address, use one of the container hosts below.

### Permanent free hosting

The repository ships ready-to-use container configuration. All three options
read the same environment variables, so a local `.env` works everywhere.

| File | Host | Notes |
| --- | --- | --- |
| `Dockerfile` | any | Installs Tesseract, runs `uvicorn` on `$PORT`. |
| `render.yaml` | Render | Blueprint for the free web-service tier. |
| `fly.toml` | Fly.io | Free allowance plus a volume at `/data`. |
| `Procfile` | Heroku-style | `web: uvicorn … --port $PORT`. |

```bash
# Any Docker host
docker build -t dr-aishu .
docker run -p 8000:8000 -v dr_aishu_data:/data dr-aishu
```

Free tiers sleep when idle and have an ephemeral disk, so the case database and
issued keys reset on restart. Pin `DR_AISHU_BOOTSTRAP_KEY` to keep one permanent
admin key across redeploys:

```bash
python -c "import secrets; print('dra_' + secrets.token_urlsafe(32))"
```

### As a Python library

```python
from app import pipeline

result = pipeline.analyse(
    text="itchy red circular rash on my arm spreading over 2 weeks",
    age=34,
    pregnant=False,
    allergies=["penicillin"],
)
print(result["diagnoses"][0]["name"], result["severity"]["triage"])
```

---

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Web interface |
| `GET` | `/insights` | Search-insights dashboard (aggregate analytics) |
| `GET` | `/developers` | Developer portal: issue and manage API keys |
| `GET` | `/api/health` | Status, knowledge-base sizes and active model |
| `GET` | `/api/symptoms` | Symptom catalogue |
| `GET` | `/api/diseases` | Condition catalogue |
| `GET` | `/api/medicines` | Formulary |
| `POST` | `/api/analyze` | Analyse free text |
| `POST` | `/api/analyze-image` | Analyse an uploaded image, optionally with text |
| `POST` | `/api/feedback` | Report a correction or rating for a case |
| `GET` | `/api/model/status` | Model metadata and metrics |
| `GET` | `/api/data/stats` | Stored case and feedback counts |
| `GET` | `/api/analytics/summary` | Aggregate counts, confidence and duration |
| `GET` | `/api/analytics/timeline` | Searches per day over a window |
| `GET` | `/api/analytics/top-symptoms` | Most-reported symptoms |
| `GET` | `/api/analytics/top-diagnoses` | Leading diagnoses and shares |
| `GET` | `/api/analytics/triage-mix` | Triage-level breakdown |
| `GET` | `/api/analytics/severity` | Severity score distribution |
| `GET` | `/api/analytics/brands` | Indian brand catalogue with pack prices |
| `POST` | `/api/analytics/demo` | Generate demo data for the charts |
| `DELETE` | `/api/analytics/demo` | Remove previously generated demo data |
| `GET` | `/api/export/training-data` | Export consented cases and uploads as JSONL |
| `POST` | `/api/model/train` | Retrain in the background |
| `DELETE` | `/api/data/case/{case_id}` | Delete a stored case |

---

## Project layout

```
app/
  kb.py            knowledge-base loading, normalisation and matching
  nlp.py           symptom, duration, severity, vitals and negation extraction
  analyzer.py      differential diagnosis and severity scoring
  learning.py      TF-IDF / transformer training and prediction
  prescription.py  medicine selection and timetable building
  routines.py      daily routine assembly
  solutions.py     per-problem action plan for the leading condition
  vision.py        OCR and image feature analysis
  pipeline.py      end-to-end orchestration
  store.py         consented case and feedback storage
  main.py          FastAPI application and endpoints
  data/            symptoms, diseases, medicines and routines JSON
static/            web UI: app.js, insights.js, style.css
templates/         Jinja2 pages: index, insights, developers
scripts/           data generation, training and notebook tooling
notebooks/         Google Colab training notebook
tests/             test suite
```

---

## How the accuracy is measured

A single train/test split is noisy, so training reports both a held-out score
and k-fold cross-validation:

```json
{
  "labels": 43,
  "accuracy": 0.851,
  "cv_accuracy": 0.847,
  "cv_std": 0.009
}
```

`cv_accuracy` is the number to trust. The knowledge base also has integrity
tests that fail the build if a symptom is wired in at zero weight, a key symptom
is not scored, or a condition references a medicine or routine that does not
exist.

---

## Testing

```bash
python -m pytest tests/ -q
```

---

## Limitations

- The knowledge base covers 43 common conditions. Rare presentations may be
  missed, and the assistant abstains when evidence is thin.
- OCR quality depends on image resolution and lighting. Preprocessing helps, but
  a blurry photo can still misread a value.
- Doses are generic adult and child guidance. A clinician must confirm anything
  that matters.
- The transformer backend is optional and untested here; the default TF-IDF
  backend is what the reported metrics cover.
