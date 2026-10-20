# AGENTS.md — Dr. Aishu

Repository notes for future agent sessions. Keep this file current.

## What this is

`Dr. Aishu` is an educational medical triage assistant (FastAPI + Python).
It parses free-text symptoms and lab-report images, produces a differential
diagnosis, a severity/triage level, safe medicine suggestions with a timetable,
and a personalised daily routine.

## Commands

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000   # run the app
python -m pytest tests/ -q                            # run the suite
python scripts/seed_training_data.py --per-disease 60 # regenerate training data
python scripts/train_model.py                         # retrain (writes app/models)
python scripts/import_symptom_dataset.py              # import external dataset (coverage report)
python scripts/build_notebook.py                      # regenerate the Colab notebook
python scripts/verify_notebook.py                     # execute notebook cells headlessly
python scripts/test_learning_loop.py                  # end-to-end feedback->retrain check
```

`scripts/test_learning_loop.py` needs a live server and an API key
(`DR_AISHU_API_KEY`, with read/export/train scopes); it exits 2 with a hint
otherwise. It does not create its own key.

`python3` is `/usr/local/bin/python`. Tesseract is installed at `/usr/bin/tesseract`
(needed only for the image endpoint). If it is ever missing, the app still runs
and reports OCR as unavailable — but note the whole image-analysis path is then
inert, and `tests/test_vision.py` silently skips its OCR cases, so a bare `pytest`
run can look greener than the deployment actually is. Check `vision.TESSERACT`
before trusting an image-related pass.

## Architecture

- `app/kb.py` — loads and normalises the JSON knowledge base; `_normalise` keeps
  `.` and `/` (decimals, blood pressure, pain scales) and `_compile` is
  plural-tolerant with word-boundary lookarounds.
- `app/nlp.py` — `extract()` returns symptoms, duration, severity, vitals and
  demographics. All vitals regexes are case-insensitive (`39.2C`, `102F`).
  Negation is scoped: a cue only applies when it attaches to the symptom, and it
  distributes correctly across "and" vs "or"/comma lists.
- `app/analyzer.py` — `diagnose()` and `assess_severity()`. Red-flag floors are
  applied **after** additive bonuses, so a numeric fever alone cannot escalate
  to emergency. Confidence is capped by the amount of evidence.
  - The leading condition's `severity_score` is not applied at full strength.
    It is discounted toward `_NEUTRAL_SEVERITY` in proportion to how many
    symptoms actually matched, and the discount is **one-sided**: it can lower a
    severe guess but never raise a mild one. Without it, a single shared symptom
    let a moderate fever inherit malaria's 8/10 and triage to `emergency` with
    no red flags — the exact failure the red-flag comment above warns about.
    `_SEVERITY_SUPPORT_SYMPTOMS` is the match count at which the profile is
    applied in full. Red-flag floors still run afterwards, so a real emergency
    escalates regardless. Pinned by
    `test_thin_evidence_does_not_inherit_a_severe_condition`,
    `test_severe_condition_still_escalates_when_well_supported` and
    `test_mild_condition_is_not_inflated_by_thin_evidence`.
- `app/pipeline.py` — orchestration. **The emergency gate lives here**: when
  triage is `emergency`, `medicines` is emptied. This is the safety invariant.
- `app/solutions.py` — per-problem action plan (`solutions` in the response).
  Every supportive step comes from the curated `SYSTEM_SUPPORT` map keyed by
  symptom system; nothing is invented. `covered_by` links a symptom to medicines
  by whole-word matching the medicine's free-text `uses` (there is **no**
  `indications` field). Red-flag symptoms sort first, ahead of key symptoms.
  In an emergency the plan is still built but with `prescription=None`, so no
  medicine is ever attached to a self-treatment step.
- `app/store.py` — SQLite persistence. Cases, feedback, search events and
  consented image uploads. `save_upload()` writes the bytes under
  `data/uploads/` and the row carries the scrubbed OCR text; `upload_examples()`
  joins uploads to cases to produce labelled training rows. `delete_case()` and
  `purge_all()` unlink the files as well as the rows.
- `app/learning.py` — TF-IDF classifier (default) with optional transformer
  backend. `train_tfidf` reports a held-out score **and** `cv_accuracy`/`cv_std`;
  trust the cross-validation number, the single split is noisy.
- `app/vision.py` — `_preprocess_for_ocr` (greyscale, upscale to 1800px,
  autocontrast) then Tesseract. The TSV confidence column index is read from the
  header (`conf` is column 10), never hard-coded.

## Knowledge-base invariants (enforced by `tests/test_knowledge_base.py`)

- Every `symptoms` weight must be **> 0**; a zero weight silently contributes
  nothing. This bit us before on `stroke.facial_droop`, `covid19.loss_of_taste`,
  `migraine.light_sensitivity`.
- Every `key_symptoms` entry must also appear in `symptoms`, and no condition
  may have an empty `key_symptoms`.
- Medicine and routine references must resolve.
- Every symptom id referenced must exist in `symptoms.json`.

When adding a condition, set `severity_score` (1-10), a `routine`, `self_care`,
`when_to_see_doctor`, and leave `medicines` empty for severity 9-10.

## Testing notes

- `tests/conftest.py` isolates the trained model so `test_learning.py` cannot
  leak state into `test_pipeline.py`.
- Vision/OCR tests skip automatically when Tesseract is absent, and use a
  TrueType font (DejaVuSans) because PIL's bitmap font garbles digits.
- `tests/test_deployment.py` checks the container and host configs (Dockerfile,
  `render.yaml`, `fly.toml`, `Procfile`, `.env.example`) agree with each other
  and with the env vars the app reads. These assertions were mutation-tested:
  each one fails when its config line is broken.
- Baseline: 279 tests pass.

## Auth and tiers

- `app/auth.py` — hashed API keys (SHA-256, plaintext shown once), tiers
  `anonymous`/`free`/`pro`, and per-key usage counters.
- `app/security.py` — `require_scope(scope)` builds a dependency that
  authenticates and checks one scope. **Pass `None` for authentication without a
  scope check** (`require_identity`); `/api/keys/me` needs this so the browser's
  anonymous key can read its own scopes. Requiring the `keys` scope there made
  the web UI show Pro-only buttons to visitors who could not use them.
- The `anonymous` tier is deliberately absent from `/api/keys/tiers`; the
  browser gets one implicitly, keyed per client IP.
- Free-tier data is ephemeral on hosts without a volume. Pin
  `DR_AISHU_BOOTSTRAP_KEY` for a permanent Pro key that survives redeploys.
- Docs on `/developers` must name real response fields. `medicines` has
  `timetable` (not `schedule`); routine blocks have `time`/`title` (not
  `activity`). `test_developers_page_documents_real_response_fields` guards this.

## Gotchas

- Jinja2 `TemplateResponse` needs the request first:
  `templates.TemplateResponse(request, "index.html")`.
- Pillow: use `get_flattened_data()` (with fallback), not the removed `getdata()`.
- Endpoint spelling is American: `/api/analyze`, `/api/analyze-image`. The image
  upload field is named `image`.
- Analysis results are top-level keys (`severity`, `diagnoses`, `medicines`,
  `routine`), not nested under an `analysis` key.
- Free text is stored only when `consent_share` is true.
- **API key storage slot is `sessionStorage["draishu_dev_key"]`** (written by
  `developers.js`). Every page that authenticates — `app.js`, `insights.js` —
  must read that exact slot and storage. `insights.js` originally read
  `localStorage["dr_aishu_key"]`, so a key issued on the developers page was
  never found and every chart silently showed an empty state.
- Routine blocks have `time`/`title`/`detail`/`category`. `"activity"` is a
  *category value*, not a key — `scripts/build_deploy_notebook.py` printed
  `block["activity"]` and raised `KeyError` on the first notebook run.
- Renderers in `static/app.js` return a string of HTML or `""`; add new result
  sections to the `render()` parts list, not to a template.
- The product catalogue (`app/data/medicine_index.json`, ~22 MB) is a
  **presentation-only** search layer. It must never influence which medicine is
  chosen: selection reads `medicines.json` and the interaction rules only. The
  index is keyed by molecule and pre-sorted best-first on disk; `_order()` in
  `app/kb.py` relies on that, so do not re-sort at request time. It is loaded
  lazily and cached — a fresh `KB()` must not parse it until `search_products`
  is called.
- Indic symptom matching is not a plain word match. Nouns carry glued case
  markers ("ఛాతీలో" = "in the chest"), so `_compile()` in `app/kb.py` allows a
  short trailing suffix and one intervening word for non-Latin terms. That gap
  must not span a conjunction (`_CONJUNCTION_PATTERN`), or "severe fever and
  headache" matches the "severe headache" term and invents a symptom. English
  terms keep strict boundaries, so "painting" never matches "pain".
- Red-flag symptoms need several natural phrasings per language, not one
  clinical form. `scripts/extend_symptom_terms.py` adds the common variants; a
  single phrasing misses the everyday description a patient actually types, and
  for chest pain or breathlessness that means the emergency is never raised.

## Performance notes

Latency work here is about not doing the same work twice, and about moving
one-off work off the request path. Every change below is pinned by a test that
compares it against the slow, obvious implementation, so a "faster" version that
changes a result fails the suite.

- **Compile regexes once, not per call.** Three separate hot spots all did
  `re.compile`/`re.search(rf"...{re.escape(term)}...")` inside a loop:
  `_find_window_negated` (per candidate match), `extract_severity` and the
  duration phrase loop, and `_USE_MATCH_RE` in `app/solutions.py` (per
  term/use pair). Use the `_word_re()` helper in `app/nlp.py` or a per-term
  cache. Profiling a warm request is the way to find these: `ncalls` on
  `re._compile` climbing with request count is the signature.
- **Symptom patterns are compiled lazily, per script.** `_compile` on all ~2000
  synonyms cost ~1.6 s at import. `Symptom.term_patterns` is now a property that
  compiles on first access, and `KB.patterns_for(script)` compiles one script
  bucket at a time. `KnowledgeBase.prewarm()` compiles every bucket; `main.py`
  calls it from a daemon thread in the lifespan so the first request never pays
  it. Do not reintroduce eager compilation, and do not iterate
  `KB.patterns_by_script` — that attribute no longer exists.
- **Index by script, and never widen the boundary class.** `patterns_for` keys
  on `script_of(term)` so an English request skips the Indic patterns. Do not
  narrow the boundary class to the term's own script as a further "optimisation":
  it was measured at ~600 ms saved but changes behaviour on mixed-script text
  (`"sore"` starts matching inside `"कान में दर्दsore eye"`), and a missed or
  invented symptom is a clinical error, not a perf win.
- **Medicine coverage is cached per (medicine, symptom) pair, not per medicine.**
  Testing every symptom against every medicine on first use was ~13k regex
  compiles and the largest single request cost. `_covered_symptoms` only tests
  the symptoms the request actually reported.
- **SQLite runs WAL + `synchronous=NORMAL`.** `store._connect()` sets both.
  WAL alone still fsyncs every commit, which made recording an analysis the
  dominant cost of a request (~5.5 ms of a 6 ms request). The pairing is the
  documented one and survives a process crash; a power loss can lose the last
  few rows, which is fine for usage counters and audit rows. Pinned by
  `test_connect_uses_the_wal_durability_pairing`.
- **OCR runs tesseract once.** `ocr_text()` takes a single TSV pass and rebuilds
  the text in `_text_from_tsv()` rather than running tesseract twice for plain
  text and boxes.
- Baseline after this work: cold import + `KB()` ~180 ms (was ~1.8 s), first
  HTTP request ~60 ms, warm `/api/analyze` ~3-6 ms in-process, OCR ~250 ms.

## Uploaded data kept for training

- **Consent is the only gate.** `store.save_upload()` writes nothing unless
  `consent_share=True`, and `pipeline.analyse()` calls it only when the same flag
  that governs the case row is set. There is no path that stores an image the
  user did not opt into.
- **Original filenames are discarded.** The file lands in `data/uploads/` under a
  random id with an extension derived from the content type, because upload
  filenames routinely contain the patient's name (`priya_sharma_report.png`).
  `_SAFE_EXT` maps the allowed types; anything else becomes `.bin`.
- **OCR text is scrubbed before it is stored**, exactly like free text, and the
  clinical values survive (`test_ocr_text_is_scrubbed_before_it_is_stored`).
- **The image is linked to its label through the case id.** `upload_examples()`
  joins `uploads` to `cases` to get `top_diagnosis`, so a photographed report
  becomes a training row without a second consent step.
- **Erasure removes bytes, not just rows.** `delete_case()` and `purge_all()`
  unlink the files; a row-only delete would leave the image on disk.
- **Write failures never fail the request.** A full disk or a locked database
  returns `{"stored": False, ...}` from `save_upload` rather than raising, since
  the user is waiting on the analysis, not the archival.
- `data/uploads/` is in `.gitignore` and `.dockerignore`; on a host with a
  persistent volume it lives beside the database under `DR_AISHU_DATA_DIR`.

## Consolidated dataset workbook and image models

- **The workbook is a build artefact, not source.** `scripts/fetch_sources.py`
  downloads the raw material, `scripts/build_dataset_workbook.py` writes
  `data/external/dr_aishu_dataset.xlsx`, and `app/dataset.py` only reads it.
  `data/external/` is gitignored, so a fresh clone reports the dataset as
  unavailable rather than failing: `dataset.available()` is False and every
  lookup returns empty. `openpyxl` is imported inside the loader for the same
  reason - it is a build dependency, not a server one.
- **The sheet and column names are a contract.** `dataset.REQUIRED_SHEETS` lists
  every sheet and the columns the API reads by name. Add a column there when the
  builder starts writing one, or `status()["missing_columns"]` will not warn.
- **`Images.modality` is what pins an image to a model.** The vision path takes
  an explicit `image_modality` (`eye` or `skin`); there is deliberately no
  auto-detection. An eye model asked about a skin photo answers with ~1.0
  confidence and the wrong label, so `image_model.predict()` without a modality
  reports both models and sets `best = None` rather than guessing. Do not add
  cross-modality auto-pick.
- **The classifier refuses to name a class below `MIN_CONFIDENCE` (0.55).** A
  low-confidence answer is worse than none in a medical app. `uncertain` is set
  and `label` is None; the UI must show the probabilities, not a diagnosis.

- **Skin accuracy is honestly ~0.40.** The six skin conditions overlap heavily in
  colour and texture and the feature vector is colour/texture only (no torch, no
  cv2 in this environment). Do not present it as a reliable diagnosis, and do not
  "fix" it by reporting the training accuracy. Eye accuracy is ~0.85.
- **Training must not leak a patient across the split.** Skin images are grouped
  by `patient_id` (from `skin_metadata.csv`) so the same patient never appears in
  both train and test; the held-out score is a real one. `train_image_model.py`
  owns this.
- **The model is compressed at save time** (`compress=3`, 131 MB to 2.9 MB).
  `data/models/` is gitignored; retrain with
  `python scripts/train_image_model.py`.

## Training data from the workbook

- **Workbook rows are symptom lists, and the classifier learns from prose.**
  `scripts/import_workbook_training.py` reads the 67 `in_kb` conditions, phrases
  their symptoms the way a patient would (reusing the openers and connectors from
  `seed_training_data.py`), and writes `data/training/workbook_2024.jsonl`.
- **This file is included by default; the raw symptom-list imports are not.**
  Measured on a held-out fifth of the seed set across five splits, adding the
  workbook prose raised accuracy in every split (0.8256-0.8314 to 0.8333-0.8663),
  which is why `collect()` loads it unconditionally under `use_seed`. The
  `--use-imports` flag still governs the raw `*.jsonl` imports, which were
  measured to lower accuracy. Re-run the comparison before changing either.
- **Only Latin-script synonyms are used.** The knowledge base's synonyms now
  carry Indic translations; a phrase drawn from one would teach the classifier
  that a Devanagari word predicts a condition. `_phrase()` filters on
  `ord(ch) < 0x250`.
- **`workbook_2024.jsonl` is gitignored**, like `mendeley_2023.jsonl`: it is
  reproducible from the script. `seed.jsonl` stays committed - it is the
  hand-written bootstrap the import is measured against.

## Operator-only surfaces (security hardening)

- **`/dataset` is an operator page, not a public one.** It exposes build
  internals (archive hashes, class counts, licences), so its five API endpoints
  require the `operator` scope and the nav link is gone. The page shows a key
  gate (`#ds-gate`) and keeps its content (`#ds-content`) `hidden` until an
  operator key unlocks it.
- **Tier escalation is closed.** `/api/keys/signup` always issues the `free`
  tier (a `tier=pro` request gets 403), and `free` no longer carries the
  `export` or `train` scopes. Only a bootstrap key (`DR_AISHU_BOOTSTRAP_KEY`)
  or a key minted out-of-band is an operator.
- **Public responses are scrubbed.** `_public_dataset_status()`,
  `_public_image_model()` and `_public_stats()` in `app/main.py` strip server
  filesystem paths, workbook hashes and internal counts from anything a
  non-operator can reach. Keep new fields out of the public helpers.
- **`.gate[hidden] { display: none }` in `dataset.css` is load-bearing.** An
  author `display: grid` outranks the UA rule for `[hidden]`, so without it the
  gate stays on screen after a successful unlock. `static/dataset.css` is
  self-contained on purpose: the page loads only `style.css` + `dataset.css`,
  so it must not assume rules that live in `insights.css`/`developers.css`.
- **The learning/training UI is gone from `/assess`.** `loadLearningStats`,
  the learning panel and the retrain button were removed from `assess.html` and
  `app.js`; the endpoints behind them are operator-only.
- **The dataset gate reuses the shared key slot**
  `sessionStorage["draishu_dev_key"]` written by `developers.js`, so a key
  created at `/developers` is picked up by the gate automatically.

## External sources: licences, Box and what not to download

- **Every source must declare a licence, even `unknown`.** `tests/test_fetch_sources.py`
  fails on a missing licence, and `status()`/`Sources` sheet surface it. "unknown"
  means all rights reserved until the uploader says otherwise - do not treat a
  blank licence as permission.
- **Box shared folders need the `shared_name` token, not the folder id.** Box has
  no anonymous REST API for a shared folder; the web app POSTs to
  `box_download_shared_file` with the opaque `shared_name` from the page's
  `Box.postStreamData`. `fetch_box_shared()` reads it from the source config, so
  it is pinned in `SOURCES` rather than scraped. A direct `api.box.com` call with
  the folder id returns 403.
- **DeepLesion is metadata-only here, on purpose.** The image archives are ~238 GB
  (`Images_png`) and ~4.9 GB (`Key_slices.zip`); this environment has ~21 GB free.
  Only `DL_info.csv` and the docs are fetched. `test_deeplesion_does_not_fetch_the_image_archive`
  guards the file list. The readme states no redistribution licence, so the sheet
  is tagged `nih-research-use`.
- **DeepLesion stays out of symptom training.** It is 32,735 radiology lesion
  annotations (bounding box, RECIST size, coarse lesion type), not symptom text.
  It has its own `DeepLesion` sheet so it can never be mistaken for a disease
  whose symptoms the classifier should learn. `coarse_lesion_type` is a numeric
  code; report the count, do not invent a label for the code.
- **The PMC FTP root now serves one dataset file.** The 2026 change moved the
  article bundles to the PMC Cloud Service; `PMC-ids.csv.gz` (an NLM identifier
  table, no article text) is what remains, and it is what we fetch. Do not add
  `oa_bulk`/`oa_file_list` paths - they 404.
- **`Dataset_Catalogue` holds facts, not prose.** It is the public-dataset table
  from the CC BY review PMC8950225 (dataset name, what it covers, URL). The
  article text itself is not copied; the review's licence covers the table, and
  attribution is recorded in the `Sources` sheet.
- **`OPTIONAL_SHEETS` in `app/dataset.py` keeps old workbooks loading.** The new
  `DeepLesion`/`Dataset_Catalogue` sheets are read when present but are not part
  of `REQUIRED_SHEETS`, so a workbook built before them still passes `available()`.
- **Google Research Health AI is a reference URL only.** It publishes no dataset
  and states no licence; it is recorded with `kind: "note"` and has no file.

