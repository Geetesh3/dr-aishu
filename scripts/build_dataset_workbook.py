"""Build Dr. Aishu's consolidated dataset workbook from the fetched sources.

Reads everything `scripts/fetch_sources.py` downloaded and writes one Excel file
with a sheet per concern, so the whole dataset can be opened, filtered and
audited without touching the raw archives:

    Diseases            one row per condition, merged from all text sources
    Symptoms            one row per symptom, with severity weight and red-flag state
    Disease_Symptoms    long form (disease, symptom, weight) - the training shape
    Precautions         per-disease precaution steps
    Images              one row per image, with its class and licence
    MedicineNet         condition index (names and links only - see the note)
    Sources             provenance for every source: url, licence, sha256, bytes
    Read_Me             column definitions and licence warnings

The workbook is a build artefact: it is regenerated from the manifest, so the
raw downloads stay the source of truth and every cell can be traced to a file
whose SHA-256 is recorded.

Licence note, because it decides what may be reused:
  * HuggingFace CSV, the GitHub repo and the Mendeley eye set declare **no
    licence**. They are included for research/reference use; treat them as
    all-rights-reserved until the uploader says otherwise.
  * SkinDisNet is CC BY-NC 4.0 (non-commercial).
  * MedicineNet is copyright WebMD and its robots.txt disallows the AI crawlers,
    so only its public condition index (name, slug, link) is stored here. No
    article prose or image bytes are copied in.

Run:
    python scripts/build_dataset_workbook.py
    python scripts/build_dataset_workbook.py --out data/external/dr_aishu_dataset.xlsx
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCES_DIR = ROOT / "data" / "external" / "sources"
MANIFEST = SOURCES_DIR / "sources.json"
DEFAULT_OUT = ROOT / "data" / "external" / "dr_aishu_dataset.xlsx"

# Column order is part of the contract: app/dataset.py reads these sheets by
# header name, and tests/test_dataset_workbook.py asserts they are present.
DISEASE_COLUMNS = [
    "disease_id", "name", "category", "icd10", "prevalence", "contagious",
    "severity_profile", "severity_score", "duration_min_days", "duration_max_days",
    "description", "key_symptoms", "symptoms", "self_care", "medicines", "routine",
    "when_to_see_doctor", "treatments", "precautions", "in_kb", "sources",
]
SYMPTOM_COLUMNS = [
    "symptom_id", "label", "system", "severity_weight", "red_flag",
    "red_flag_level", "weight", "synonyms", "in_kb", "sources",
]
DISEASE_SYMPTOM_COLUMNS = ["disease", "disease_id", "symptom", "symptom_id", "weight", "source"]
PRECAUTION_COLUMNS = ["disease", "precaution_1", "precaution_2", "precaution_3", "precaution_4", "source"]
IMAGE_COLUMNS = [
    "image_id", "file", "diagnosis", "modality", "split", "patient_id",
    "age", "sex", "lesion_location", "bytes", "source", "license", "sha256_of_archive",
]
MEDICINENET_COLUMNS = ["condition", "slug", "url", "letter", "source", "license", "note"]
DEEPLESION_COLUMNS = [
    "file_name", "patient_index", "study_index", "series_id", "key_slice_index",
    "measurement_coordinates", "bounding_boxes", "lesion_diameters_px",
    "normalized_lesion_location", "coarse_lesion_type", "possibly_noisy",
    "slice_range", "spacing_mm_px", "image_size", "patient_sex", "patient_age",
    "split", "source", "license", "sha256_of_archive",
]
DATASET_CATALOGUE_COLUMNS = [
    "study_refs", "disease", "dataset", "url", "source", "license", "note",
]
SOURCE_COLUMNS = [
    "id", "title", "url", "license", "doi", "status", "bytes",
    "sha256", "fetched_at", "note",
]


def _norm(text: str) -> str:
    """Loose key for matching names across sources: lowercase alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


# The KB names its conditions clinically ("Meningitis (suspected)", "Type 2
# Diabetes (uncontrolled)"), so a normalised-name match only finds 16 of 67.
# These are the hand-checked equivalences to the HuggingFace vocabulary.
#
# It is a whitelist on purpose: fuzzy matching would pair the KB's
# `hypertension` with HF's "Ocular Hypertension" and `depression` with
# "Postpartum Depression", which are different conditions. Anything not listed
# stays unmatched rather than picking up the wrong treatments.
HF_ALIASES: dict[str, str] = {
    "influenza": "Influenza (Flu)",
    "acute_bronchitis": "Bronchitis",
    "pneumonia": "Pneumonia",
    "asthma": "Asthma",
    "sinusitis": "Acute Sinusitis",
    "gastroenteritis": "Gastroenteritis (Stomach Flu)",
    "gerd": "Gastroesophageal Reflux Disease (GERD)",
    "migraine": "Migraine",
    "tension_headache": "Tension Headache",
    "stroke": "Stroke",
    "type2_diabetes": "Type 2 Diabetes",
    "hypothyroidism": "Hypothyroidism",
    "anemia": "Anemia",
    "uti": "Urinary Tract Infection (UTI)",
    "kidney_stones": "Kidney Stone",
    "arthritis_oa": "Osteoarthritis",
    "rheumatoid_arthritis": "Rheumatoid Arthritis",
    "gout": "Gout",
    "fibromyalgia": "Fibromyalgia",
    "urticaria": "Urticaria (Hives)",
    "eczema": "Eczema",
    "fungal_skin": "Fungal Infection of the Skin",
    "cellulitis": "Cellulitis",
    "otitis_media": "Otitis Media",
    "anxiety_disorder": "Anxiety",
    "dengue": "Dengue Fever",
    "hemorrhoids": "Hemorrhoids",
    "parkinsons": "Parkinson Disease",
    "pcos": "Polycystic Ovarian Syndrome",
    "varicose_veins": "Varicose Veins",
    "scabies": "Scabies",
    "psoriasis": "Psoriasis",
    "insomnia_disorder": "Primary Insomnia",
    "panic_disorder": "Panic disorder",
    "sciatica": "Sciatica",
    "vaginal_candidiasis": "Vaginal Yeast Infection",
}


def _split_list(text: str) -> list[str]:
    """Split a comma-separated cell, trimming blanks."""
    return [p.strip() for p in (text or "").split(",") if p.strip()]


# --------------------------------------------------------------------------- #
# source readers
# --------------------------------------------------------------------------- #
def read_hf(path: Path) -> list[dict]:
    """400 diseases with symptoms and treatments."""
    out = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Name") or "").strip()
            if not name:
                continue
            out.append({
                "name": name,
                "symptoms": _split_list(row.get("Symptoms")),
                "treatments": _split_list(row.get("Treatments")),
                "source": "hf_diseases_symptoms",
            })
    return out


def read_github(dirpath: Path) -> dict:
    """The 41-disease set with descriptions, precautions and severities."""
    desc, prec, sev = {}, {}, {}
    p = dirpath / "symptom_Description.csv"
    if p.exists():
        with p.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                desc[row["Disease"].strip()] = row["Description"].strip()
    p = dirpath / "symptom_precaution.csv"
    if p.exists():
        with p.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                prec[row["Disease"].strip()] = [
                    row[f"Precaution_{i}"].strip() for i in range(1, 5)
                    if row.get(f"Precaution_{i}", "").strip()
                ]
    p = dirpath / "Symptom-severity.csv"
    if p.exists():
        with p.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    sev[row["Symptom"].strip()] = float(row["weight"])
                except (KeyError, TypeError, ValueError):
                    continue

    rows = []
    p = dirpath / "dataset.csv"
    if p.exists():
        with p.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("Disease") or "").strip()
                if not name:
                    continue
                syms = []
                for key, val in row.items():
                    if key.startswith("Symptom") and val and val.strip():
                        syms.append(val.strip())
                rows.append({"name": name, "symptoms": syms})
    return {"rows": rows, "descriptions": desc, "precautions": prec, "severity": sev}


def read_medicinenet_index() -> list[dict]:
    """Condition names and links from the public A-Z index pages.

    Only factual index metadata is taken - the condition name, its slug and the
    URL. Article prose and images are deliberately not copied: MedicineNet's
    robots.txt disallows the AI crawlers, and the text is copyright WebMD.
    """
    import html as H

    out, seen = [], set()
    for letter in "abcdefghijklmnopqrstuvwxy":
        p = SOURCES_DIR / "medicinenet" / "alpha" / f"alpha_{letter}.htm"
        if not p.exists():
            continue
        page = p.read_text(encoding="utf-8", errors="replace")
        for url, text in re.findall(
            r'href="(https?://www\.medicinenet\.com/[^"]+/article\.htm)"[^>]*>([^<]{2,160})</a>',
            page,
        ):
            slug = url.rstrip("/").rsplit("/", 1)[-2]
            if slug in ("diseases_and_conditions", "medications") or slug in seen:
                continue
            seen.add(slug)
            label = H.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text))).strip()
            out.append({
                "condition": label,
                "slug": slug,
                "url": url,
                "letter": letter.upper(),
            })
    out.sort(key=lambda r: (r["letter"], r["condition"].lower()))
    return out


def read_deeplesion_metadata(manifest: dict) -> list[dict]:
    """DeepLesion lesion annotations from `DL_info.csv`.

    DeepLesion is radiology ground truth: one row per annotated lesion (not per
    image, not per symptom), carrying the bounding box, RECIST diameters, the
    coarse lesion type and patient demographics. It is kept in its own sheet
    because it must not be mixed into symptom/disease training - the label space
    is unrelated. Only the CSV and docs are fetched; the ~238 GB image archive
    is not downloaded (see the source note in fetch_sources.py).
    """
    entry = manifest["sources"].get("deeplesion_metadata", {})
    csv_path = SOURCES_DIR / "deeplesion" / "DL_info.csv"
    if not csv_path.exists():
        return []
    license_ = entry.get("license", "nih-research-use")
    sha = entry.get("sha256", {})
    sha_csv = sha.get("DL_info.csv", "") if isinstance(sha, dict) else ""
    # One source row records a missing age as the literal string "NaN"; show it
    # as blank rather than leaking the placeholder into the sheet and the UI.
    blank_nan = lambda v: "" if (v or "").strip().lower() in ("nan", "none") else v
    out = []
    with csv_path.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            fname = (row.get("File_name") or "").strip()
            if not fname:
                continue
            out.append({
                "file_name": fname,
                "patient_index": row.get("Patient_index", ""),
                "study_index": row.get("Study_index", ""),
                "series_id": row.get("Series_ID", ""),
                "key_slice_index": row.get("Key_slice_index", ""),
                "measurement_coordinates": row.get("Measurement_coordinates", ""),
                "bounding_boxes": row.get("Bounding_boxes", ""),
                "lesion_diameters_px": row.get("Lesion_diameters_Pixel_", ""),
                "normalized_lesion_location": row.get("Normalized_lesion_location", ""),
                "coarse_lesion_type": row.get("Coarse_lesion_type", ""),
                "possibly_noisy": row.get("Possibly_noisy", ""),
                "slice_range": row.get("Slice_range", ""),
                "spacing_mm_px": row.get("Spacing_mm_px_", ""),
                "image_size": row.get("Image_size", ""),
                "patient_sex": row.get("Patient_sex", ""),
                "patient_age": blank_nan(row.get("Patient_age", "")),
                "split": row.get("Train_Val_Test", ""),
                "source": "deeplesion_metadata",
                "license": license_,
                "sha256_of_archive": sha_csv,
            })
    return out


def read_pmc_catalogue(manifest: dict) -> list[dict]:
    """Public disease-dataset catalogue from the CC BY review (PMC8950225).

    Table 18 of the review lists public datasets used for ML disease diagnosis
    with their URLs. It is factual metadata (dataset name, host, link), so it is
    safe to store under the article's CC BY licence; the article prose itself is
    not copied.
    """
    import html as H

    entry = manifest["sources"].get("pmc_ml_disease_diagnosis_review", {})
    page_path = SOURCES_DIR / "pmc_ml_disease_diagnosis_review.html"
    if not page_path.exists():
        return []
    page = page_path.read_text(encoding="utf-8", errors="replace")
    tables = re.findall(r"<table.*?</table>", page, re.S)
    if not tables:
        return []
    # The catalogue is the last table, identified by its Study/Disease/Dataset/URL header.
    catalogue = None
    for table in reversed(tables):
        head = re.sub(r"<[^>]+>", " ", table[:600]).lower()
        if "url" in head and "disease" in head and "dataset" in head:
            catalogue = table
            break
    if catalogue is None:
        return []
    out = []
    for tr in re.findall(r"<tr.*?</tr>", catalogue, re.S):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)
        cells = [
            H.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c))).strip()
            for c in cells
        ]
        if len(cells) < 4 or cells[0].lower().startswith("study"):
            continue
        refs, disease, dataset, url = cells[0], cells[1], cells[2], cells[3]
        if not url.lower().startswith("http"):
            continue
        # The cell is a list of citation links; stripping the tags leaves spaces
        # around the brackets and commas ("[ 41 , 42 ]"). Tidy them back up.
        refs = re.sub(r"\s*,\s*", ", ", refs)
        refs = re.sub(r"\[\s*", "[", refs)
        refs = re.sub(r"\s*\]", "]", refs)
        out.append({
            "study_refs": refs,
            "disease": disease,
            "dataset": dataset,
            "url": url,
            "source": "pmc_ml_disease_diagnosis_review",
            "license": entry.get("license", "cc-by-4.0"),
            "note": "Public dataset named in the review's dataset catalogue (Table 18).",
        })
    return out


def read_eye_images(manifest: dict) -> list[dict]:
    """2298 eye images in five class folders, from the Mendeley zip."""
    entry = manifest["sources"].get("mendeley_eye_images", {})
    zpath = SOURCES_DIR / "mendeley_eye_images.zip"
    if not zpath.exists():
        return []
    sha = entry.get("sha256", "")
    out = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            if len(parts) < 2:
                continue
            cls = parts[-2]
            out.append({
                "image_id": Path(parts[-1]).stem,
                "file": info.filename,
                "diagnosis": cls,
                "modality": "eye",
                "split": "raw",
                "bytes": info.file_size,
                "source": "mendeley_eye_images",
                "license": entry.get("license", "unknown"),
                "sha256_of_archive": sha,
            })
    return out


def read_skin_images(manifest: dict) -> list[dict]:
    """SkinDisNet: 1710 clinical images with patient metadata, plus augmentations.

    The archive nests a second zip; `fetch_sources.py` leaves the inner archive
    in place, so this reads it without extracting the images to disk.
    """
    entry = manifest["sources"].get("skindisnet_images", {})
    outer = SOURCES_DIR / "skindisnet_images.zip"
    if not outer.exists():
        return []
    sha = entry.get("sha256", "")

    inner_bytes = None
    with zipfile.ZipFile(outer) as z:
        for info in z.infolist():
            if info.filename.endswith("SkinDisNet_2.zip"):
                inner_bytes = z.read(info)
                break
    if inner_bytes is None:
        return []

    meta: dict[str, dict] = {}
    images: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as z:
        # Two passes: the metadata CSV can sit anywhere in the archive, so read
        # it first. Reading it while walking the images left `meta` empty and
        # every row fell back to the folder name as its diagnosis.
        for info in z.infolist():
            if info.filename.endswith("SkinDisNet_Metadata.csv"):
                text = z.read(info).decode("utf-8-sig", errors="replace")
                for row in csv.DictReader(io.StringIO(text)):
                    meta[(row.get("Image_id") or "").strip()] = row
        for info in z.infolist():
            if info.is_dir():
                continue
            if not info.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = info.filename.split("/")
            if len(parts) < 3:
                continue
            split_dir, cls_dir = parts[0], parts[1]
            stem = Path(parts[-1]).stem
            # Augmented files are "<id>_NN"; map back to the source image id.
            base = re.sub(r"_\d+$", "", stem)
            m = meta.get(base, {})
            images.append({
                "image_id": stem,
                "file": info.filename,
                "diagnosis": (m.get("Diagnosis") or cls_dir).strip(),
                "modality": "skin",
                "split": "augmented" if split_dir.lower().startswith("aug") else "preprocessed",
                "patient_id": (m.get("Patient_id") or "").strip(),
                "age": (m.get("Age") or "").strip(),
                "sex": (m.get("Sex") or "").strip(),
                "lesion_location": (m.get("Leision_location") or "").strip(),
                "bytes": info.file_size,
                "source": "skindisnet_images",
                "license": entry.get("license", "cc-by-nc-4.0"),
                "sha256_of_archive": sha,
            })
    return images


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def merge_diseases(hf: list[dict], gh: dict, kb_diseases: dict) -> list[dict]:
    """One row per condition, folding the two text sources onto the KB.

    The KB is the spine: its 67 conditions keep their curated severity and
    duration. HF and the GitHub set add conditions the KB does not cover, plus
    treatments, descriptions and precautions where the KB has none.
    """
    merged: "OrderedDict[str, dict]" = OrderedDict()

    def blank(name: str) -> dict:
        return {
            "disease_id": "", "name": name, "category": "", "icd10": "",
            "prevalence": "", "contagious": "", "severity_profile": "",
            "severity_score": "", "duration_min_days": "", "duration_max_days": "",
            "description": "", "key_symptoms": "", "symptoms": "", "self_care": "",
            "medicines": "", "routine": "", "when_to_see_doctor": "", "treatments": "",
            "precautions": "", "in_kb": "no", "sources": [],
        }

    # 1. KB spine
    for did, d in kb_diseases.items():
        row = blank(d.name)
        row.update({
            "disease_id": did,
            "name": d.name,
            "category": d.category or "",
            "icd10": d.icd10 or "",
            "prevalence": d.prevalence,
            "contagious": "yes" if d.contagious else "no",
            "severity_profile": d.severity_profile or "",
            "severity_score": d.severity_score,
            "duration_min_days": d.typical_duration_days[0] if d.typical_duration_days else "",
            "duration_max_days": d.typical_duration_days[1] if len(d.typical_duration_days) > 1 else "",
            "description": d.description or "",
            "key_symptoms": "; ".join(d.key_symptoms or []),
            "symptoms": "; ".join(sorted(d.symptoms or {})),
            "self_care": "; ".join(d.self_care or []),
            "medicines": "; ".join(d.medicines or []),
            "routine": d.routine or "",
            "when_to_see_doctor": "; ".join(d.when_to_see_doctor or []),
            "in_kb": "yes",
            "sources": ["knowledge_base"],
        })
        merged[_norm(d.name)] = row
        merged[_norm(did)] = row  # allow match on the id too

    # 2. GitHub set: 41 conditions, all of which the KB covers by name.
    for entry in gh["rows"]:
        key = _norm(entry["name"])
        row = merged.get(key) or merged.get(_norm(entry["name"]))
        if row is None:
            row = blank(entry["name"])
            merged[key] = row
        if not row["description"]:
            row["description"] = gh["descriptions"].get(entry["name"], "")
        prec = gh["precautions"].get(entry["name"])
        if prec and not row["precautions"]:
            row["precautions"] = "; ".join(prec)
        if "github_disease_symptom" not in row["sources"]:
            row["sources"].append("github_disease_symptom")

    # 3. HuggingFace set: 400 conditions, most of which the KB does not have.
    hf_by_name = {_norm(e["name"]): e for e in hf}
    # Resolve KB conditions to their HF vocabulary name before merging, so a
    # curated condition picks up HF's treatments even when the names differ.
    kb_to_hf: dict[str, str] = {}
    for did in kb_diseases:
        if did in HF_ALIASES:
            kb_to_hf[did] = HF_ALIASES[did]
    for entry in hf:
        key = _norm(entry["name"])
        row = merged.get(key)
        # Also try to find a KB row whose alias points at this HF name.
        if row is None:
            for did, hfname in kb_to_hf.items():
                if _norm(hfname) == key and did in kb_diseases:
                    row = merged.get(_norm(kb_diseases[did].name))
                    break
        if row is None:
            row = blank(entry["name"])
            merged[key] = row
        if entry["treatments"] and not row["treatments"]:
            row["treatments"] = "; ".join(entry["treatments"])
        if entry["symptoms"] and not row["symptoms"]:
            row["symptoms"] = "; ".join(entry["symptoms"])
        if "hf_diseases_symptoms" not in row["sources"]:
            row["sources"].append("hf_diseases_symptoms")

    # Drop the id aliases used only for matching.
    out, seen = [], set()
    for row in merged.values():
        if id(row) in seen:
            continue
        seen.add(id(row))
        row["sources"] = ", ".join(row["sources"])
        out.append(row)
    out.sort(key=lambda r: (r["in_kb"] != "yes", r["name"].lower()))
    return out


def merge_symptoms(gh: dict, kb_symptoms: dict) -> list[dict]:
    """One row per symptom: KB symptoms plus the GitHub severity weights."""
    rows: "OrderedDict[str, dict]" = OrderedDict()

    for sid, s in kb_symptoms.items():
        rows[_norm(sid)] = {
            "symptom_id": sid,
            "label": s.label or sid,
            "system": s.system or "",
            "severity_weight": gh["severity"].get(s.label, gh["severity"].get(sid, "")),
            "red_flag": "yes" if s.red_flag else "no",
            "red_flag_level": s.red_flag_level or "",
            "weight": s.weight,
            "synonyms": "; ".join(s.synonyms or []),
            "in_kb": "yes",
            "sources": "knowledge_base",
        }

    for raw, weight in gh["severity"].items():
        key = _norm(raw)
        if key in rows:
            continue
        rows[key] = {
            "symptom_id": raw.replace(" ", "_"),
            "label": raw,
            "system": "",
            "severity_weight": weight,
            "red_flag": "",
            "red_flag_level": "",
            "weight": "",
            "synonyms": "",
            "in_kb": "no",
            "sources": "github_disease_symptom",
        }

    out = list(rows.values())
    out.sort(key=lambda r: (r["in_kb"] != "yes", r["label"].lower()))
    return out


def build_disease_symptom_rows(diseases: list[dict], gh: dict) -> list[dict]:
    """Long form (disease, symptom, weight) for the conditions we can resolve."""
    by_name = {_norm(d["name"]): d for d in diseases}
    out = []
    for entry in gh["rows"]:
        d = by_name.get(_norm(entry["name"]))
        for sym in entry["symptoms"]:
            out.append({
                "disease": entry["name"],
                "disease_id": (d or {}).get("disease_id", ""),
                "symptom": sym,
                "symptom_id": sym.replace(" ", "_"),
                "weight": gh["severity"].get(sym, ""),
                "source": "github_disease_symptom",
            })
    return out


# --------------------------------------------------------------------------- #
# workbook writing
# --------------------------------------------------------------------------- #
def _write_sheet(ws, columns: list[str], rows: list[dict]) -> None:
    ws.append(columns)
    for row in rows:
        ws.append([row.get(c, "") for c in columns])
    ws.freeze_panes = "A2"


def build_workbook(out: Path) -> dict:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    if not MANIFEST.exists():
        print("sources.json missing - run scripts/fetch_sources.py first", file=sys.stderr)
        return {"ok": False, "error": "no manifest"}

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    from app.kb import KB

    hf = read_hf(SOURCES_DIR / "hf_diseases_symptoms.csv") if (SOURCES_DIR / "hf_diseases_symptoms.csv").exists() else []
    gh = read_github(SOURCES_DIR / "github")
    diseases = merge_diseases(hf, gh, KB.diseases)
    symptoms = merge_symptoms(gh, KB.symptoms)
    ds_rows = build_disease_symptom_rows(diseases, gh)

    precautions = [
        {"disease": k, "precaution_1": v[0] if len(v) > 0 else "",
         "precaution_2": v[1] if len(v) > 1 else "",
         "precaution_3": v[2] if len(v) > 2 else "",
         "precaution_4": v[3] if len(v) > 3 else "",
         "source": "github_disease_symptom"}
        for k, v in sorted(gh["precautions"].items())
    ]

    images = read_eye_images(manifest) + read_skin_images(manifest)
    deeplesion = read_deeplesion_metadata(manifest)
    catalogue = read_pmc_catalogue(manifest)
    mn = read_medicinenet_index()
    mn_entry = manifest["sources"].get("medicinenet_index", {})
    for r in mn:
        r["source"] = "medicinenet_index"
        r["license"] = mn_entry.get("license", "copyright-webmd")
        r["note"] = "Index entry only; article text and images are not copied."

    source_rows = []
    for sid, e in manifest["sources"].items():
        sha = e.get("sha256")
        if isinstance(sha, dict):
            sha = "; ".join(f"{k}:{v[:12]}" for k, v in sha.items())
        source_rows.append({
            "id": sid, "title": e.get("title", ""), "url": e.get("url", ""),
            "license": e.get("license", ""), "doi": e.get("doi", ""),
            "status": e.get("status", ""), "bytes": e.get("bytes", ""),
            "sha256": sha or "", "fetched_at": e.get("fetched_at", ""),
            "note": e.get("note", ""),
        })
    source_rows.sort(key=lambda r: r["id"])

    wb = Workbook()
    wb.remove(wb.active)

    head_fill = PatternFill("solid", fgColor="1F4E5F")
    head_font = Font(bold=True, color="FFFFFF")
    sheets = [
        ("Diseases", DISEASE_COLUMNS, diseases),
        ("Symptoms", SYMPTOM_COLUMNS, symptoms),
        ("Disease_Symptoms", DISEASE_SYMPTOM_COLUMNS, ds_rows),
        ("Precautions", PRECAUTION_COLUMNS, precautions),
        ("Images", IMAGE_COLUMNS, images),
        ("DeepLesion", DEEPLESION_COLUMNS, deeplesion),
        ("Dataset_Catalogue", DATASET_CATALOGUE_COLUMNS, catalogue),
        ("MedicineNet", MEDICINENET_COLUMNS, mn),
        ("Sources", SOURCE_COLUMNS, source_rows),
    ]
    for name, cols, rows in sheets:
        ws = wb.create_sheet(name)
        _write_sheet(ws, cols, rows)
        for i, col in enumerate(cols, start=1):
            ws.cell(row=1, column=i).fill = head_fill
            ws.cell(row=1, column=i).font = head_font
            ws.column_dimensions[get_column_letter(i)].width = min(38, max(12, len(col) + 4))

    readme = wb.create_sheet("Read_Me", 0)
    readme_rows = [
        ("Dr. Aishu consolidated dataset", ""),
        ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("", ""),
        ("Sheet", "What it holds"),
        ("Diseases", f"{len(diseases)} conditions; the 67 knowledge-base conditions plus "
                     f"the rest from the HuggingFace and GitHub sets."),
        ("Symptoms", f"{len(symptoms)} symptoms with severity weight and red-flag state."),
        ("Disease_Symptoms", f"{len(ds_rows)} (disease, symptom) pairs - the training shape."),
        ("Precautions", f"{len(precautions)} conditions with up to four precaution steps."),
        ("Images", f"{len(images)} images: skin (SkinDisNet) and eye (Mendeley), with class labels."),
        ("DeepLesion", f"{len(deeplesion)} lesion annotations (NIH DeepLesion): bounding box, "
                       "RECIST size, coarse lesion type and patient demographics. Radiology "
                       "ground truth, kept separate from symptom data."),
        ("Dataset_Catalogue", f"{len(catalogue)} public disease datasets named in the CC BY "
                              "review PMC8950225 (Table 18), with links."),
        ("MedicineNet", f"{len(mn)} condition index entries (name, slug, link)."),
        ("Sources", "Provenance: url, licence, status, byte size and SHA-256 per source."),
        ("", ""),
        ("Licences - read before reusing", ""),
        ("unknown", "HuggingFace CSV, GitHub repo and Mendeley eye set declare no licence. "
                    "Treat as all rights reserved until the uploader states otherwise. The "
                    "Mendeley 2023 disease/symptom CSV also declares no licence."),
        ("cc-by-nc-4.0", "SkinDisNet: non-commercial use only, attribution required."),
        ("cc-by-4.0", "The PMC ML-disease-diagnosis review (PMC8950225): attribution required. "
                      "Only its factual dataset catalogue is stored, not the article prose."),
        ("nih-research-use", "DeepLesion (NIH Clinical Center): research use. The readme states "
                             "no redistribution licence, and only the annotation CSV and docs "
                             "are included - the ~238 GB image archive is not downloaded."),
        ("public-domain-nlm", "PMC-ids.csv.gz is an NLM identifier table (PMCID/PMID/DOI/ISSN). "
                              "No article text is included."),
        ("copyright-google", "Google Research Health AI is a landing page with no dataset and no "
                             "licence; recorded as a reference URL only, no content copied."),
        ("copyright-webmd", "MedicineNet is copyright WebMD. Its robots.txt disallows the AI "
                            "crawlers, so only the public condition index is stored - no article "
                            "text or image bytes are included. Visit the URL to read an article."),
        ("", ""),
        ("Matching", "disease_id joins to the app's knowledge base; in_kb = yes means the "
                     "condition is one Dr. Aishu can already reason about."),
    ]
    for row in readme_rows:
        readme.append(list(row))
    readme.column_dimensions["A"].width = 30
    readme.column_dimensions["B"].width = 96
    for r in readme.iter_rows(min_row=1, max_row=1):
        for c in r:
            c.font = Font(bold=True, size=13)
    for r in readme.iter_rows(min_row=4, max_row=4):
        for c in r:
            c.fill = head_fill
            c.font = head_font
    for r in readme.iter_rows(min_row=14, max_row=14):
        for c in r:
            c.fill = head_fill
            c.font = head_font
    for row in readme.iter_rows(min_row=1, max_row=readme.max_row):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)

    return {
        "ok": True, "out": str(out.relative_to(ROOT)),
        "sheets": {name: len(rows) for name, _, rows in sheets},
        "bytes": out.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    result = build_workbook(args.out)
    if not result.get("ok"):
        return 1
    print(f"wrote {result['out']} ({result['bytes']:,} bytes)")
    for name, n in result["sheets"].items():
        print(f"  {name:20} {n:>6} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
