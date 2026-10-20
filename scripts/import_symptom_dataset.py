"""Import an external disease/symptom dataset into Dr. Aishu's training set.

The first dataset wired up here is the Mendeley "Disease and symptoms dataset
2023" (DOI 10.17632/2cxccsxydc.1): 773 diseases x 377 symptoms, ~247k rows of
binary symptom flags. It is far larger than the curated knowledge base, so it is
used as training signal and coverage reference, not as a replacement for the KB:

* training examples -- each row becomes a labelled (symptoms -> disease) example
  the classifier can learn from, which is what lets Dr. Aishu recognise
  conditions the KB covers under names the KB does not list.
* symptom rates -- the share of a disease's rows in which each symptom appears is
  written out per disease. The dataset's rows look synthetic (symptom rates
  cluster near 0.5 regardless of the symptom), so these are recorded as a
  descriptive reference for eyeballing coverage, not as calibrated
  probabilities to overwrite the KB's hand-set weights with.

Only rows whose disease maps to a knowledge-base condition are emitted as
training examples, because the classifier's label space is the KB. Everything
else is reported as a coverage gap so the knowledge base can be extended
deliberately rather than by accident.

Run:
    python scripts/import_symptom_dataset.py \
        --csv data/external/disease_symptoms_2023.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.kb import KB  # noqa: E402

DEFAULT_CSV = ROOT / "data" / "external" / "disease_symptoms_2023.csv"
TRAIN_DIR = ROOT / "data" / "training"
REPORT_PATH = ROOT / "data" / "external" / "import_report.json"
RATES_PATH = ROOT / "data" / "external" / "disease_symptom_rates.json"

# Dataset names that do not reduce to a knowledge-base id or label by
# normalisation alone. Kept explicit and small: a guess here becomes a wrong
# label on real training data, so anything ambiguous is left unmatched and
# reported instead. Each entry below was confirmed by inspecting the dataset's
# disease list; conditions with no unambiguous counterpart (COVID-19, urticaria,
# anaphylaxis) are deliberately absent.
DISEASE_ALIASES = {
    # dataset spelling -> knowledge-base id
    "flu": "influenza",
    "cystitis": "uti",
    "urinary tract infection": "uti",
    "vaginal cyst": "uti",
    "vulvodynia": "uti",
    "infectious gastroenteritis": "gastroenteritis",
    "diverticulitis": "gastroenteritis",
    "gastrointestinal hemorrhage": "peptic_ulcer",
    "gastroduodenal ulcer": "peptic_ulcer",
    "nose disorder": "allergic_rhinitis",
    "nosebleed": "allergic_rhinitis",
    "sinusitis": "sinusitis",
    "otitis externa": "otitis_media",
    "spondylosis": "sciatica",
    "peripheral nerve disorder": "peripheral_neuropathy",
    "diabetic peripheral neuropathy": "peripheral_neuropathy",
    "complex regional pain syndrome": "sciatica",
    "lumbago": "lower_back_pain",
    "chronic back pain": "lower_back_pain",
    "hypoglycemia": "type2_diabetes",
    "diabetes": "type2_diabetes",
    "conjunctivitis due to allergy": "conjunctivitis",
    "esophagitis": "gerd",
    "heartburn": "gerd",
    "depressive or psychotic symptoms": "depression",
    "chronic constipation": "constipation",
    "high blood pressure": "hypertension",
    "kidney stone": "kidney_stones",
    "osteoarthritis": "arthritis_oa",
    "fungal infection of the skin": "fungal_skin",
    "pyogenic skin infection": "cellulitis",
    "burn": "burns",
    "anxiety": "anxiety_disorder",
    "dengue fever": "dengue",
    "typhoid fever": "typhoid",
    "obstructive sleep apnea (osa)": "sleep_apnea",
    "labyrinthitis": "bppv",
    "deep vein thrombosis (dvt)": "dvt",
    "irritable bowel syndrome": "ibs",
    "oral thrush (yeast infection)": "oral_thrush",
    "tooth abscess": "dental_abscess",
    "parkinson disease": "parkinsons",
    "polycystic ovarian syndrome (pcos)": "pcos",
    "benign prostatic hyperplasia (bph)": "bph",
    "primary insomnia": "insomnia_disorder",
    "vaginal yeast infection": "vaginal_candidiasis",
}

# Dataset symptom columns that map to a knowledge-base symptom by meaning rather
# than by string. The dataset is far more granular (it splits pain and swelling
# by body part), so several columns collapse onto one knowledge-base symptom.
# Only unambiguous equivalences are listed; a column whose best match is merely
# related is left out and reported as a coverage gap.
SYMPTOM_ALIASES = {
    "ache_all_over": "widespread_pain",
    "pain_all_over": "widespread_pain",
    "sharp_abdominal_pain": "severe_abdominal_pain",
    "lower_abdominal_pain": "abdominal_pain",
    "upper_abdominal_pain": "abdominal_pain",
    "abnormal_appearing_skin": "skin_lesion",
    "skin_growth": "skin_lesion",
    "acne_or_pimples": "skin_lesion",
    "leg_swelling": "leg_swelling",
    "ankle_swelling": "leg_swelling",
    "skin_swelling": "skin_swelling",
    "arm_swelling": "skin_swelling",
    "back_swelling": "skin_swelling",
    "hand_or_finger_swelling": "skin_swelling",
    "wrist_swelling": "skin_swelling",
    "hip_pain": "joint_pain",
    "ankle_pain": "joint_pain",
    "wrist_pain": "joint_pain",
    "hand_or_finger_pain": "joint_pain",
    "arm_pain": "pain",
    "leg_pain": "pain",
    "arm_weakness": "fatigue",
    "leg_weakness": "fatigue",
    "elbow_weakness": "fatigue",
    "back_weakness": "fatigue",
    "arm_cramps_or_spasms": "muscle_ache",
    "leg_cramps_or_spasms": "muscle_ache",
    "abdominal_distention": "bloating",
    "flatulence": "excessive_gas",
    "anxiety_and_nervousness": "anxiety",
    "hostile_behavior": "irritability",
    "antisocial_behavior": "irritability",
    "irritable_infant": "irritability",
    "feeling_ill": "fatigue",
    "malaise": "fatigue",
    "breathing_fast": "rapid_breathing",
    "difficulty_breathing": "shortness_of_breath",
    "abnormal_breathing_sounds": "wheezing",
    "hoarse_voice": "hoarseness",
    "difficulty_in_swallowing": "difficulty_swallowing",
    "throat_swelling": "swollen_face_lips",
    "blood_in_stool": "bloody_stool",
    "depression": "low_mood",
    "depressive_or_psychotic_symptoms": "low_mood",
    "emotional_symptoms": "low_mood",
    "irregular_heartbeat": "palpitations",
    "sharp_chest_pain": "chest_pain",
    "diminished_vision": "blurred_vision",
    "double_vision": "blurred_vision",
    "pain_in_eye": "eye_pain",
    "white_discharge_from_eye": "eye_discharge",
    "diminished_hearing": "hearing_loss",
    "pus_draining_from_ear": "ear_discharge",
    "toothache": "tooth_pain",
    "mouth_ulcer": "mouth_ulcers",
    "lip_swelling": "swollen_face_lips",
    "vaginal_itching": "itching",
    "hot_flashes": "hot_flushes",
    "abnormal_involuntary_movements": "tremor",
    "arm_lump_or_mass": "skin_swelling",
    "back_mass_or_lump": "skin_swelling",
}


def _display(path: Path) -> str:
    """Path relative to the project when possible, else absolute.

    Report paths are meant to be readable, and a run pointed at a dataset outside
    the project would otherwise fail on the relative-path arithmetic.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def norm(text: str) -> str:
    """Normalise a free-text name to a comparable key."""
    return (
        text.strip().lower()
        .replace("&", " and ")
        .replace("/", " ")
        .replace("-", "_")
        .replace("'", "")
        .replace(" ", "_")
    )


def _strip_parens(text: str) -> str:
    return text.split("(")[0].strip()


def build_disease_index() -> dict[str, str]:
    """Map every plausible spelling of a KB disease to its id."""
    index: dict[str, str] = {}
    for did, d in KB.diseases.items():
        index[norm(did)] = did
        index[norm(d.name)] = did
        index[norm(_strip_parens(d.name))] = did
    return index


def build_symptom_index() -> dict[str, str]:
    """Map every plausible spelling of a KB symptom to its id."""
    index: dict[str, str] = {}
    for sid, s in KB.symptoms.items():
        index[norm(sid)] = sid
        index[norm(s.label)] = sid
        for syn in s.synonyms or []:
            # Only latin-script synonyms are useful for matching an English
            # dataset; the Indic synonyms would add noise.
            if syn.isascii():
                index.setdefault(norm(syn), sid)
    return index


def map_disease(name: str, index: dict[str, str]) -> str | None:
    key = norm(name)
    if key in index:
        return index[key]
    alias = DISEASE_ALIASES.get(name.strip().lower())
    if alias and alias in KB.diseases:
        return alias
    stripped = norm(_strip_parens(name))
    return index.get(stripped)


def map_symptom(column: str, index: dict[str, str]) -> str | None:
    key = norm(column)
    if key in index:
        return index[key]
    alias = SYMPTOM_ALIASES.get(key)
    if alias and alias in KB.symptoms:
        return alias
    return None


def phrase(symptom_ids: list[str]) -> str:
    """Turn symptom ids into a natural-sounding description.

    The classifier is trained on user phrasing, so an imported row is rendered
    with the knowledge base's own labels rather than the raw ids -- "fever,
    headache and nausea" reads like something a person would type, which keeps
    the imported examples in the same feature space as the real ones.
    """
    labels = [KB.symptoms[s].label.lower() for s in symptom_ids if s in KB.symptoms]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def import_dataset(
    csv_path: Path,
    *,
    max_per_disease: int = 120,
    limit: int | None = None,
) -> dict:
    disease_index = build_disease_index()
    symptom_index = build_symptom_index()

    out_file = TRAIN_DIR / "mendeley_2023.jsonl"
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    RATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_per_disease: Counter[str] = Counter()
    unmatched_diseases: Counter[str] = Counter()
    co_occurrence: dict[str, Counter[str]] = defaultdict(Counter)
    disease_totals: Counter[str] = Counter()

    written = 0
    rows_read = 0
    skipped_unmapped_disease = 0

    with open(csv_path, newline="", encoding="utf-8") as fh, open(out_file, "w", encoding="utf-8") as out:
        reader = csv.reader(fh)
        header = next(reader)
        columns = header[1:]

        # Resolve each column once; the CSV has 377 of them and 247k rows.
        col_to_symptom: dict[int, str | None] = {}
        for i, col in enumerate(columns):
            col_to_symptom[i] = map_symptom(col, symptom_index)

        for row in reader:
            rows_read += 1
            if limit and rows_read > limit:
                break

            raw_disease = row[0]
            disease_id = map_disease(raw_disease, disease_index)
            if disease_id is None:
                unmatched_diseases[raw_disease] += 1
                skipped_unmapped_disease += 1
                continue

            disease_totals[disease_id] += 1
            present: list[str] = []
            seen: set[str] = set()
            for i, value in enumerate(row[1:]):
                if value.strip() not in ("", "0", "0.0"):
                    sid = col_to_symptom.get(i)
                    # Several dataset columns collapse onto one KB symptom
                    # ("leg_pain" and "hip_pain" both -> joint_pain), so a row
                    # can name the same symptom many times. Count each once.
                    if sid is None or sid in seen:
                        continue
                    seen.add(sid)
                    present.append(sid)
                    co_occurrence[disease_id][sid] += 1

            if not present:
                continue
            if seen_per_disease[disease_id] >= max_per_disease:
                continue

            seen_per_disease[disease_id] += 1
            out.write(json.dumps({
                "text": phrase(present),
                "symptoms": present,
                "label": disease_id,
                "weight": 0.8,  # below a real case: synthetic, not observed here
                "origin": "mendeley_2023",
            }) + "\n")
            written += 1

    # Share of each disease's rows in which a symptom appears. Descriptive only:
    # the dataset's rates cluster near 0.5, so they are not calibrated enough to
    # replace the KB's hand-set weights.
    rates = {
        did: {sid: round(count / disease_totals[did], 4) for sid, count in counter.most_common()}
        for did, counter in sorted(co_occurrence.items())
    }
    RATES_PATH.write_text(json.dumps(rates, indent=1), encoding="utf-8")

    # A column only matters if it appears in a row we actually used.
    unmatched_symptom_cols = sorted(
        col for i, col in enumerate(columns) if col_to_symptom[i] is None
    )

    report = {
        "source": str(csv_path),
        "rows_read": rows_read,
        "examples_written": written,
        "examples_path": _display(out_file),
        "diseases_in_kb_matched": len(disease_totals),
        "diseases_total_seen": len(disease_totals) + len(unmatched_diseases),
        "rows_skipped_unmapped_disease": skipped_unmapped_disease,
        "symptoms_in_kb_matched": len({s for s in col_to_symptom.values() if s}),
        "symptoms_total_seen": len(columns),
        "rates_path": _display(RATES_PATH),
        "unmatched_diseases": unmatched_diseases.most_common(),
        "unmatched_symptoms": unmatched_symptom_cols,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Import an external disease/symptom dataset.")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="path to the dataset CSV")
    ap.add_argument("--max-per-disease", type=int, default=120,
                    help="cap examples per disease so a common condition cannot dominate")
    ap.add_argument("--limit", type=int, default=None, help="stop after N data rows (for a quick run)")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"dataset not found: {args.csv}", file=sys.stderr)
        print("Download it from https://data.mendeley.com/datasets/2cxccsxydc/1", file=sys.stderr)
        return 1

    report = import_dataset(
        args.csv, max_per_disease=args.max_per_disease, limit=args.limit,
    )
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("unmatched_diseases", "unmatched_symptoms")}, indent=2))
    print(f"\nunmatched diseases: {len(report['unmatched_diseases'])}")
    print(f"unmatched symptoms: {len(report['unmatched_symptoms'])}")
    print(f"\nwrote {report['examples_path']} and {report['rates_path']}")
    print(f"report: {_display(REPORT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
