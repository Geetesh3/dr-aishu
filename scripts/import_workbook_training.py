"""Turn the consolidated workbook into narrative training examples.

The classifier learns from how people *write*, not from symptom lists, so the
workbook's rows cannot be fed in raw. This script reads the 67 conditions the
workbook marks `in_kb`, takes each one's symptom set and duration range, and
phrases them the way a patient would, reusing the openers and connectors from
`scripts/seed_training_data.py` so the vocabulary matches the rest of training.

Why bother when the seed set already covers these conditions: the seed set draws
its symptoms from the knowledge base's own weights, while the workbook also
carries the imported co-occurrence and severity signal. The two are correlated
but not identical, and a model that has seen both phrasings is more robust to a
patient who lists a less typical symptom.

The output is written to `data/training/workbook_2024.jsonl`, which
`scripts/train_model.py` picks up only under `--use-imports`. That flag is off
by default because imports have been measured to *lower* held-out accuracy (see
the note in `collect()`); this script exists so the measurement can be repeated
against the current data rather than assumed.

Run:
    python scripts/import_workbook_training.py --per-disease 30
    python scripts/train_model.py --use-imports      # then compare the report
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import dataset  # noqa: E402
from app.kb import KB  # noqa: E402
from scripts.seed_training_data import (  # noqa: E402
    CONNECTORS, CONTEXT_PHRASES, OPENERS, SEVERITY_PHRASES, _duration_phrase,
)

OUT = ROOT / "data" / "training" / "workbook_2024.jsonl"


def _phrase(symptom_id: str, rng: random.Random) -> str:
    s = KB.symptoms.get(symptom_id)
    if not s:
        return symptom_id.replace("_", " ")
    # Synonyms now carry translations in Indic scripts. Training examples are
    # English, so only Latin-script forms are used; mixing scripts would teach
    # the classifier that a Devanagari word predicts a condition.
    def latin(text: str) -> bool:
        return bool(text) and all(ord(ch) < 0x250 for ch in text)

    options = [s.label.lower()] + [
        x for x in s.synonyms if len(x) > 3 and latin(x)
    ]
    return rng.choice(options)


def _symptom_ids(row: dict) -> list[str]:
    """Symptom ids from the workbook, split into key and supporting."""
    def split(field: str) -> list[str]:
        raw = row.get(field) or ""
        return [s.strip() for s in str(raw).split(";") if s.strip()]

    key = [s for s in split("key_symptoms") if s in KB.symptoms]
    rest = [s for s in split("symptoms") if s in KB.symptoms and s not in key]
    return key, rest


def build(row: dict, rng: random.Random) -> tuple[str, list[str]] | None:
    key, rest = _symptom_ids(row)
    if not key and not rest:
        return None

    # Always mention a key symptom; add supporting ones with some probability,
    # mirroring how a person leads with the thing that bothers them most.
    picked = list(key)
    for sid in rest:
        if rng.random() < 0.6:
            picked.append(sid)
    if not picked:
        return None
    picked = picked[:6]
    rng.shuffle(picked)

    phrases = [_phrase(sid, rng) for sid in picked]
    parts = [rng.choice(OPENERS)]
    if len(phrases) == 1:
        parts.append(phrases[0])
    else:
        body = phrases[0]
        for p in phrases[1:]:
            body += f" {rng.choice(CONNECTORS)} {p}"
        parts.append(body)

    lo = row.get("duration_min_days")
    hi = row.get("duration_max_days")
    try:
        lo = max(1, int(float(lo))) if lo not in (None, "") else 1
        hi = max(lo, min(int(float(hi)), 60)) if hi not in (None, "") else lo
    except (TypeError, ValueError):
        lo = hi = 3
    if rng.random() < 0.75:
        parts.append(_duration_phrase(rng.randint(lo, hi), rng))

    if rng.random() < 0.30:
        parts.append(f"the symptoms are {rng.choice(SEVERITY_PHRASES)}")
    if rng.random() < 0.25:
        parts.append(rng.choice(CONTEXT_PHRASES).format(age=rng.choice([8, 16, 22, 30, 41, 52, 63, 71, 78])))

    text = ", ".join(parts[:1]) + " " + " ".join(parts[1:])
    text = text.replace(" ,", ",").replace(",,", ",").strip().rstrip(".") + "."
    return text, picked


def generate(per_disease: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    skipped: list[str] = []
    for row in dataset.diseases(in_kb_only=True):
        label = row.get("disease_id")
        if not label or label not in KB.diseases:
            continue
        seen: set[str] = set()
        made = 0
        guard = 0
        while made < per_disease and guard < per_disease * 25:
            guard += 1
            built = build(row, rng)
            if not built:
                break
            text, symptoms = built
            if text[:120] in seen:
                continue
            seen.add(text[:120])
            rows.append({
                "text": text,
                "symptoms": symptoms,
                "label": label,
                "weight": 1.0,
                "origin": "workbook",
            })
            made += 1
        if made == 0:
            skipped.append(label)
    rng.shuffle(rows)
    if skipped:
        print(f"no examples for {len(skipped)} conditions: {skipped[:8]}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-disease", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not dataset.available():
        print("no workbook - run scripts/build_dataset_workbook.py first", file=sys.stderr)
        return 2

    rows = generate(args.per_disease, args.seed)
    if not rows:
        print("nothing generated", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels: dict[str, int] = {}
    for r in rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    print(f"wrote {len(rows)} examples across {len(labels)} labels -> {args.out}")
    print("examples per label:", min(labels.values()), "-", max(labels.values()))
    for r in rows[:3]:
        print(f"  [{r['label']}] {r['text'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
