"""Generate labelled training data for Dr. Aishu from the medical knowledge base.

This is the bootstrap step. Before the app has seen any real cases, the 42
conditions in the knowledge base are turned into thousands of varied symptom
descriptions, so the classifier has something to learn from on day one. Once real
consented cases accumulate, they are mixed in and eventually dominate.

Run:  python scripts/seed_training_data.py --per-disease 40
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.kb import KB  # noqa: E402

OUT_DIR = ROOT / "data" / "training"
OUT_FILE = OUT_DIR / "seed.jsonl"

OPENERS = [
    "I have been feeling unwell with",
    "For the past few days I have had",
    "I am suffering from",
    "My main problem is",
    "Doctor, I have",
    "Recently I noticed",
    "I have been experiencing",
    "Symptoms I have:",
    "Over the last week I have had",
    "I would like help with",
]

CONNECTORS = ["and", "along with", "plus", "together with", "as well as", ","]


def _duration_phrase(n_days: int, rng: random.Random) -> str:
    """Pick a natural phrasing, converting to weeks only when it reads well."""
    if n_days >= 7 and rng.random() < 0.5:
        weeks = max(1, round(n_days / 7))
        templates = [
            f"for {weeks} week" + ("s" if weeks > 1 else ""),
            f"since {weeks} week" + ("s" if weeks > 1 else ""),
            f"for about {weeks} week" + ("s" if weeks > 1 else ""),
            f"for the last {weeks} week" + ("s" if weeks > 1 else ""),
            f"over the past {weeks} week" + ("s" if weeks > 1 else ""),
        ]
        return rng.choice(templates)
    days = max(1, n_days)
    templates = [
        f"for {days} day" + ("s" if days > 1 else ""),
        f"for about {days} day" + ("s" if days > 1 else ""),
        f"for the last {days} day" + ("s" if days > 1 else ""),
        f"over the past {days} day" + ("s" if days > 1 else ""),
        f"{days} day" + ("s" if days > 1 else "") + " now",
        f"started {days} day" + ("s" if days > 1 else "") + " ago",
    ]
    return rng.choice(templates)

SEVERITY_PHRASES = ["mild", "moderate", "severe", "quite bad", "very uncomfortable"]

CONTEXT_PHRASES = [
    "I am {age} years old", "age {age}", "I am a {age} year old",
]

# Deliberately confusing but clinically distinct co-mentions, to teach boundaries.
NEIGHBOUR_NOISE = {
    "common_cold": ["influenza", "allergic_rhinitis", "sinusitis"],
    "influenza": ["common_cold", "covid19"],
    "covid19": ["influenza", "common_cold"],
    "migraine": ["tension_headache", "sinusitis"],
    "tension_headache": ["migraine"],
    "gastroenteritis": ["constipation", "gerd"],
    "gerd": ["peptic_ulcer", "gastroenteritis"],
    "uti": ["kidney_stones"],
    "kidney_stones": ["uti"],
    "arthritis_oa": ["rheumatoid_arthritis", "gout"],
    "gout": ["rheumatoid_arthritis", "cellulitis"],
    "eczema": ["fungal_skin", "urticaria"],
    "fungal_skin": ["eczema"],
    "urticaria": ["eczema", "cellulitis"],
    "dengue": ["influenza", "typhoid", "malaria"],
    "malaria": ["dengue", "typhoid", "influenza"],
    "typhoid": ["malaria", "gastroenteritis"],
    "anxiety_disorder": ["depression"],
    "depression": ["anxiety_disorder", "fibromyalgia"],
    "fibromyalgia": ["depression", "arthritis_oa"],
    "type2_diabetes": ["hyperthyroidism", "hypothyroidism"],
    "hyperthyroidism": ["type2_diabetes", "anxiety_disorder"],
    "hypothyroidism": ["depression", "anemia"],
    "anemia": ["hypothyroidism"],
    "hypertension": ["anxiety_disorder", "angina"],
    "angina": ["hypertension", "gerd"],
    "asthma": ["acute_bronchitis", "pneumonia"],
    "pneumonia": ["acute_bronchitis", "influenza"],
    "tonsillitis": ["common_cold", "influenza"],
    "conjunctivitis": ["allergic_rhinitis"],
    "otitis_media": ["tonsillitis", "sinusitis"],
}


def phrase_for(symptom_id: str, rng: random.Random) -> str:
    """Pick a natural surface form for a symptom."""
    s = KB.symptoms.get(symptom_id)
    if not s:
        return symptom_id.replace("_", " ")
    options = [s.label.lower()] + [x for x in s.synonyms if len(x) > 3]
    choice = rng.choice(options)
    return choice


def build_description(disease_id: str, rng: random.Random) -> tuple[str, list[str]]:
    disease = KB.diseases[disease_id]
    # Prefer the disease's own symptom weights: heavier symptoms are likelier to be mentioned.
    weighted = [(sid, w) for sid, w in disease.symptoms.items()]
    if not weighted:
        return "", []

    pool = [sid for sid, _ in weighted]
    weights = [w for _, w in weighted]
    k = rng.randint(min(2, len(pool)), min(len(pool), 6))
    picked: list[str] = []
    attempts = 0
    while len(picked) < k and attempts < 40:
        candidate = rng.choices(pool, weights=weights, k=1)[0]
        if candidate not in picked:
            picked.append(candidate)
        attempts += 1

    # Occasionally include a symptom from a neighbouring condition to create
    # realistic overlap that the model must learn to separate.
    if rng.random() < 0.18:
        neighbour = rng.choice(NEIGHBOUR_NOISE.get(disease_id, []) or [disease_id])
        if neighbour in KB.diseases:
            extra = list(KB.diseases[neighbour].symptoms.keys())
            if extra:
                noise_sid = rng.choice(extra)
                if noise_sid not in picked:
                    picked.append(noise_sid)

    phrases = [phrase_for(sid, rng) for sid in picked]
    rng.shuffle(phrases)

    parts = [rng.choice(OPENERS)]
    if len(phrases) == 1:
        parts.append(phrases[0])
    else:
        body = phrases[0]
        for p in phrases[1:]:
            body += f" {rng.choice(CONNECTORS)} {p}"
        parts.append(body)

    # Duration
    lo, hi = disease.typical_duration_days
    lo = max(1, int(lo)) if lo >= 1 else 1
    hi = max(lo, min(int(hi), 60))
    n = rng.randint(lo, hi)
    if rng.random() < 0.75:
        parts.append(_duration_phrase(n, rng))

    # Severity and context
    if rng.random() < 0.35:
        parts.append(f"the symptoms are {rng.choice(SEVERITY_PHRASES)}")
    if rng.random() < 0.30:
        parts.append(rng.choice(CONTEXT_PHRASES).format(age=rng.choice([8, 16, 22, 30, 41, 52, 63, 71, 78])))

    text = ", ".join(parts[:1]) + " " + " ".join(parts[1:])
    text = text.replace(" ,", ",").replace(",,", ",")
    return text.strip().rstrip(".") + ".", picked


def generate(per_disease: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for disease_id in KB.diseases:
        seen: set[str] = set()
        made = 0
        guard = 0
        while made < per_disease and guard < per_disease * 25:
            guard += 1
            text, symptoms = build_description(disease_id, rng)
            if not text or not symptoms:
                continue
            key = text[:120]
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "text": text,
                "symptoms": symptoms,
                "label": disease_id,
                "weight": 1.0,
                "origin": "seed",
            })
            made += 1
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed Dr. Aishu training data from the knowledge base.")
    ap.add_argument("--per-disease", type=int, default=40,
                    help="synthetic examples per condition (default 40)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()

    rows = generate(args.per_disease, args.seed)
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


if __name__ == "__main__":
    main()
