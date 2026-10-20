"""Generate realistic demo searches so the insights dashboard has data.

A fresh install has an empty database, which makes every chart blank. This
module runs real symptom text through the real analysis pipeline and records the
results as demo events, so the charts, the training set and the diagnosis mix are
all genuinely produced by the app rather than being fabricated numbers.

Rows are written with ``is_demo = 1`` and are counted separately in the UI, and
``purge_demo()`` removes them without touching real traffic.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from . import pipeline, store

# Symptom presentations a walk-in clinic actually sees, weighted so common,
# mild complaints dominate and emergencies stay rare. Each entry is
# (weight, narrative, age, sex).
PRESENTATIONS: list[tuple[int, str, float, str]] = [
    (18, "runny nose, sneezing and a sore throat for three days", 28, "female"),
    (16, "high fever, body aches and chills with a dry cough", 34, "male"),
    (14, "burning when I pee and going very often with lower abdominal pain", 26, "female"),
    (12, "throbbing one-sided headache with nausea and light sensitivity", 31, "female"),
    (11, "itchy red rash on my forearm that has been spreading for a week", 24, "male"),
    (10, "stomach cramps, watery diarrhoea and feeling sick since yesterday", 22, "female"),
    (9, "sore throat with white patches and painful swallowing for two days", 19, "male"),
    (8, "dry itchy scaly patches on both elbows for a month", 41, "male"),
    (8, "feeling very tired and dizzy, look pale, for two weeks", 29, "female"),
    (7, "lower back pain after lifting a heavy box two days ago", 38, "male"),
    (7, "heartburn and burning in the chest after meals for a week", 45, "male"),
    (6, "knees ache and stiff in the morning, worse after walking", 56, "female"),
    (6, "frequent urination and increased thirst over the last month", 52, "male"),
    (6, "itchy watery eyes and sneezing every morning during pollen season", 27, "female"),
    (5, "constipated and bloated for four days, hard stools", 33, "female"),
    (5, "toothache with swelling of the gum for two days", 30, "male"),
    (5, "dry cough that has lasted three weeks, worse at night", 43, "female"),
    (4, "anxiety, racing thoughts and trouble sleeping for a month", 35, "female"),
    (4, "painful red eye with discharge since this morning", 21, "male"),
    (4, "ring-shaped itchy patch on my leg that keeps growing", 25, "male"),
    (3, "ear pain and muffled hearing after a cold for three days", 17, "female"),
    (3, "sudden severe headache with vomiting, worst headache of my life", 48, "female"),
    (2, "crushing chest pain spreading to my left arm with sweating", 58, "male"),
    (2, "high fever, stiff neck and a rash that appeared this morning", 23, "male"),
]

DEMO_SOURCES = ("web", "api", "colab")


def generate(count: int = 250, days: int = 30, seed: int | None = None) -> dict[str, Any]:
    """Run `count` demo presentations through the real pipeline and store them.

    Results are back-dated across `days` so the timeline chart shows a realistic
    spread rather than a single spike.
    """
    rng = random.Random(seed)
    weights = [w for w, *_ in PRESENTATIONS]
    created = 0
    failed = 0
    now = datetime.now(timezone.utc)

    for _ in range(count):
        _, text, age, sex = rng.choices(PRESENTATIONS, weights=weights, k=1)[0]
        try:
            result = pipeline.analyse(text=text, top_n=6, age=age, record=False)
        except Exception:
            failed += 1
            continue

        # Persist case + event in one call, then back-date both so the timeline
        # spreads across the window instead of spiking on today. The narratives
        # are scripted by us, so consent is trivially satisfied and demo data
        # exercises both the free-text and the structured training paths.
        stored = store.record_analysis(
            text, result, consent_share=True, source=rng.choice(DEMO_SOURCES),
            sex=sex, is_demo=True,
        )
        when = now - timedelta(
            days=rng.randint(0, max(days - 1, 0)),
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )
        if stored["event_id"]:
            _backdate_event(stored["event_id"], when)
        if stored["case_id"]:
            _backdate_case(stored["case_id"], when)
        created += 1

    return {"created": created, "failed": failed, "days": days}


def _backdate_event(event_id: str, when: datetime) -> None:
    stamp = when.strftime("%Y-%m-%dT%H:%M:%S")
    with store._lock, store._connect() as conn:
        conn.execute("UPDATE search_events SET created_at = ? WHERE id = ?", (stamp, event_id))


def _backdate_case(case_id: str, when: datetime) -> None:
    stamp = when.strftime("%Y-%m-%dT%H:%M:%S")
    with store._lock, store._connect() as conn:
        conn.execute("UPDATE cases SET created_at = ? WHERE id = ?", (stamp, case_id))


def purge_demo() -> dict[str, int]:
    """Delete only the generated demo rows, leaving real traffic untouched."""
    with store._lock, store._connect() as conn:
        # Demo cases were written with consent so they could train the model, and
        # are linked to their event by case_id. Find them through that link --
        # matching on a source string would miss them, since demo events carry
        # real-looking sources like "web" and "api".
        case_ids = [
            r["case_id"] for r in conn.execute(
                "SELECT DISTINCT case_id FROM search_events WHERE is_demo = 1 AND case_id IS NOT NULL"
            ).fetchall()
        ]
        cur = conn.execute("DELETE FROM search_events WHERE is_demo = 1")
        events_removed = cur.rowcount

        cases_removed = 0
        if case_ids:
            marks = ",".join("?" for _ in case_ids)
            conn.execute(f"DELETE FROM feedback WHERE case_id IN ({marks})", case_ids)
            cases_removed = conn.execute(
                f"DELETE FROM cases WHERE id IN ({marks})", case_ids
            ).rowcount

    return {"events_removed": events_removed, "cases_removed": cases_removed}


def demo_training_examples() -> list[dict[str, Any]]:
    """Scripted (text, label) pairs for the classifier.

    The demo presentations are written by us, so they can be used as labelled
    free text without any consent question. Labels come from the engine's own
    top diagnosis, keeping retraining consistent with live behaviour.
    """
    examples: list[dict[str, Any]] = []
    for _, text, age, _sex in PRESENTATIONS:
        try:
            # record=False: this only needs labels, and a default write here
            # would inject un-flagged events into real analytics.
            result = pipeline.analyse(text=text, top_n=1, age=age, record=False)
        except Exception:
            continue
        dxs = result.get("diagnoses") or []
        if not dxs:
            continue
        examples.append({
            "text": text,
            "symptoms": [s.get("id") for s in (result.get("extracted") or {}).get("symptoms", [])],
            "label": dxs[0]["id"],
            "weight": 1.0,
            "origin": "demo",
        })
    return examples
