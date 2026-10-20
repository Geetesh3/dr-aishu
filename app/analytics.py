"""Aggregate analytics over recorded searches.

Every analysis writes one row to ``search_events`` (see ``store.record_event``).
This module turns those rows into the aggregates the insights dashboard plots:
top symptoms, top diagnoses, triage mix, severity distribution and a daily
timeline.

Everything returned here is a count or an average over structured fields. No
free text is read, so nothing in this module can leak what a user typed.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from . import store
from .kb import KB


def _rows(days: int, include_demo: bool) -> list[dict[str, Any]]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(days, 1))).strftime("%Y-%m-%dT%H:%M:%S")
    sql = "SELECT * FROM search_events WHERE created_at >= ?"
    params: list[Any] = [cutoff]
    if not include_demo:
        sql += " AND is_demo = 0"
    with store._lock, store._connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _label_symptom(sid: str) -> str:
    return KB.symptom_label(sid)


def _label_disease(did: str | None) -> str:
    if not did:
        return "Unclassified"
    d = KB.diseases.get(did)
    return d.name if d else did


def summary(days: int = 30, include_demo: bool = True, top_n: int = 10) -> dict[str, Any]:
    """One payload with every aggregate the dashboard needs."""
    rows = _rows(days, include_demo)

    symptom_counts: Counter[str] = Counter()
    diagnosis_counts: Counter[str] = Counter()
    triage_counts: Counter[str] = Counter()
    severity_counts: Counter[int] = Counter()
    daily_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    confidence_sum = 0.0
    confidence_n = 0
    duration_values: list[float] = []
    temperature_values: list[float] = []
    image_count = 0
    consented = 0

    for r in rows:
        for sid in json.loads(r.get("symptoms") or "[]"):
            symptom_counts[sid] += 1
        if r.get("top_diagnosis"):
            diagnosis_counts[r["top_diagnosis"]] += 1
        if r.get("triage"):
            triage_counts[r["triage"]] += 1
        if r.get("severity_score") is not None:
            severity_counts[int(r["severity_score"])] += 1
        if r.get("created_at"):
            daily_counts[r["created_at"][:10]] += 1
        if r.get("source"):
            source_counts[r["source"]] += 1
        if r.get("confidence") is not None:
            confidence_sum += float(r["confidence"])
            confidence_n += 1
        if r.get("duration_days") is not None:
            duration_values.append(float(r["duration_days"]))
        if r.get("temperature_c") is not None:
            temperature_values.append(float(r["temperature_c"]))
        if r.get("had_image"):
            image_count += 1
        if r.get("consent_share"):
            consented += 1

    # Fill the timeline so the chart shows quiet days as gaps rather than
    # collapsing them, which would make activity look denser than it was.
    today = datetime.now(timezone.utc).date()
    timeline = []
    for offset in range(days - 1, -1, -1):
        day = (today - timedelta(days=offset)).isoformat()
        timeline.append({"date": day, "count": daily_counts.get(day, 0)})

    total = len(rows)
    return {
        "window_days": days,
        "include_demo": include_demo,
        "totals": {
            "searches": total,
            "with_image": image_count,
            "consented_for_training": consented,
            "avg_confidence": round(confidence_sum / confidence_n, 1) if confidence_n else None,
            "avg_duration_days": round(sum(duration_values) / len(duration_values), 1) if duration_values else None,
            "avg_temperature_c": round(sum(temperature_values) / len(temperature_values), 1) if temperature_values else None,
            "distinct_symptoms": len(symptom_counts),
            "distinct_diagnoses": len(diagnosis_counts),
        },
        "top_symptoms": [
            {"id": sid, "label": _label_symptom(sid), "count": n}
            for sid, n in symptom_counts.most_common(top_n)
        ],
        "top_diagnoses": [
            {"id": did, "label": _label_disease(did), "count": n,
             "share": round(100.0 * n / total, 1) if total else 0.0}
            for did, n in diagnosis_counts.most_common(top_n)
        ],
        "triage_mix": [
            {"triage": t, "count": n, "share": round(100.0 * n / total, 1) if total else 0.0}
            for t, n in triage_counts.most_common()
        ],
        "severity_distribution": [
            {"score": s, "count": severity_counts.get(s, 0)} for s in range(0, 11)
        ],
        "timeline": timeline,
        "sources": [{"source": s, "count": n} for s, n in source_counts.most_common()],
    }
