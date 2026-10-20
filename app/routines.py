"""Personalised daily routine generation.

Builds a time-blocked day plan from the routine template attached to the leading
diagnosis, then personalises it: medicine slots from the prescription plan are
merged in, sleep/wake times shift with age, activity is limited by severity, and
condition-specific extra blocks (glucose checks, hydration targets) are added.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .analyzer import Candidate, SeverityAssessment
from .kb import KB
from .nlp import ExtractedFacts
from .prescription import PrescriptionPlan


@dataclass
class RoutinePlan:
    name: str
    focus: str
    blocks: list[dict[str, Any]] = field(default_factory=list)
    habits: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    hydration_target_l: float = 2.0
    sleep_target_hours: float = 8.0
    notes: list[str] = field(default_factory=list)


def _minutes(t: str) -> int:
    try:
        hh, mm = (int(x) for x in t.split(":")[:2])
        return hh * 60 + mm
    except Exception:
        return 12 * 60


def _fmt(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _shift(blocks: list[dict[str, Any]], delta_minutes: int) -> list[dict[str, Any]]:
    out = []
    for b in blocks:
        if b.get("time") in ("NOW", None):
            out.append(dict(b))
            continue
        out.append({**b, "time": _fmt(_minutes(b["time"]) + delta_minutes)})
    return out


def build_routine(
    candidates: list[Candidate],
    severity: SeverityAssessment,
    facts: ExtractedFacts,
    prescription: PrescriptionPlan | None,
) -> RoutinePlan:
    top = candidates[0] if candidates else None
    # An emergency always uses the emergency response template, whatever the
    # leading differential suggests.
    if severity.triage == "emergency":
        routine_key = "emergency_care"
    else:
        routine_key = top.routine if top and top.routine in KB.routines else "general_wellness"
    template = KB.routines.get(routine_key) or KB.routines["general_wellness"]

    blocks = [dict(b) for b in template["blocks"]]
    notes: list[str] = []

    # --- personalise wake/sleep timing -------------------------------------
    if facts.age is not None:
        if facts.age < 5:
            blocks = _shift(blocks, 30)
            notes.append("Routine shifted later for a young child; naps are expected in addition.")
        elif facts.age >= 65:
            blocks = _shift(blocks, -30)
            notes.append("Earlier schedule suggested for an older adult, with a mid-afternoon rest.")

    # --- severity limits activity ------------------------------------------
    if severity.triage in ("urgent", "emergency"):
        blocks = [b for b in blocks if b.get("category") not in ("activity", "work")]
        notes.append("Activity and work blocks removed while urgent care is arranged. Rest is the priority.")
    elif severity.triage == "see_doctor":
        for b in blocks:
            if b.get("category") == "activity":
                b["detail"] += " Keep it gentle and stop if symptoms worsen."

    # --- merge medicine slots ----------------------------------------------
    med_blocks: list[dict[str, Any]] = []
    for slot in (prescription.timetable if prescription else []):
        for entry in slot["entries"]:
            time = entry.get("time")
            if not time:
                continue
            detail = f"{entry['medicine']} - {entry['dose']}"
            if entry.get("with_food"):
                detail += f" ({entry['with_food']})"
            med_blocks.append({
                "time": time,
                "title": f"Medicine: {entry['medicine']}",
                "detail": detail,
                "category": "medication",
            })
    # As-needed medicines become a standing reminder rather than a fixed slot.
    for slot in (prescription.timetable if prescription else []):
        if slot["slot"] == "as_needed":
            for entry in slot["entries"]:
                notes.append(f"As-needed medicine available: {entry['medicine']} - {entry['instruction']}")

    for mb in med_blocks:
        # Avoid duplicating an existing medication block at the same time.
        if not any(b.get("category") == "medication" and b.get("time") == mb["time"] for b in blocks):
            blocks.append(mb)

    # --- condition-specific extras -----------------------------------------
    conditions = {c.lower() for c in facts.chronic_conditions}
    if "diabetes" in conditions or (top and top.category == "endocrine"):
        notes.append("Check blood glucose before breakfast and record the reading.")
    if facts.is_pregnant:
        notes.append("Add a prenatal vitamin with folic acid as advised by your clinician.")
    if top and top.contagious:
        notes.append("You may be contagious - rest at home, ventilate rooms and avoid close contact.")
    if top and top.category == "respiratory":
        notes.append("Monitor breathing. If it becomes hard to speak a full sentence, seek urgent care.")

    # --- habits and things to avoid ----------------------------------------
    habits = list(top.self_care) if top else [
        "Drink 2 L of water across the day",
        "Take a 20-30 minute walk",
        "Keep a consistent sleep and wake time",
    ]
    habits.extend([
        "Keep the same sleep and wake time every day, including weekends",
        "Take screen breaks every 45 minutes",
    ])

    avoid: list[str] = []
    if top:
        if top.category == "respiratory":
            avoid += ["Smoking and second-hand smoke", "Cold drinks if they trigger coughing"]
        if top.category == "gastrointestinal":
            avoid += ["Spicy, oily and very fatty food", "Alcohol and excess caffeine", "Lying down within 3 hours of eating"]
        if top.category == "endocrine":
            avoid += ["Refined sugar and sugary drinks", "Skipping meals"]
        if top.category in ("cardiovascular",):
            avoid += ["Adding salt at the table", "Heavy lifting until cleared by a doctor"]
        if top.category == "skin":
            avoid += ["Scratching the affected area", "Scented soaps and harsh detergents"]
        if top.category == "musculoskeletal":
            avoid += ["High-impact activity during a flare", "Prolonged sitting without a break"]
    if facts.is_pregnant:
        avoid.append("Alcohol and unpasteurised foods")
    for med in (prescription.recommended if prescription else []):
        for w in med.warnings:
            if "avoid" in w.lower() or "do not" in w.lower():
                avoid.append(f"{med.name}: {w}")

    # --- hydration / sleep targets -----------------------------------------
    hydration = 2.0
    if top and top.category in ("gastrointestinal", "urinary", "infectious"):
        hydration = 3.0
    elif top and top.category == "respiratory":
        hydration = 2.5
    if facts.weight_kg:
        hydration = max(hydration, round(facts.weight_kg * 0.033, 1))
    if facts.age is not None and facts.age < 12:
        hydration = max(1.2, round(hydration * 0.6, 1))

    sleep_hours = 8.0
    if facts.age is not None and facts.age < 5:
        sleep_hours = 11.0
    elif facts.age is not None and facts.age < 13:
        sleep_hours = 10.0
    elif facts.age is not None and facts.age >= 65:
        sleep_hours = 7.5
    if severity.triage in ("urgent", "emergency"):
        sleep_hours = max(sleep_hours, 9.0)

    blocks.sort(key=lambda b: (_minutes(b["time"]) if b.get("time") not in ("NOW", None) else -1))

    return RoutinePlan(
        name=template["name"],
        focus=template["focus"],
        blocks=blocks,
        habits=list(dict.fromkeys(habits)),
        avoid=list(dict.fromkeys(avoid)),
        hydration_target_l=hydration,
        sleep_target_hours=sleep_hours,
        notes=notes,
    )
