"""Medicine selection, safety filtering and dose-timetable generation.

Selects candidate medicines for the leading diagnoses, filters them against the
patient's allergies, age, pregnancy status and chronic conditions, checks for
pairwise interactions, and lays the survivors out on a clock timetable.

Nothing here prescribes. Every output is labelled as a suggestion to discuss with
a clinician, and prescription-only items are clearly separated from
over-the-counter options.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .analyzer import Candidate
from .kb import KB
from .nlp import ExtractedFacts

SLOTS = [
    ("early_morning", "Early morning", 5 * 60, 8 * 60),
    ("morning", "Morning", 8 * 60, 11 * 60),
    ("midday", "Midday", 11 * 60, 14 * 60),
    ("afternoon", "Afternoon", 14 * 60, 17 * 60),
    ("evening", "Evening", 17 * 60, 20 * 60),
    ("night", "Night", 20 * 60, 24 * 60),
]

# Cross-condition safety rules applied on top of the per-drug data.
CONDITION_CONTRAINDICATION_HINTS = {
    "peptic ulcer": ["ibuprofen", "diclofenac", "indomethacin", "aspirin", "prednisolone"],
    "kidney disease": ["ibuprofen", "diclofenac", "indomethacin", "metformin", "spironolactone", "furosemide"],
    "heart failure": ["ibuprofen", "diclofenac", "prednisolone", "spironolactone"],
    "asthma": ["propranolol", "bisoprolol", "aspirin"],
    "diabetes": ["prednisolone", "glucosamine"],
    "liver disease": ["paracetamol", "methotrexate", "clarithromycin", "fluconazole"],
    "epilepsy": ["metoclopramide", "ciprofloxacin"],
    "thyroid disease": ["antacid", "ferrous_sulfate", "omeprazole", "ispaghula"],
}

ALLERGY_CLASS_HINTS = {
    "penicillin": ["amoxicillin", "penicillin_v", "cloxacillin"],
    "sulfa": ["silver_sulfadiazine", "hydrochlorothiazide"],
    "sulfonamide": ["silver_sulfadiazine", "hydrochlorothiazide"],
    "nsaid": ["ibuprofen", "diclofenac", "indomethacin", "aspirin"],
    "aspirin": ["aspirin", "ibuprofen"],
    "shellfish": ["glucosamine"],
    "macrolide": ["azithromycin", "clarithromycin"],
    "cephalosporin": [],
    "codeine": [],
}


@dataclass
class SelectedMedicine:
    id: str
    name: str
    drug_class: str
    otc: bool
    prescription_note: str | None
    dose: str
    frequency: str
    times: list[str]
    with_food: str
    max_days: int | None
    uses: list[str]
    warnings: list[str]
    side_effects: list[str]
    reason: str
    for_conditions: list[str] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    # Ceiling the user must not cross; the UI advises "never exceed the maximum
    # daily dose", so the number itself has to reach the client.
    max_daily: str | None = None
    # Indian brand names for this generic, from the merged pharmacy catalogue.
    # Presentation only: brands never affect which medicine is selected.
    brands: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExcludedMedicine:
    id: str
    name: str
    reason: str


@dataclass
class CurrentMedication:
    """A medicine the patient reports already taking, resolved if possible."""
    reported: str
    medicine_id: str | None
    name: str
    resolved: bool


@dataclass
class MedicationAlert:
    """A clash between a suggested medicine and one the patient already takes."""
    suggested: str
    existing: str
    note: str


@dataclass
class PrescriptionPlan:
    recommended: list[SelectedMedicine] = field(default_factory=list)
    prescription_only: list[SelectedMedicine] = field(default_factory=list)
    excluded: list[ExcludedMedicine] = field(default_factory=list)
    interactions: list[dict[str, Any]] = field(default_factory=list)
    timetable: list[dict[str, Any]] = field(default_factory=list)
    general_advice: list[str] = field(default_factory=list)
    current_medications: list[CurrentMedication] = field(default_factory=list)
    medication_alerts: list[MedicationAlert] = field(default_factory=list)


def _slot_for(time_str: str) -> str:
    if not time_str or ":" not in time_str:
        return "as_needed"
    hh, mm = (int(x) for x in time_str.split(":")[:2])
    minutes = hh * 60 + mm
    for key, _label, lo, hi in SLOTS:
        if lo <= minutes < hi:
            return key
    return "night"


def _dose_for(med: dict[str, Any], facts: ExtractedFacts) -> str:
    child = facts.age is not None and facts.age < 18
    if child and med.get("dose_child") and med["dose_child"] != "not recommended":
        base = med["dose_child"]
        if facts.weight_kg and "mg/kg" in base:
            return f"{base} (calculated for {facts.weight_kg:.0f} kg)"
        return base
    return med["dose_adult"]


def _max_daily_for(med: dict[str, Any], facts: ExtractedFacts) -> str | None:
    """The daily ceiling to show the user, chosen for their age band."""
    child = facts.age is not None and facts.age < 18
    value = med.get("max_daily_child") if child else med.get("max_daily_adult")
    if not value or value in ("not recommended", "n/a"):
        return med.get("max_daily_adult")
    return value


def _allergy_conflicts(med: dict[str, Any], allergies: list[str]) -> list[str]:
    hits: list[str] = []
    text = " ".join(allergies).lower()
    for allergen, drug_ids in ALLERGY_CLASS_HINTS.items():
        if allergen in text and med["id"] in drug_ids:
            hits.append(f"allergy to {allergen}")
    for allergy in allergies:
        a = allergy.strip().lower()
        if len(a) < 3:
            continue
        if a in med["name"].lower() or a in (med.get("generic") or "").lower() or a in med["id"]:
            hits.append(f"allergy to {allergy.strip()}")
    return hits


def _condition_conflicts(med: dict[str, Any], conditions: list[str]) -> list[str]:
    hits: list[str] = []
    for condition in conditions:
        c = condition.lower()
        for key, drug_ids in CONDITION_CONTRAINDICATION_HINTS.items():
            if key in c and med["id"] in drug_ids:
                hits.append(f"caution with {condition}")
    for contra in med.get("contraindications", []):
        cl = contra.lower()
        for condition in conditions:
            if condition.lower() in cl:
                hits.append(f"contraindicated in {condition}")
    return hits


def _resolve_current_medications(facts: ExtractedFacts) -> list[CurrentMedication]:
    """Resolve each reported medicine to a curated id where the formulary has it.

    A drug the formulary does not carry (warfarin, for example) still matters:
    it appears as an interaction partner on other records, so the raw name is
    kept and matched by name later.
    """
    out: list[CurrentMedication] = []
    for reported in facts.current_medicines:
        med_id = KB.resolve_medicine(reported)
        if med_id:
            out.append(CurrentMedication(
                reported=reported, medicine_id=med_id,
                name=KB.medicines[med_id]["name"], resolved=True))
        else:
            out.append(CurrentMedication(
                reported=reported, medicine_id=None, name=reported.strip(), resolved=False))
    return out


def _interaction_tokens(current: list[CurrentMedication]) -> set[str]:
    """Every string that could identify an existing medicine in `interacts_with`.

    Both the resolved id and the words the patient actually typed are included,
    because a drug missing from the formulary is only identifiable by its name.
    """
    tokens: set[str] = set()
    for entry in current:
        if entry.medicine_id:
            tokens.add(entry.medicine_id.lower())
        for word in re.split(r"[^a-z0-9]+", entry.reported.lower()):
            if len(word) >= 4:
                tokens.add(word)
    return tokens


def _existing_medication_conflicts(
    med: dict[str, Any], tokens: set[str]
) -> list[str]:
    """Names of the patient's existing medicines that clash with `med`."""
    clashes: list[str] = []
    for partner in med.get("interacts_with", []):
        partner_key = str(partner).lower()
        if partner_key in tokens:
            clashes.append(partner_key)
    return sorted(set(clashes))


def _severity_permits_rx(candidate: Candidate, triage: str) -> bool:
    """Only surface prescription-only medicines when a clinician is already involved.

    A condition the patient states they already have is the exception. Their
    ongoing treatment is decided by their clinician, not by the severity of
    today's unrelated complaint, so those options are listed for discussion
    rather than suppressed because the current triage reads as self-care.
    """
    if candidate.declared_condition:
        return True
    return triage in ("see_doctor", "urgent", "emergency")


def select_medicines(
    candidates: list[Candidate],
    facts: ExtractedFacts,
    triage: str,
    max_conditions: int = 6,
    include_rx: bool = True,
    min_relative_confidence: float = 0.55,
) -> PrescriptionPlan:
    """Select medicines for the leading diagnoses, and for every condition the
    patient declared they already have.

    Diagnoses within `min_relative_confidence` of the top candidate contribute
    medicines. This prevents treatments from a weak alternative differential (for
    example an antimalarial when influenza is far more likely) leaking into the
    plan. Declared chronic conditions are exempt from that cutoff, because the
    patient named them outright and they are the reason many consultations
    happen; they are ranked first so a long-term condition is never crowded out
    by a self-limiting acute one.
    """
    plan = PrescriptionPlan()
    seen: set[str] = set()
    excluded_seen: set[str] = set()
    # Drug classes already represented, so two medicines of the same class
    # (for example ibuprofen and diclofenac, both NSAIDs) are never both
    # suggested - that would silently double the dose and the risk.
    classes_used: dict[str, str] = {}

    allergies = list(facts.allergies)
    conditions = list(facts.chronic_conditions)
    if facts.is_pregnant and "pregnancy" not in conditions:
        conditions.append("pregnancy")

    # What the patient already takes, and the tokens used to match it against
    # the formulary's interaction data.
    plan.current_medications = _resolve_current_medications(facts)
    existing_tokens = _interaction_tokens(plan.current_medications)
    if plan.current_medications:
        plan.general_advice.append(
            "Your existing medicines were checked against every suggestion below."
        )

    if candidates:
        cutoff = candidates[0].confidence * min_relative_confidence
        selected = [c for c in candidates if c.confidence >= cutoff][:max_conditions]
        # A condition the patient named as their own (high blood pressure, thyroid
        # disease, diabetes) must contribute its medicines even when today's
        # symptoms outrank it. Otherwise the plan treats only the acute complaint
        # and silently drops the long-term problem the patient actually asked about.
        declared = [c for c in candidates if c.declared_condition and c.medicine_ids]
        selected_ids = {c.id for c in selected}
        for c in declared:
            if c.id not in selected_ids:
                selected.append(c)
                selected_ids.add(c.id)
        # Chronic conditions take priority when the cap bites, because an acute
        # complaint like a cold is expected to resolve on its own.
        selected.sort(key=lambda c: (not c.declared_condition, -c.confidence))
        selected = selected[:max_conditions]
    else:
        selected = []

    for candidate in selected:
        for med_id in candidate.medicine_ids:
            if med_id in seen:
                # Still record which other condition it serves.
                for m in plan.recommended + plan.prescription_only:
                    if m.id == med_id and candidate.name not in m.for_conditions:
                        m.for_conditions.append(candidate.name)
                continue

            med = KB.medicines.get(med_id)
            if not med:
                continue

            # Age floor
            if facts.age is not None and med.get("age_min") is not None:
                if facts.age < med["age_min"]:
                    if med_id not in excluded_seen:
                        plan.excluded.append(ExcludedMedicine(
                            id=med_id, name=med["name"],
                            reason=f"not recommended under {med['age_min']} years"))
                        excluded_seen.add(med_id)
                    continue

            # Pregnancy
            if facts.is_pregnant and med.get("pregnancy") in ("avoid", "contraindicated"):
                if med_id not in excluded_seen:
                    plan.excluded.append(ExcludedMedicine(
                        id=med_id, name=med["name"],
                        reason=f"avoid in pregnancy ({med.get('pregnancy')})"))
                    excluded_seen.add(med_id)
                continue

            conflicts = _allergy_conflicts(med, allergies) + _condition_conflicts(med, conditions)
            if conflicts:
                if med_id not in excluded_seen:
                    plan.excluded.append(ExcludedMedicine(
                        id=med_id, name=med["name"], reason="; ".join(sorted(set(conflicts)))))
                    excluded_seen.add(med_id)
                continue

            # Existing-medication interaction. Checked before the drug is
            # offered, because the patient cannot be expected to spot that a new
            # suggestion clashes with something they already take.
            existing_clashes = _existing_medication_conflicts(med, existing_tokens)
            if existing_clashes:
                label = ", ".join(KB.medicines[c]["name"] if c in KB.medicines else c
                                  for c in existing_clashes)
                if med_id not in excluded_seen:
                    plan.excluded.append(ExcludedMedicine(
                        id=med_id, name=med["name"],
                        reason=f"may interact with a medicine you already take ({label})"))
                    excluded_seen.add(med_id)
                    plan.medication_alerts.append(MedicationAlert(
                        suggested=med["name"], existing=label,
                        note=f"{med['name']} may interact with {label}, which you reported taking. "
                             f"Your clinician can choose a safer alternative."))
                continue

            # Same-class duplication guard.
            drug_class = med.get("drug_class") or ""
            if drug_class and drug_class in classes_used and classes_used[drug_class] != med_id:
                if med_id not in excluded_seen:
                    plan.excluded.append(ExcludedMedicine(
                        id=med_id, name=med["name"],
                        reason=f"same drug class as {classes_used[drug_class]}, already suggested"))
                    excluded_seen.add(med_id)
                continue

            # Prescription-only handling
            if not med["otc"] and not include_rx:
                if med_id not in excluded_seen:
                    plan.excluded.append(ExcludedMedicine(
                        id=med_id, name=med["name"], reason="prescription-only medicine"))
                    excluded_seen.add(med_id)
                continue

            entry = SelectedMedicine(
                id=med_id,
                name=med["name"],
                drug_class=med["drug_class"],
                otc=med["otc"],
                prescription_note=med.get("prescription_note"),
                dose=_dose_for(med, facts),
                frequency=med["frequency"],
                times=list(med["times"]),
                with_food=med["with_food"],
                max_days=med.get("max_days"),
                max_daily=_max_daily_for(med, facts),
                uses=list(med.get("uses", [])),
                warnings=list(med.get("warnings", [])),
                side_effects=list(med.get("side_effects", [])),
                reason=f"commonly used for {candidate.name}",
                for_conditions=[candidate.name],
                brands=KB.brands_for(med_id, 4),
            )
            seen.add(med_id)
            if drug_class:
                classes_used.setdefault(drug_class, med["name"])

            if med["otc"]:
                plan.recommended.append(entry)
            else:
                if _severity_permits_rx(candidate, triage):
                    entry.safety_notes.append(
                        "Prescription-only: this must be authorised by a clinician before use.")
                    plan.prescription_only.append(entry)
                else:
                    plan.excluded.append(ExcludedMedicine(
                        id=med_id, name=med["name"],
                        reason="prescription-only and current severity does not require it"))
                    excluded_seen.add(med_id)

    plan.interactions = check_interactions(
        [m.id for m in plan.recommended] + [m.id for m in plan.prescription_only]
    )
    plan.timetable = build_timetable(plan.recommended + plan.prescription_only)
    plan.general_advice = _general_advice(facts, triage)
    return plan


def check_interactions(med_ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()
    for med_id in med_ids:
        med = KB.medicines.get(med_id)
        if not med:
            continue
        for other_id in med.get("interacts_with", []):
            if other_id not in med_ids:
                continue
            pair = frozenset({med_id, other_id})
            if pair in seen:
                continue
            seen.add(pair)
            other = KB.medicines.get(other_id, {})
            out.append({
                "between": [med["name"], other.get("name", other_id)],
                "severity": "moderate",
                "note": f"{med['name']} and {other.get('name', other_id)} may interact. "
                        f"Discuss timing and dose with a pharmacist or doctor.",
            })
    return out


def build_timetable(medicines: list[SelectedMedicine]) -> list[dict[str, Any]]:
    """Group medicines into time slots ordered by clock time."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    as_needed: list[dict[str, Any]] = []

    for med in medicines:
        if not med.times or med.times == ["as needed"] or med.times == ["as prescribed"]:
            as_needed.append({
                "medicine": med.name,
                "dose": med.dose,
                "instruction": f"{med.frequency} - {med.with_food}",
                "otc": med.otc,
            })
            continue
        for t in med.times:
            slot = _slot_for(t)
            buckets[slot].append({
                "time": t,
                "medicine": med.name,
                "dose": med.dose,
                "with_food": med.with_food,
                "otc": med.otc,
                "max_days": med.max_days,
            })

    order = [s[0] for s in SLOTS]
    timetable: list[dict[str, Any]] = []
    for key in order:
        if key not in buckets:
            continue
        label = next(s[1] for s in SLOTS if s[0] == key)
        entries = sorted(buckets[key], key=lambda e: e["time"])
        timetable.append({"slot": key, "label": label, "entries": entries})

    if as_needed:
        timetable.append({"slot": "as_needed", "label": "As needed", "entries": as_needed})
    return timetable


def _general_advice(facts: ExtractedFacts, triage: str) -> list[str]:
    advice: list[str] = []
    if triage == "emergency":
        advice.append("Call emergency services now. Do not take new medicines without guidance.")
    elif triage == "urgent":
        advice.append("Seek urgent medical assessment today before starting any new medicine.")
    else:
        advice.append("These are general suggestions - confirm with a pharmacist or doctor before use.")

    if facts.allergies:
        advice.append(f"Known allergies recorded ({', '.join(facts.allergies)}) were used to filter this list.")
    if facts.is_pregnant:
        advice.append("You are pregnant - many medicines are unsuitable. Confirm everything with your clinician.")
    if facts.age is not None and facts.age < 12:
        advice.append("Child dosing must be weight-based and confirmed by a paediatric clinician.")
    if facts.chronic_conditions:
        advice.append("Chronic conditions were checked against each medicine's contraindications.")
    advice.append("Never exceed the stated maximum daily dose. Stop and seek help if a rash, swelling "
                  "or breathing difficulty develops.")
    return advice
