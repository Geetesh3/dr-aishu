"""Per-problem action plans for the leading diagnosis.

Answers the question a user actually asks after a diagnosis: *so what do I do
about each thing that is wrong with me?* The diagnosis already carries
condition-level self-care and red flags; this module turns the **reported
symptoms** into a per-problem plan, ordered by how much each symptom
distinguishes the leading condition.

Design rule: nothing here invents clinical guidance. Every line is derived from
data already in the knowledge base -- the condition's `self_care`, its
`when_to_see_doctor`, the symptom's own system and red-flag flag, and the
medicine plan. Where the KB says nothing specific for a symptom we say so
instead of guessing, because a plausible-sounding invented instruction is worse
than an honest gap in a medical tool.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from .kb import KB

# Symptom system -> the everyday measures that support it. These are general
# supportive-care categories, not treatments for a named disease: rest, fluids,
# avoiding irritants. They are deliberately non-specific so they cannot be read
# as a diagnosis or a prescription.
SYSTEM_SUPPORT: dict[str, list[str]] = {
    "general": [
        "Rest more than usual and avoid strenuous activity while you recover.",
        "Keep fluids up through the day, sipping steadily rather than all at once.",
    ],
    "resp": [
        "Keep the air moist and avoid smoke, dust and strong perfumes.",
        "Warm fluids soothe an irritated throat and loosen a cough.",
    ],
    "gi": [
        "Eat small, bland meals and avoid very spicy, oily or heavy food.",
        "Wash hands before eating and after the toilet to avoid spreading it.",
    ],
    "msk": [
        "Protect the sore area: avoid heavy lifting and sudden twisting.",
        "Gentle movement within comfort helps; hard activity usually delays recovery.",
    ],
    "neuro": [
        "Rest in a quiet, dim room and limit screens during bad spells.",
        "Note what you were doing when it started - this helps a clinician later.",
    ],
    "skin": [
        "Keep the area clean and dry, and do not scratch or squeeze it.",
        "Use lukewarm water and a plain, unperfumed moisturiser.",
    ],
    "eye": [
        "Do not rub the eye, and do not share towels or eye make-up.",
        "Wash hands before and after touching the eye or applying drops.",
    ],
    "ear": [
        "Keep the ear dry and do not put anything inside the canal.",
        "Avoid flying or diving until the pain settles.",
    ],
    "urinary": [
        "Drink water steadily through the day to flush the urinary tract.",
        "Pass urine when you feel the need rather than holding on.",
    ],
    "cardio": [
        "Stop and sit down if symptoms come on with effort, and note how long they last.",
        "Avoid heavy exertion until a clinician has assessed this.",
    ],
    "endocrine": [
        "Keep meal times regular and do not skip meals.",
        "Note any change in thirst, weight or energy so it can be tracked.",
    ],
    "psych": [
        "Keep a steady sleep and wake time, and cut back on caffeine late in the day.",
        "Tell someone you trust how you are feeling; you do not have to manage this alone.",
    ],
    "immune": [
        "Rest and keep fluids up; your body is actively fighting something.",
        "Avoid close contact with people who are unwell or immunosuppressed.",
    ],
}

# Which symptoms to act on first. A red-flagged or key symptom matters more than
# a background ache, so the plan is not simply the order the user typed them.
_PRIORITY = {"red_flag": 0, "key": 1, "reported": 2}


def _system_support(symptom_id: str) -> list[str]:
    symptom = KB.symptoms.get(symptom_id)
    if not symptom:
        return []
    return SYSTEM_SUPPORT.get(symptom.system, [])


# `_medicine_covers` runs once per (symptom, medicine-use) pair and the same
# pairs recur on every request, so both the term set and the match result are
# cached. Without this the whole-word scan dominates request time: it was ~44% of
# a warm analysis.
_TERMS_CACHE: dict[str, tuple[str, ...]] = {}
_USE_MATCH_CACHE: dict[tuple[str, str], bool] = {}


def _symptom_terms(symptom_id: str) -> tuple[str, ...]:
    """Normalised label plus synonyms for a symptom, computed once.

    Keyed on the synonym tuple as well as the id, so a knowledge base that is
    reloaded with new synonyms does not serve stale terms.
    """
    symptom = KB.symptoms.get(symptom_id)
    if not symptom:
        return ()
    key = symptom_id + "\x00" + "\x00".join(symptom.synonyms or ())
    cached = _TERMS_CACHE.get(key)
    if cached is not None:
        return cached
    terms = {KB.normalise(symptom.label)}
    terms.update(KB.normalise(syn) for syn in (symptom.synonyms or ()))
    resolved = tuple(t for t in terms if t)
    _TERMS_CACHE[key] = resolved
    return resolved


# Whole-word matcher per term. There are far fewer distinct terms than
# (term, use) pairs, so compiling once per term rather than once per pair removes
# thousands of redundant compilations on the first request that builds a plan.
_USE_TERM_RE: dict[str, re.Pattern] = {}


def _term_re(term: str) -> re.Pattern:
    pattern = _USE_TERM_RE.get(term)
    if pattern is None:
        pattern = re.compile(rf"(?<![a-z]){re.escape(term)}(?![a-z])")
        _USE_TERM_RE[term] = pattern
    return pattern


def _USE_MATCH_RE(term: str, use: str) -> bool:  # noqa: N802 - local helper
    """Memoised whole-word search of `term` in `use`."""
    key = (term, use)
    hit = _USE_MATCH_CACHE.get(key)
    if hit is None:
        hit = _term_re(term).search(use) is not None
        _USE_MATCH_CACHE[key] = hit
    return hit


# Which symptoms a medicine's `uses` cover, memoised per (medicine, symptom).
# The first request that builds a plan used to test every symptom against every
# medicine -- ~800 whole-word searches -- when only the handful of symptoms the
# patient actually reported can ever appear in the plan. Caching per pair means a
# request pays only for the symptoms it has, and repeats are free.
_COVERS_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, bool]] = {}


def _covered_symptoms(
    med_name: str, normalised_uses: list[str], candidates: Iterable[str]
) -> list[str]:
    """Which of `candidates` this medicine's `uses` mention."""
    key = (med_name, tuple(normalised_uses))
    known = _COVERS_CACHE.setdefault(key, {})
    covered: list[str] = []
    for sid in candidates:
        hit = known.get(sid)
        if hit is None:
            hit = _medicine_covers(sid, normalised_uses)
            known[sid] = hit
        if hit:
            covered.append(sid)
    return covered


def _medicine_covers(symptom_id: str, normalised_uses: list[str]) -> bool:
    """Does a medicine's free-text `uses` mention this symptom?

    Whole-word on the symptom label and its synonyms so "pain" in a use does not
    match "painful urination", and "fever" does not match "yellow fever".

    The scan runs once per (symptom, use) pair and the same pairs recur across
    requests, so the answer is memoised. The cache is keyed on the exact terms
    used, which makes it safe against a knowledge base that is reloaded at
    runtime: new terms produce a new key rather than a stale hit.
    """
    terms = _symptom_terms(symptom_id)
    if not terms:
        return False
    for use in normalised_uses:
        for term in terms:
            if _USE_MATCH_RE(term, use):
                return True
    return False


def _problem_note(symptom, steps: list[str], evidence: dict) -> str | None:
    """One short note per problem: the most important thing to say about it."""
    if symptom.red_flag:
        return "This is a warning sign for the leading condition - see the red flags above."
    if evidence.get("certainty") == "uncertain":
        return (
            "You were not sure about this one. If it is present it changes the picture, "
            "so describe it clearly to your clinician."
        )
    if not steps:
        return (
            "No specific supportive measure is recorded for this symptom; "
            "describe it to your clinician."
        )
    return None


def build_solutions(
    facts,
    candidates: list[Any],
    severity,
    prescription,
    max_problems: int = 6,
) -> dict[str, Any]:
    """Build the per-problem action plan for the leading condition.

    Returns a dict with `problems` (one entry per reported symptom, most
    informative first), `condition` (the condition-level plan) and `notes`.
    """
    top = candidates[0] if candidates else None
    if top is None:
        return {"problems": [], "condition": None, "notes": []}

    disease = KB.diseases.get(top.id)
    key_ids = set(disease.key_symptoms) if disease else set()

    # Order the user's own symptoms by how much they point at this condition.
    # Red flags come first regardless of diagnostic weight: a warning sign is
    # something to act on, not just evidence. A symptom that is both stays at the
    # top, so the red-flag test has to run before the key-symptom one.
    def sort_key(item: tuple[str, dict]) -> tuple[int, str]:
        sid = item[0]
        symptom = KB.symptoms.get(sid)
        if symptom and symptom.red_flag:
            rank = _PRIORITY["red_flag"]
        elif sid in key_ids:
            rank = _PRIORITY["key"]
        else:
            rank = _PRIORITY["reported"]
        return (rank, KB.symptom_label(sid))

    ordered = sorted(facts.symptoms.items(), key=sort_key)

    # Map each medicine to the symptoms it plausibly addresses, so a problem can
    # say "this is what the plan already covers" without naming a new drug.
    # Medicines carry free-text `uses` ("fever", "dry irritating cough"), so the
    # link is a whole-word match against the symptom label and its synonyms.
    meds_by_symptom: dict[str, list[str]] = {}
    if prescription is not None:
        candidates = list(facts.symptoms)
        for med in list(prescription.recommended) + list(prescription.prescription_only):
            uses = [KB.normalise(u) for u in (getattr(med, "uses", None) or [])]
            if not uses:
                continue
            for sid in _covered_symptoms(med.name, uses, candidates):
                meds_by_symptom.setdefault(sid, []).append(med.name)

    problems: list[dict[str, Any]] = []
    for sid, evidence in ordered[:max_problems]:
        symptom = KB.symptoms.get(sid)
        if not symptom:
            continue
        steps = list(_system_support(sid))
        covered_by = meds_by_symptom.get(sid, [])

        problems.append({
            "id": sid,
            "label": symptom.label,
            "system": symptom.system,
            "is_key": sid in key_ids,
            "is_red_flag": bool(symptom.red_flag),
            "red_flag_level": symptom.red_flag_level if symptom.red_flag else None,
            "certainty": evidence.get("certainty", "reported"),
            "steps": steps,
            "covered_by": covered_by,
            "note": _problem_note(symptom, steps, evidence),
        })

    condition = {
        "id": top.id,
        "name": top.name,
        "self_care": list(top.self_care),
        "when_to_see_doctor": list(top.when_to_see_doctor),
        "typical_duration_days": top.typical_duration_days,
    }

    notes: list[str] = []
    if top.missing_key_symptoms:
        notes.append(
            "You did not report: " + ", ".join(top.missing_key_symptoms).lower()
            + ". Their absence makes this condition less likely, or means it is early."
        )
    if top.contagious:
        notes.append(
            f"{top.name} can spread to others. Wash hands often and avoid close contact "
            "while you are unwell."
        )
    if len(ordered) > max_problems:
        notes.append(
            f"{len(ordered) - max_problems} further symptom(s) were reported; ask your "
            "clinician about any that worry you."
        )
    if severity.triage in {"emergency", "urgent"}:
        notes.append(
            "At this triage level, do not start new medicines on your own - "
            "get assessed first."
        )

    return {"problems": problems, "condition": condition, "notes": notes}
