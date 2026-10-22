"""Differential diagnosis ranking and severity / triage assessment.

The scorer combines symptom coverage, key-symptom coverage and disease
prevalence, then adjusts for onset modifiers, explicit negations and duration.
The severity engine layers red flags, vitals, age, chronic conditions and
pregnancy on top of the condition's own severity profile to produce a triage
level with an auditable list of contributing factors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .kb import KB
from .nlp import ExtractedFacts

MODIFIER_MULTIPLIERS = {
    "sudden_onset": 1.06,
    "gradual_onset": 1.02,
    "worsening": 1.10,
    "improving": 0.94,
    "recurrent": 1.05,
}

# A condition the patient states they already have is evidence for that
# diagnosis even when they report no symptom of it. High blood pressure and
# hypothyroidism are usually symptomless, so without this they never appeared in
# the ranked list no matter how plainly the patient named them. The value is a
# multiplier applied to the knowledge-base score, deliberately below what a real
# symptom match earns, so a declared condition is acknowledged without
# displacing a diagnosis the current symptoms actually support.
DECLARED_CONDITION_BOOST = {
    "hypertension": {"hypertension": 2.6},
    "thyroid disease": {"hypothyroidism": 2.2, "hyperthyroidism": 1.4},
    "diabetes": {"type2_diabetes": 2.6},
    "asthma": {"asthma": 2.4},
    "heart failure": {"heart_failure": 2.4},
    "heart disease": {"angina": 1.8, "heart_failure": 1.5},
    "kidney disease": {"uti": 1.3},
    "copd": {"acute_bronchitis": 1.6, "asthma": 1.3},
    "tuberculosis": {"pneumonia": 1.3},
    "epilepsy": {},
}

# When one declared condition maps to several diseases, only the single most
# likely one is treated as "declared". "Thyroid problem" is the reason this
# exists: mapping it to both hypothyroidism and hyperthyroidism would put
# levothyroxine and carbimazole -- drugs that do opposite things -- in the same
# plan. The others stay in the ranked differential for the clinician to see.
DECLARED_PRIMARY = {
    condition: max(mapping, key=mapping.get)
    for condition, mapping in DECLARED_CONDITION_BOOST.items()
    if mapping
}

TRIAGE_LEVELS = ["self_care", "see_doctor", "urgent", "emergency"]

TRIAGE_META = {
    "self_care": {
        "label": "Self-care at home",
        "advice": "Manage at home with the care plan below. Seek medical advice if symptoms worsen or do not improve.",
        "colour": "green",
    },
    "see_doctor": {
        "label": "See a doctor soon",
        "advice": "Book a medical appointment within 24-72 hours for assessment.",
        "colour": "amber",
    },
    "urgent": {
        "label": "Urgent medical attention",
        "advice": "Go to an urgent care centre or hospital today. Do not wait for symptoms to settle.",
        "colour": "orange",
    },
    "emergency": {
        "label": "EMERGENCY",
        "advice": "Call emergency services or go to the nearest emergency department immediately.",
        "colour": "red",
    },
}

# How the leading condition's severity profile is applied. A profile describes a
# condition that is actually present; the leading candidate is only a guess until
# enough symptoms agree, so the profile is blended toward the middle of the scale
# in proportion to the evidence. `_SEVERITY_SUPPORT_SYMPTOMS` is the match count
# at which the profile is applied in full. Red-flag floors are applied afterwards
# and are unaffected, so a genuine emergency still escalates.
_NEUTRAL_SEVERITY = 5.0
_SEVERITY_SUPPORT_SYMPTOMS = 4.0


@dataclass
class Candidate:
    id: str
    name: str
    category: str
    icd10: str
    confidence: float
    raw_score: float
    matched_symptoms: list[str] = field(default_factory=list)
    missing_key_symptoms: list[str] = field(default_factory=list)
    description: str = ""
    severity_profile: str = "mild"
    severity_score: int = 3
    typical_duration_days: list[float] = field(default_factory=list)
    contagious: bool = False
    self_care: list[str] = field(default_factory=list)
    when_to_see_doctor: list[str] = field(default_factory=list)
    medicine_ids: list[str] = field(default_factory=list)
    routine: str = ""
    # True when the patient named this condition as something they already have,
    # rather than it being inferred from today's symptoms. Chronic conditions
    # like high blood pressure produce no symptoms of their own, so they must be
    # carried through to the medicine plan even though the acute differential
    # outranks them.
    declared_condition: bool = False


@dataclass
class SeverityAssessment:
    score: int
    label: str
    triage: str
    triage_label: str
    advice: str
    colour: str
    red_flags: list[str] = field(default_factory=list)
    factors: list[dict[str, Any]] = field(default_factory=list)
    duration_assessment: dict[str, Any] = field(default_factory=dict)


def _disease_score(disease, present_ids: set[str], modifiers: list[str], negated: set[str]) -> tuple[float, list[str], list[str]]:
    total_possible = sum(disease.symptoms.values()) or 1.0
    matched_weight = 0.0
    matched: list[str] = []

    for sid, weight in disease.symptoms.items():
        if sid in present_ids:
            sym_weight = KB.symptoms[sid].weight if sid in KB.symptoms else 1.0
            matched_weight += weight * sym_weight
            matched.append(sid)

    coverage = matched_weight / total_possible

    key = disease.key_symptoms
    key_hit = sum(1 for k in key if k in present_ids)
    key_coverage = (key_hit / len(key)) if key else 0.0
    missing_key = [k for k in key if k not in present_ids]

    score = 0.55 * coverage + 0.30 * key_coverage + 0.15 * disease.prevalence

    # Require more than one supporting symptom before granting full weight, so a
    # single common symptom (e.g. "headache") cannot rank a condition highly.
    evidence_factor = min(1.0, 0.55 + 0.225 * len(matched))
    score *= evidence_factor

    for mod in modifiers:
        score *= MODIFIER_MULTIPLIERS.get(mod, 1.0)

    # Explicit denial of a defining symptom is strong evidence against the condition.
    denied_keys = [k for k in key if k in negated]
    if denied_keys:
        score *= max(0.35, 1.0 - 0.25 * len(denied_keys))
    # Any single symptom present gives partial credit even without key coverage.
    if matched and key_hit == 0:
        score *= 0.85
    return score, matched, missing_key


def diagnose(
    facts: ExtractedFacts,
    top_n: int = 6,
    ml_predictions: list[dict[str, Any]] | None = None,
    kb_weight: float = 0.65,
    lab_boosts: dict[str, float] | None = None,
) -> list[Candidate]:
    """Rank candidate conditions.

    Blends two independent signals: the auditable knowledge-base score and, when
    a trained model is available, the classifier's probability distribution. The
    knowledge base keeps the larger share so results stay explainable; the model
    contributes the patterns it learned from real and synthetic cases.

    ``lab_boosts`` multiplies the score of a condition the laboratory findings
    support. A value can also raise a condition the text never named -- that is
    the point of reading a report -- but such a lab-only candidate enters at a
    low base score so it can never outrank a condition with real symptoms.
    """
    present_ids = set(facts.symptoms.keys())
    negated = set(facts.negated_symptoms)
    boosts = lab_boosts or {}
    if not present_ids and not ml_predictions and not boosts:
        return []

    # Fold the patient's own declared conditions into the boost table. These are
    # merged rather than added so a condition the patient names cannot stack with
    # a lab boost into an unjustified ranking.
    for condition in facts.chronic_conditions:
        for did, mult in DECLARED_CONDITION_BOOST.get(condition, {}).items():
            boosts[did] = max(boosts.get(did, 1.0), mult)

    scored: dict[str, tuple[float, Any, list[str], list[str]]] = {}
    for disease in KB.diseases.values():
        score, matched, missing_key = _disease_score(disease, present_ids, facts.modifiers, negated)
        boost = boosts.get(disease.id, 1.0)
        if matched:
            scored[disease.id] = (score * boost, disease, matched, missing_key)
        elif boost > 1.0:
            # A declared condition with no reported symptom enters at a floor
            # score: it is real and worth listing, but it is not an acute finding.
            scored[disease.id] = (0.18 * boost, disease, [], list(disease.key_symptoms))

    # Fold in model predictions, including conditions the lexicon missed entirely.
    ml_map: dict[str, float] = {}
    if ml_predictions:
        for p in ml_predictions:
            label = p.get("label")
            if label in KB.diseases:
                ml_map[label] = float(p.get("probability", 0.0))

    if not scored and not ml_map:
        return []

    max_kb = max((v[0] for v in scored.values()), default=1.0) or 1.0
    max_ml = max(ml_map.values(), default=0.0) or 1.0

    blended: dict[str, float] = {}
    for did, (score, *_rest) in scored.items():
        kb_norm = score / max_kb
        ml_norm = (ml_map.get(did, 0.0) / max_ml) if ml_map else 0.0
        blended[did] = kb_weight * kb_norm + (1 - kb_weight) * ml_norm

    for did, prob in ml_map.items():
        if did in blended:
            continue
        disease = KB.diseases[did]
        # A model-only candidate keeps its probability but is discounted, because
        # no knowledge-base symptom matched it.
        blended[did] = (1 - kb_weight) * (prob / max_ml) * 0.85
        scored[did] = (0.0, disease, [], [k for k in disease.key_symptoms])

    ranked = sorted(blended.items(), key=lambda kv: -kv[1])[: max(top_n, 1)]

    # Convert blend scores into an interpretable distribution. Softmax with a
    # low temperature sharpens the gap between the leader and the rest, so a
    # clearly separated diagnosis reads high while an ambiguous one reads low.
    temperature = 0.22
    top_score = ranked[0][1] or 1.0
    exps = [(did, pow(2.718281828, (score / top_score) / temperature)) for did, score in ranked]
    total_exp = sum(e for _, e in exps) or 1.0
    shares = {did: (e / total_exp) for did, e in exps}

    candidates: list[Candidate] = []
    for did, blend_score in ranked:
        _score, disease, matched, missing_key = scored[did]
        # A ranking share only says "best of these"; it says nothing about how
        # much evidence exists. Cap confidence by the number of supporting
        # symptoms so a one-symptom guess cannot read as near-certain.
        evidence_cap = min(94.0, 28.0 + 9.0 * len(matched))
        confidence = round(min(100 * shares[did], evidence_cap), 1)
        candidates.append(
            Candidate(
                id=disease.id,
                name=disease.name,
                category=disease.category,
                icd10=disease.icd10,
                confidence=min(confidence, 96.0),
                raw_score=round(blend_score, 4),
                matched_symptoms=[KB.symptom_label(s) for s in matched],
                missing_key_symptoms=[KB.symptom_label(s) for s in missing_key],
                description=disease.description,
                severity_profile=disease.severity_profile,
                severity_score=disease.severity_score,
                typical_duration_days=disease.typical_duration_days,
                contagious=disease.contagious,
                self_care=disease.self_care,
                when_to_see_doctor=disease.when_to_see_doctor,
                medicine_ids=disease.medicines,
                routine=disease.routine,
                declared_condition=any(
                    disease.id == DECLARED_PRIMARY.get(c)
                    for c in facts.chronic_conditions
                ),
            )
        )
    return candidates


def _collect_red_flags(facts: ExtractedFacts) -> tuple[list[str], list[str]]:
    """Return (emergency_flags, urgent_flags) with human-readable labels."""
    emergency: list[str] = []
    urgent: list[str] = []
    for sid in facts.symptoms:
        symptom = KB.symptoms.get(sid)
        if not symptom or not symptom.red_flag:
            continue
        (emergency if symptom.red_flag_level == "emergency" else urgent).append(symptom.label)

    if facts.oxygen_saturation is not None and facts.oxygen_saturation < 94:
        emergency.append(f"Low oxygen saturation ({facts.oxygen_saturation}%)")
    if facts.temperature_c is not None and facts.temperature_c >= 40:
        emergency.append(f"Very high temperature ({facts.temperature_c} C)")
    elif facts.temperature_c is not None and facts.temperature_c >= 39:
        urgent.append(f"High temperature ({facts.temperature_c} C)")
    if facts.systolic_bp is not None and facts.systolic_bp >= 180:
        emergency.append(f"Very high systolic blood pressure ({facts.systolic_bp} mmHg)")
    if facts.systolic_bp is not None and facts.systolic_bp <= 90:
        emergency.append(f"Low systolic blood pressure ({facts.systolic_bp} mmHg)")
    if facts.pulse is not None and (facts.pulse >= 130 or facts.pulse <= 45):
        emergency.append(f"Abnormal heart rate ({facts.pulse} bpm)")

    # Laboratory findings that change urgency on their own. A platelet count in
    # the danger zone or a critically low haemoglobin needs same-day review even
    # when the reported symptoms sound mild.
    labs = facts.lab_values or {}
    platelets = labs.get("platelets")
    if isinstance(platelets, dict) and platelets.get("value") is not None:
        pv = float(platelets["value"])
        if pv < 20000:
            emergency.append(f"Platelet count critically low ({int(pv):,}/cumm)")
        elif pv < 50000:
            urgent.append(f"Platelet count very low ({int(pv):,}/cumm)")
    hb = labs.get("hemoglobin")
    if isinstance(hb, dict) and hb.get("value") is not None:
        hv = float(hb["value"])
        if hv < 7:
            urgent.append(f"Haemoglobin severely low ({hv} g/dL)")
    glucose = labs.get("glucose")
    if isinstance(glucose, dict) and glucose.get("value") is not None:
        gv = float(glucose["value"])
        if gv >= 300 or gv < 54:
            urgent.append(f"Blood glucose dangerously out of range ({gv} mg/dL)")
    return emergency, urgent


def assess_severity(facts: ExtractedFacts, candidates: list[Candidate]) -> SeverityAssessment:
    factors: list[dict[str, Any]] = []
    score = 2.0

    top = candidates[0] if candidates else None
    if top:
        # A condition's severity profile describes what happens when that
        # condition is actually present. With thin evidence the leading
        # candidate may be a coincidence -- "fever and chills" alone matches
        # malaria, but also influenza -- so a *severe* profile is only granted
        # in proportion to how many symptoms actually matched. Without this,
        # one shared symptom inherits the severity of the worst disease that
        # happens to list it, and a moderate fever reads as an emergency with no
        # red flags present.
        #
        # The adjustment is deliberately one-sided: it can discount a severe
        # guess but never inflate a mild one. Raising "runny nose" toward
        # moderate because evidence is thin would trade a false alarm for
        # needless worry, and mild profiles are already correct when they match.
        base = float(top.severity_score)
        support = min(1.0, len(top.matched_symptoms) / _SEVERITY_SUPPORT_SYMPTOMS)
        if base > _NEUTRAL_SEVERITY:
            score = _NEUTRAL_SEVERITY + (base - _NEUTRAL_SEVERITY) * support
        else:
            score = base
        factors.append(
            {"factor": f"Most likely condition: {top.name}",
             "impact": f"base severity {top.severity_score}/10, "
                       f"weighted by {len(top.matched_symptoms)} matching symptom(s)"}
        )

    emergency_flags, urgent_flags = _collect_red_flags(facts)
    red_flags = emergency_flags + urgent_flags

    if facts.severity_self_reported is not None:
        self_score = float(facts.severity_self_reported)
        score = max(score, (score + self_score) / 2 + 1.0)
        factors.append(
            {"factor": f"Self-reported severity ({facts.severity_text})", "impact": f"{self_score:.0f}/10"}
        )

    # Duration vs the expected course of the leading condition.
    duration_assessment: dict[str, Any] = {}
    if top and facts.duration_days is not None:
        lo, hi = (top.typical_duration_days + [3, 14])[:2]
        d = facts.duration_days
        if d > hi * 1.5:
            score += 2.0
            status = "longer than expected"
            factors.append(
                {"factor": f"Symptoms lasting {d:.1f} days", "impact": f"longer than typical ({lo:.0f}-{hi:.0f} days)"}
            )
        elif d < lo * 0.5 and lo >= 1:
            score += 0.5
            status = "very recent onset"
        else:
            status = "within expected range"
        duration_assessment = {
            "reported_days": round(d, 2),
            "reported_text": facts.duration_text,
            "expected_days": [lo, hi],
            "status": status,
        }

    if facts.age is not None:
        if facts.age < 1:
            score += 3.0
            factors.append({"factor": "Infant under 1 year", "impact": "+high risk"})
        elif facts.age < 5:
            score += 1.5
            factors.append({"factor": "Young child (under 5)", "impact": "+moderate risk"})
        elif facts.age >= 75:
            score += 2.0
            factors.append({"factor": "Age 75 or older", "impact": "+high risk"})
        elif facts.age >= 65:
            score += 1.0
            factors.append({"factor": "Age 65 or older", "impact": "+moderate risk"})

    high_risk_conditions = {"diabetes", "heart failure", "kidney disease", "cancer",
                            "immunocompromised", "hiv", "copd", "liver disease"}
    for condition in facts.chronic_conditions:
        if condition in high_risk_conditions:
            score += 1.5
            factors.append({"factor": f"Chronic condition: {condition}", "impact": "+high risk"})
        elif condition != "pregnancy":
            score += 0.75
            factors.append({"factor": f"Chronic condition: {condition}", "impact": "+moderate risk"})

    if facts.is_pregnant:
        score += 1.5
        factors.append({"factor": "Pregnancy", "impact": "+high risk"})

    if facts.temperature_c is not None:
        t = facts.temperature_c
        if t >= 39.5:
            score += 2.0
            factors.append({"factor": f"Temperature {t} C", "impact": "+high fever"})
        elif t >= 38.5:
            score += 1.0
            factors.append({"factor": f"Temperature {t} C", "impact": "+fever"})

    # Red flags act as a floor, applied after every additive factor so a single
    # warning sign cannot be pushed over the next threshold by an unrelated
    # bonus. A fever of 39.5C should read urgent, not emergency.
    if emergency_flags:
        score = max(score, 9.0)
        factors.append({"factor": "Emergency red-flag symptoms present",
                        "impact": f"critical ({len(emergency_flags)} flag(s))"})
    elif urgent_flags:
        score = max(score, 7.0)
        factors.append({"factor": "Warning symptoms present needing prompt review",
                        "impact": f"urgent ({len(urgent_flags)} flag(s))"})

    score = int(round(max(1.0, min(10.0, score))))

    if score >= 9:
        label, triage = "critical", "emergency"
    elif score >= 7:
        label, triage = "severe", "urgent"
    elif score >= 5:
        label, triage = "moderate", "see_doctor"
    else:
        label, triage = "mild", "self_care"

    meta = TRIAGE_META[triage]
    return SeverityAssessment(
        score=score,
        label=label,
        triage=triage,
        triage_label=meta["label"],
        advice=meta["advice"],
        colour=meta["colour"],
        red_flags=red_flags,
        factors=factors,
        duration_assessment=duration_assessment,
    )
