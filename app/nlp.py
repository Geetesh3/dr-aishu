"""Natural-language extraction of clinical facts from free text.

Handles symptom mentions (with negation and uncertainty), duration phrases,
self-reported severity, vitals and demographics. Everything is rule-based so the
pipeline stays deterministic and auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .kb import KB

# ---- negation / uncertainty ---------------------------------------------------
NEGATION_CUES = [
    "no", "not", "without", "never", "denies", "deny", "absent", "negative for",
    "free of", "no longer", "doesn't have", "does not have", "don't have",
    "do not have", "haven't had", "have not had", "had no", "nil",
]
HEDGE_CUES = [
    "maybe", "perhaps", "possibly", "might have", "may have", "could be",
    "i think", "not sure", "unsure", "sometimes", "occasionally", "mild maybe",
    "suspected", "worried about", "concerned about",
]
# Symptoms inside these windows are treated as present regardless of earlier negation.
ASSERTION_BREAKERS = ["but", "however", "although", "except", "yet", "still", "then"]

# Cue patterns are compiled once. They are consulted for every candidate match,
# so building the regex inline turned each analysis into thousands of compiles.
_WORD_RE_CACHE: dict[str, re.Pattern] = {}


def _word_re(term: str) -> re.Pattern:
    """Whole-word matcher for a literal term, compiled once per term."""
    pattern = _WORD_RE_CACHE.get(term)
    if pattern is None:
        pattern = re.compile(rf"(?<![a-z]){re.escape(term)}(?![a-z])")
        _WORD_RE_CACHE[term] = pattern
    return pattern


_CUE_RE = {cue: _word_re(cue) for cue in set(NEGATION_CUES) | set(HEDGE_CUES)}

# ---- duration -----------------------------------------------------------------
DURATION_UNITS = {
    "minute": 1 / 1440, "minutes": 1 / 1440, "min": 1 / 1440, "mins": 1 / 1440,
    "hour": 1 / 24, "hours": 1 / 24, "hr": 1 / 24, "hrs": 1 / 24,
    "day": 1.0, "days": 1.0, "d": 1.0,
    "week": 7.0, "weeks": 7.0, "wk": 7.0, "wks": 7.0,
    "month": 30.0, "months": 30.0, "mth": 30.0, "mths": 30.0,
    "year": 365.0, "years": 365.0, "yr": 365.0, "yrs": 365.0,
}
WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
    "couple": 2, "few": 3, "several": 4,
}

SEVERITY_TERMS = {
    "mild": 2, "slight": 2, "minor": 2, "a little": 2, "low grade": 2,
    "moderate": 5, "medium": 5, "medium grade": 5,
    "severe": 8, "bad": 7, "serious": 8, "intense": 8, "excruciating": 10,
    "unbearable": 10, "worst": 10, "agonizing": 10, "agonising": 10,
    "extreme": 9, "very severe": 9, "terrible": 8, "horrible": 8,
}


@dataclass
class ExtractedFacts:
    symptoms: dict[str, dict[str, Any]] = field(default_factory=dict)
    modifiers: list[str] = field(default_factory=list)
    duration_days: float | None = None
    duration_text: str | None = None
    severity_self_reported: int | None = None
    severity_text: str | None = None
    age: float | None = None
    weight_kg: float | None = None
    temperature_c: float | None = None
    oxygen_saturation: float | None = None
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    pulse: int | None = None
    is_pregnant: bool = False
    chronic_conditions: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    current_medicines: list[str] = field(default_factory=list)
    negated_symptoms: list[str] = field(default_factory=list)
    lab_values: dict[str, Any] = field(default_factory=dict)


# ---- laboratory values ------------------------------------------------------
# Printed reports give numbers, not sentences: "Hb 9.2 g/dL", "Platelet Count
# 1.4 lakh/cumm", "TLC 14000". None of those match a symptom synonym, so before
# this table existed a lab report produced zero findings. Each entry names the
# unit conversions that appear on Indian reports and the healthy range, so a
# value can be interpreted without a second lookup.
#
# `scale` maps a written magnitude to a multiplier: a platelet count written
# "1.4 lakh" is 140000, and "140" is 140 thousand.
LAB_TESTS: dict[str, dict[str, Any]] = {
    "hemoglobin": {
        "label": "Haemoglobin", "unit": "g/dL", "low": 12.0, "high": 17.0,
        "patterns": [r"h(?:a?emo|b)globin", r"\bhb\b", r"\bhaem\b"],
    },
    "platelets": {
        "label": "Platelet count", "unit": "/cumm", "low": 150000, "high": 450000,
        "patterns": [r"platelet(?:s)?(?:\s*count)?", r"\bplt\b"],
    },
    "wbc": {
        "label": "Total white cell count", "unit": "/cumm", "low": 4000, "high": 11000,
        "patterns": [r"\bwbc\b", r"white\s*blood\s*cell(?:s)?", r"\btlc\b",
                     r"total\s*leucocyte\s*count", r"leucocyte\s*count"],
    },
    "glucose": {
        "label": "Blood glucose", "unit": "mg/dL", "low": 70, "high": 140,
        "patterns": [r"(?:fasting\s*)?(?:blood\s*)?(?:sugar|glucose)(?:\s*(?:fasting|pp|post\s*prandial))?"],
    },
    "hba1c": {
        "label": "HbA1c", "unit": "%", "low": 4.0, "high": 5.7,
        "patterns": [r"hba1c", r"glycated\s*h(?:a?emo|b)globin", r"glycosylated\s*h(?:a?emo|b)globin"],
    },
    "creatinine": {
        "label": "Serum creatinine", "unit": "mg/dL", "low": 0.6, "high": 1.3,
        "patterns": [r"creatinine", r"\bs\.?\s*creat\b"],
    },
    "tsh": {
        "label": "TSH", "unit": "uIU/mL", "low": 0.4, "high": 4.0,
        "patterns": [r"\btsh\b", r"thyroid\s*stimulating\s*hormone"],
    },
    "esr": {
        "label": "ESR", "unit": "mm/hr", "low": 0, "high": 20,
        "patterns": [r"\besr\b", r"erythrocyte\s*sedimentation"],
    },
    "crp": {
        "label": "C-reactive protein", "unit": "mg/L", "low": 0, "high": 5,
        "patterns": [r"\bcrp\b", r"c[\s\-]?reactive\s*protein"],
    },
    "uric_acid": {
        "label": "Uric acid", "unit": "mg/dL", "low": 3.5, "high": 7.2,
        "patterns": [r"uric\s*acid", r"\burate\b"],
    },
    "bilirubin": {
        "label": "Total bilirubin", "unit": "mg/dL", "low": 0.1, "high": 1.2,
        "patterns": [r"(?:total\s*)?bilirubin"],
    },
    "rbc": {
        "label": "Red cell count", "unit": "million/cumm", "low": 4.5, "high": 5.9,
        "patterns": [r"\brbc\b", r"red\s*blood\s*cell(?:s)?\s*count"],
    },
    "hematocrit": {
        "label": "Haematocrit", "unit": "%", "low": 36, "high": 50,
        "patterns": [r"h(?:a?emato|ema)crit", r"\bpcv\b", r"packed\s*cell\s*volume"],
    },
}

# Words that scale a written count. Indian reports commonly use lakh for a
# platelet count ("1.4 lakh") and thousand for a white cell count ("12 thousand").
LAB_SCALES: list[tuple[str, float]] = [
    (r"lakhs?|lacs?|\blakh\b", 100_000.0),
    (r"crores?|crore", 10_000_000.0),
    (r"thousands?|\bk\b", 1_000.0),
    (r"x\s*10\s*\^?\s*3|×\s*10\s*\^?\s*3|10\s*\^3", 1_000.0),
    (r"x\s*10\s*\^?\s*6|×\s*10\s*\^?\s*6", 1_000_000.0),
]

# Named test results that are a positive/negative or a pathogen name rather than
# a number. These are the strongest lab signals for the vector-borne fevers.
LAB_QUALITATIVE: list[tuple[str, str, str]] = [
    # (symptom-free flag id, regex, plain-language note)
    ("dengue_ns1", r"\bns\s*1\b[^\n]{0,20}?(?:positive|reactive|detected)",
     "Dengue NS1 antigen positive"),
    ("dengue_igm", r"dengue[^\n]{0,20}?(?:igm|igg)[^\n]{0,20}?(?:positive|reactive)",
     "Dengue antibody positive"),
    ("dengue_word", r"dengue\s*(?:fever|positive|suspected|reported)|(?:positive|suspected)\s*for\s*dengue",
     "Report mentions dengue"),
    ("malaria_positive", r"(?:malaria|mp|plasmodium)[^\n]{0,30}?(?:positive|reactive|detected|seen|parasite)",
     "Malaria parasite detected"),
    ("widal_positive", r"widal[^\n]{0,40}?(?:positive|reactive|significant\s*titre)",
     "Widal test positive"),
    ("typhoid_word", r"typhoid|enteric\s*fever", "Report mentions typhoid"),
    ("urine_infection", r"(?:urine|pus\s*cells?|leucocytes?)[^\n]{0,30}?(?:positive|plenty|many|numerous|\d{2,}\s*/\s*hpf)",
     "Urine shows pus cells"),
    ("covid_positive", r"(?:covid|sars[\s\-]?cov[\s\-]?2|rt[\s\-]?pcr)[^\n]{0,25}?(?:positive|detected|reactive)",
     "COVID-19 test positive"),
]


def _to_number(raw: str) -> float | None:
    """Parse a printed number, tolerating thousands separators."""
    raw = raw.strip().replace(",", "").replace(" ", "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# Units a report may print between the number and the next word.
_UNIT = r"(?:g/?dl|gm/?dl|mg/?dl|mmol/?l|miu/?l|uiu/?ml|u?iu/?ml|mm/?hr|%|/cumm|/cu\.?mm|/ul|/mm3|lakhs?|lacs?|thousands?|k|cells?/cumm)"

# A value is either label-then-number ("Haemoglobin 8.1") or number-then-label
# ("8.1 g/dL Haemoglobin"). Both are anchored to a single field: only
# whitespace, an optional unit and an optional separator may sit between the
# label and the number. Allowing arbitrary characters is what let "Hb 9.2 g/dL,
# ESR 45" read the haemoglobin as the ESR.
_LABEL_FIRST = r"(?:{p})\s*[:=\-]?\s*(\d[\d,]*(?:\.\d+)?)"
_NUMBER_FIRST = r"(\d[\d,]*(?:\.\d+)?)\s*{u}?\s*[:=\-]?\s*(?:{p})"


def _scale_after(text: str, end: int) -> tuple[float, bool]:
    """Magnitude word immediately after a number, e.g. "1.4 lakh".

    Only a short window is read: a "k" or "lakh" further down the line belongs to
    a different field.
    """
    window = text[end: end + 14]
    for pattern, factor in LAB_SCALES:
        if re.match(r"\s*" + pattern, window, re.I):
            return factor, True
    return 1.0, False


def extract_labs(text: str) -> dict[str, Any]:
    """Read numeric laboratory values from a report's OCR or typed text.

    Returns a mapping of test id to value, unit, healthy range and a low/normal/
    high flag, plus any qualitative positive results. Each test is matched in its
    own field, so a report listing several results on one line does not read one
    result as another's value.
    """
    if not text:
        return {}

    out: dict[str, Any] = {}
    for test_id, spec in LAB_TESTS.items():
        for pattern in spec["patterns"]:
            candidates: list[tuple[int, float, float]] = []
            for template in (_LABEL_FIRST, _NUMBER_FIRST):
                regex = template.format(p=pattern, u=_UNIT)
                for m in re.finditer(regex, text, re.I):
                    raw = m.group(1)
                    value = _to_number(raw)
                    if value is None:
                        continue
                    factor, scaled = _scale_after(text, m.end(1))
                    value *= factor
                    # A bare count under 1000 for platelets or white cells is
                    # written in thousands ("plt 140" is 140000, "tlc 12" is 12000).
                    if test_id in ("platelets", "wbc") and not scaled and value < 1000:
                        value *= 1000.0
                    candidates.append((m.start(), value, factor))
                if candidates:
                    break

            if not candidates:
                continue

            start, value, _factor = candidates[0]
            window = text[max(0, start - 25): start + 25]

            # Unit conversions that reports mix freely.
            if test_id == "glucose" and re.search(r"mmol", window, re.I):
                value *= 18.0
            if test_id == "hemoglobin" and value > 30:
                value /= 10.0  # written in g/L
            if test_id == "creatinine" and re.search(r"u?mol", window, re.I):
                value /= 88.4

            # Reject readings that cannot be a real result for that test, which is
            # how a page number or a date is kept from becoming a lab value.
            low, high = spec["low"], spec["high"]
            if value <= 0 or value > high * 20:
                continue

            if value < low:
                flag = "low"
            elif value > high:
                flag = "high"
            else:
                flag = "normal"

            out[test_id] = {
                "label": spec["label"],
                "value": round(value, 2),
                "unit": spec["unit"],
                "reference_low": low,
                "reference_high": high,
                "flag": flag,
            }
            break

    qualitative: list[dict[str, str]] = []
    for flag_id, pattern, note in LAB_QUALITATIVE:
        if re.search(pattern, text, re.I):
            qualitative.append({"id": flag_id, "note": note})
    if qualitative:
        out["_qualitative"] = qualitative

    return out


CHRONIC_TERMS = {
    "diabetes": "diabetes", "diabetic": "diabetes", "sugar problem": "diabetes",
    "hypertension": "hypertension", "high blood pressure": "hypertension", "bp problem": "hypertension",
    "asthma": "asthma", "copd": "copd", "heart disease": "heart disease",
    "heart failure": "heart failure", "kidney disease": "kidney disease",
    "renal failure": "kidney disease", "liver disease": "liver disease",
    "cancer": "cancer", "hiv": "hiv", "tb": "tuberculosis", "tuberculosis": "tuberculosis",
    "epilepsy": "epilepsy", "thyroid": "thyroid disease", "pregnant": "pregnancy",
    "pregnancy": "pregnancy", "immunocompromised": "immunocompromised",
}


def _find_window_negated(
    text: str,
    start: int,
    end: int,
    all_matches: list[dict[str, Any]] | None = None,
) -> tuple[bool, bool]:
    """Return (negated, hedged) for a match span, scanning back a bounded window.

    A negation cue is only applied when it belongs to this symptom. If another
    symptom sits between the cue and this one, the cue attaches to that nearer
    symptom instead, so "no energy and fever" keeps the fever.
    """
    window_start = max(0, start - 45)
    left = text[window_start:start]
    clause_start = 0
    for breaker in ASSERTION_BREAKERS:
        idx = left.rfind(" " + breaker + " ")
        if idx != -1:
            clause_start = max(clause_start, idx + len(breaker) + 2)
    scope_start = window_start + clause_start
    scope = text[scope_start:start]

    hedged = any(_CUE_RE[cue].search(scope) for cue in HEDGE_CUES)

    # Nearest negation cue to the symptom, searching backwards.
    cue_pos: int | None = None
    for cue in NEGATION_CUES:
        for m in _CUE_RE[cue].finditer(scope):
            pos = scope_start + m.start()
            if cue_pos is None or pos > cue_pos:
                cue_pos = pos
    if cue_pos is None:
        return False, hedged

    # A cue followed by a question is not an assertion of absence.
    if re.search(r"(do you|have you|is there|any)\s+[^?]*\?$", scope):
        return False, hedged

    # If another symptom lies between the cue and this match, decide whether the
    # cue still reaches us. "or" and comma lists distribute the negation
    # ("no nausea or vomiting" denies both), while "and" does not, because
    # "no energy and fever" only denies the energy. When unsure, keep the
    # symptom: a missed condition is more dangerous than an extra one.
    intervening = sorted(
        (o for o in (all_matches or [])
         if o["start"] != start and cue_pos <= o["start"] < start),
        key=lambda o: o["start"],
    )
    if intervening:
        prev_end = intervening[-1]["end"]
        connector = text[prev_end:start].strip(" ,")
        explicit_denial = re.search(r"\b(?:denies|deny|negative for|free of)\b", scope)
        # "or"/"nor" distribute the negation; so does an explicit denial verb
        # ("denies A and B"). A bare "and" does not.
        if connector in ("or", "and/or", "nor") or explicit_denial:
            return True, hedged
        return False, hedged
    return True, hedged


def extract_duration(text: str) -> tuple[float | None, str | None]:
    """Parse the first duration phrase, in days. Handles ranges (takes the midpoint).

    Age phrases are removed first. The pipeline prepends structured form values
    to the narrative, so "age 58 years" would otherwise match the bare
    "<n> years" pattern and be reported as a 21,170-day symptom duration, which
    then inflates severity and the dashboard averages.
    """
    text = re.sub(
        r"\bage\s+\d+(?:\.\d+)?\s*(?:years?|yrs?|y|months?|mths?)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    number = r"(\d+(?:\.\d+)?|[a-z]+)"
    unit = r"(minutes?|mins?|hours?|hrs?|days?|weeks?|wks?|months?|mths?|years?|yrs?)"

    # "for 3 days", "since 2 weeks", "3-4 days", "past 5 days", "last 10 days"
    pattern = re.compile(
        rf"(?:for|since|past|last|about|around|over|nearly|approximately|almost)?\s*"
        rf"{number}\s*(?:-|to|or)\s*{number}\s*{unit}",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if m:
        a, b, u = m.group(1), m.group(2), m.group(3).lower()
        fa = float(a) if a.replace(".", "", 1).isdigit() else WORD_NUMBERS.get(a.lower())
        fb = float(b) if b.replace(".", "", 1).isdigit() else WORD_NUMBERS.get(b.lower())
        if fa is not None and fb is not None:
            return (fa + fb) / 2 * DURATION_UNITS.get(u, 1.0), m.group(0).strip()

    pattern_single = re.compile(
        rf"(?:for|since|past|last|about|around|over|nearly|approximately|almost)\s*"
        rf"{number}\s*{unit}",
        re.IGNORECASE,
    )
    m = pattern_single.search(text)
    if m:
        a, u = m.group(1), m.group(2).lower()
        fa = float(a) if a.replace(".", "", 1).isdigit() else WORD_NUMBERS.get(a.lower())
        if fa is not None:
            return fa * DURATION_UNITS.get(u, 1.0), m.group(0).strip()

    # Bare "<n> <unit>" anywhere.
    pattern_bare = re.compile(rf"{number}\s*{unit}", re.IGNORECASE)
    m = pattern_bare.search(text)
    if m:
        a, u = m.group(1), m.group(2).lower()
        fa = float(a) if a.replace(".", "", 1).isdigit() else WORD_NUMBERS.get(a.lower())
        if fa is not None:
            return fa * DURATION_UNITS.get(u, 1.0), m.group(0).strip()

    # "today", "yesterday", "this morning", "overnight"
    relative = {"today": 0.5, "yesterday": 1.0, "this morning": 0.5, "tonight": 0.5,
                "overnight": 0.5, "this week": 4.0, "all week": 5.0, "since yesterday": 1.0}
    for phrase, days in relative.items():
        if _word_re(phrase).search(text):
            return days, phrase
    return None, None


def extract_severity(text: str) -> tuple[int | None, str | None]:
    best: tuple[int, str] | None = None
    for term, score in sorted(SEVERITY_TERMS.items(), key=lambda kv: -len(kv[0])):
        if _word_re(term).search(text):
            if best is None or score > best[0]:
                best = (score, term)
    # Explicit numeric pain scale: "pain 7/10", "severity of 8 out of 10"
    m = re.search(r"(\d{1,2})\s*(?:/|out of)\s*10", text)
    if m and 0 <= int(m.group(1)) <= 10:
        score = int(m.group(1))
        if best is None or score > best[0]:
            best = (score, f"{score}/10")
    return best if best else (None, None)


def extract_vitals(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"\b(3[5-9](?:\.\d)?|4[0-2](?:\.\d)?)\s*(?:deg(?:rees)?\s*)?(?:c|celsius|centigrade)\b", text, re.I)
    if m:
        out["temperature_c"] = float(m.group(1))
    m = re.search(r"\b(9[5-9](?:\.\d)?|10[0-9](?:\.\d)?|11[0-2](?:\.\d)?)\s*(?:deg(?:rees)?\s*)?(?:f|fahrenheit)\b", text, re.I)
    if m:
        out["temperature_c"] = round((float(m.group(1)) - 32) * 5 / 9, 1)
    m = re.search(r"\b(1[0-5]?\d)\s*(?:f\b|degrees f|fahrenheit)", text, re.I)
    if m and "temperature_c" not in out and int(m.group(1)) >= 95:
        out["temperature_c"] = round((float(m.group(1)) - 32) * 5 / 9, 1)
    m = re.search(r"\b(?:temp(?:erature)?|fever)\s*(?:is|of|:)?\s*(10[0-4]|9[6-9])\b", text, re.I)
    if m and "temperature_c" not in out:
        val = float(m.group(1))
        out["temperature_c"] = round((val - 32) * 5 / 9, 1)

    m = re.search(r"\b(?:spo2|saturation|sat|o2)\s*(?:is|of|:)?\s*(\d{2,3})\s*%?", text, re.I)
    if m and 50 <= int(m.group(1)) <= 100:
        out["oxygen_saturation"] = int(m.group(1))

    m = re.search(r"\b(?:bp|blood pressure)\s*(?:is|of|:)?\s*(\d{2,3})\s*/\s*(\d{2,3})", text, re.I)
    if m:
        out["systolic_bp"] = int(m.group(1))
        out["diastolic_bp"] = int(m.group(2))

    m = re.search(r"\b(?:pulse|heart rate|hr)\s*(?:is|of|:)?\s*(\d{2,3})", text, re.I)
    if m and 25 <= int(m.group(1)) <= 250:
        out["pulse"] = int(m.group(1))
    return out


def extract_demographics(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"\b(\d{1,3})\s*(?:years?|yrs?)\s*(?:old|of age)?\b", text)
    if m and 0 < int(m.group(1)) < 120:
        out["age"] = float(m.group(1))
    m = re.search(r"\bage[d]?\s*[:=]?\s*(\d{1,3})\b", text)
    if m and "age" not in out and 0 < int(m.group(1)) < 120:
        out["age"] = float(m.group(1))
    m = re.search(r"\b(\d{2,3}(?:\.\d)?)\s*(?:kg|kgs|kilograms?)\b", text)
    if m and 2 <= float(m.group(1)) <= 400:
        out["weight_kg"] = float(m.group(1))
    if re.search(r"\b(?:i am|i'm|is)\s*(?:a\s*)?(?:pregnan\w*|expecting)\b", text) or \
       re.search(r"\bpregnan\w*\b", text):
        out["is_pregnant"] = True
    return out


def extract_history(text: str) -> dict[str, list[str]]:
    conditions: list[str] = []
    for term, canon in CHRONIC_TERMS.items():
        if _word_re(term).search(text):
            if canon not in conditions:
                conditions.append(canon)
    allergies: list[str] = []
    for m in re.finditer(r"allerg\w*\s*(?:to|:)?\s*([a-z0-9 ,\-]{2,60})", text):
        chunk = m.group(1)
        chunk = re.split(r"\b(?:and|but|however|,)\b", chunk)[0].strip()
        if chunk:
            allergies.append(chunk.strip())
    medicines: list[str] = []
    for m in re.finditer(r"\b(?:taking|on|using|prescribed)\s+([a-z][a-z0-9 ,\-]{2,60})", text):
        chunk = m.group(1)
        chunk = re.split(r"\b(?:and|but|however|,)\b", chunk)[0].strip()
        if chunk and chunk not in medicines:
            medicines.append(chunk.strip())
    return {"chronic_conditions": conditions, "allergies": allergies, "current_medicines": medicines}


def extract_symptoms(text: str) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Match symptom synonyms, resolving overlaps and negations."""
    normalised = KB.normalise(text)
    candidates: list[dict[str, Any]] = []
    scripts_present = KB.scripts_in(normalised)

    # Only scan the buckets the text can actually contain. An English description
    # has no Devanagari in it, so the ~1150 Indic patterns are skipped instead of
    # being run and failing on every request. "latin" is always included because
    # clinical text mixes scripts ("BP 150/95" beside Hindi words).
    for script in scripts_present:
        for symptom_id, pattern in KB.patterns_for(script):
            for m in pattern.finditer(normalised):
                candidates.append(
                    {
                        "id": symptom_id,
                        "start": m.start(),
                        "end": m.end(),
                        "length": m.end() - m.start(),
                        "matched": m.group(0),
                    }
                )

    # Negation is resolved against the full match set, so a cue can be attributed
    # to the symptom it actually belongs to rather than the nearest one by luck.
    for c in candidates:
        negated, hedged = _find_window_negated(normalised, c["start"], c["end"], candidates)
        c["negated"] = negated
        c["hedged"] = hedged

    # Resolve overlaps: prefer longer matches; a later match wins ties.
    candidates.sort(key=lambda c: (-c["length"], c["start"]))
    chosen: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for c in candidates:
        if any(not (c["end"] <= s or c["start"] >= e) for s, e in occupied):
            continue
        chosen.append(c)
        occupied.append((c["start"], c["end"]))

    present: dict[str, dict[str, Any]] = {}
    negated_ids: list[str] = []
    for c in sorted(chosen, key=lambda c: c["start"]):
        if c["negated"]:
            if c["id"] not in negated_ids:
                negated_ids.append(c["id"])
            continue
        entry = present.setdefault(
            c["id"], {"id": c["id"], "mentions": [], "certainty": "reported"}
        )
        entry["mentions"].append(c["matched"])
        if c["hedged"]:
            entry["certainty"] = "uncertain"

    modifiers = []
    seen_modifiers: set[str] = set()
    for script in scripts_present:
        for mod_id, pattern in KB.modifier_patterns_for(script):
            if mod_id in seen_modifiers:
                continue
            if pattern.search(normalised):
                seen_modifiers.add(mod_id)
                modifiers.append(mod_id)
    return present, modifiers, negated_ids


def extract(text: str) -> ExtractedFacts:
    """Full extraction pass over a free-text clinical description."""
    normalised = KB.normalise(text)
    symptoms, modifiers, negated = extract_symptoms(normalised)
    duration_days, duration_text = extract_duration(normalised)
    severity, severity_text = extract_severity(normalised)
    facts = ExtractedFacts(
        symptoms=symptoms,
        modifiers=modifiers,
        duration_days=duration_days,
        duration_text=duration_text,
        severity_self_reported=severity,
        severity_text=severity_text,
        lab_values=extract_labs(text),
        negated_symptoms=negated,
        **extract_vitals(normalised),
        **extract_demographics(normalised),
        **extract_history(normalised),
    )
    return facts
