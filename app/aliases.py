"""Hand-checked spelling repairs for clinical terms typed by patients.

Patients write "headach", "diabeties", "thyriod". The lexicon matches whole
words, so every one of those is silently dropped: the complaint never reaches
the diagnosis, and the medicine for it is never suggested. A person who types
"diabeties" is offered no diabetes medicine at all.

The repair is a whitelist rather than edit-distance matching, for the same
reason `scripts/build_dataset_workbook.py` keeps a whitelist for its HuggingFace
equivalences. Measured against the terms the matcher can actually reach, a
one-edit rule also "corrects" real English into unrelated clinical words:

    score -> sore     sound -> wound     rehydration -> dehydration
    acting -> aching  collapse -> ?      scalp -> scaly

Those are different complaints with different triage, so a silent substitution
could raise the wrong red flag. Only the pairs below are applied, and every
value is a term the symptom or chronic-condition lexicons already match.
"""

from __future__ import annotations

import re

from .kb import DATA_DIR

# Each key is a misspelling seen in real patient text; each value must be a word
# the symptom/chronic lexicons already match. Keys are matched as whole words
# against normalised text, so they cannot fire inside a longer word.
SPELLING_FIXES: dict[str, str] = {
    # --- the reported case and its close neighbours ---
    "headach": "headache",
    "headac": "headache",
    "diabeties": "diabetes",
    "diabetis": "diabetes",
    "diabities": "diabetes",
    "diabates": "diabetes",
    "thyriod": "thyroid",
    "thyrod": "thyroid",
    # --- symptom misspellings ---
    "fevr": "fever",
    "feverr": "fever",
    "sneesing": "sneezing",
    "sneazing": "sneezing",
    "snezing": "sneezing",
    "vomitting": "vomiting",
    "vomting": "vomiting",
    "vometing": "vomiting",
    "nausia": "nausea",
    "diarhea": "diarrhea",
    "diarreah": "diarrhea",
    "diahrea": "diarrhea",
    "constipaton": "constipation",
    "breathlesness": "breathlessness",
    "brethlessness": "breathlessness",
    "dizzyness": "dizziness",
    "dizzines": "dizziness",
    "giddines": "dizziness",
    "itchyness": "itching",
    "toothach": "toothache",
    "toothake": "toothache",
    "earach": "earache",
    "backach": "backache",
    "stomache": "stomach pain",
    "stomachache": "stomach pain",
    "tummyache": "stomach pain",
    "caugh": "cough",
    "palpitation": "palpitations",
    "palpitatons": "palpitations",
    "sweling": "swelling",
    # --- chronic-condition misspellings ---
    "hypertention": "hypertension",
    "hypertensoin": "hypertension",
    "hypertenshion": "hypertension",
    "asthama": "asthma",
    "astma": "asthma",
    "arthiritis": "arthralgia",
    "arthrities": "arthralgia",
    "arthritus": "arthralgia",
    "artritis": "arthralgia",
    "migrain": "migraine",
    "migrane": "migraine",
    "obesety": "obesity",
    "obeesity": "obesity",
    "overwieght": "overweight",
    "overweigth": "overweight",
    "tubercolosis": "tuberculosis",
    "epilepsiy": "epilepsy",
    "presure": "pressure",
    "suger": "sugar",
}

# A correction is skipped when the token is a word the app already ships in its
# own data, so a real word is never rewritten into a clinical term by accident.
_GUARD_FILES = (
    "diseases.json",
    "symptoms.json",
    "medicines.json",
    "routines.json",
    "symptom_terms_i18n.json",
    "translations_extra.json",
    "indian_brands.json",
)

_TOKEN_RE = re.compile(r"[a-z]+")

_COMBINED = re.compile(
    r"(?<![a-z])(?:"
    + "|".join(re.escape(t) for t in sorted(SPELLING_FIXES, key=len, reverse=True))
    + r")(?![a-z])"
)

_GUARD: frozenset[str] | None = None


def _guard() -> frozenset[str]:
    """English words already present in the shipped knowledge base.

    Built once on first use, from the small JSON files only. The 22 MB product
    catalogue is deliberately excluded: it is lazily loaded elsewhere and must
    not be parsed at import, and its brand names are not English vocabulary.
    """
    global _GUARD
    if _GUARD is None:
        words: set[str] = set()
        for name in _GUARD_FILES:
            path = DATA_DIR / name
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            words.update(_TOKEN_RE.findall(text.lower()))
        _GUARD = frozenset(words)
    return _GUARD


def repairs_in(text: str) -> dict[str, str]:
    """Misspellings present in `text` that would be repaired, as {bad: good}."""
    guard = _guard()
    return {
        m.group(0): SPELLING_FIXES[m.group(0)]
        for m in _COMBINED.finditer(text)
        if m.group(0) not in guard
    }


def apply_spelling_fixes(text: str) -> str:
    """Rewrite known clinical misspellings in an already-normalised string."""
    guard = _guard()

    def _sub(m: re.Match[str]) -> str:
        token = m.group(0)
        return token if token in guard else SPELLING_FIXES[token]

    return _COMBINED.sub(_sub, text)
