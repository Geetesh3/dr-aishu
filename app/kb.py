"""Knowledge base loader and indexing for the Health AI system.

Loads the JSON datasets (symptoms, diseases, medicines, routines) once at import
time and exposes indexed lookups plus a normalised symptom-matching lexicon.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DATA_DIR = Path(__file__).parent / "data"


def _load(name: str, default: Any = None) -> Any:
    path = DATA_DIR / name
    if not path.exists():
        if default is None:
            raise FileNotFoundError(path)
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Symptom:
    id: str
    label: str
    system: str
    synonyms: list[str]
    red_flag: bool
    red_flag_level: str | None
    weight: float
    # Compiled on first use, never at import. Compiling every synonym up front
    # costs ~1.6 s of process start-up, and a request only ever needs the
    # synonyms in one or two scripts, so most of that work is thrown away.
    _term_patterns: list[tuple[str, re.Pattern]] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def term_patterns(self) -> list[tuple[str, re.Pattern]]:
        """(term, pattern) pairs, built once from `synonyms` on first access."""
        if self._term_patterns is None:
            self._term_patterns = [(t, _compile(t)) for t in self.synonyms]
        return self._term_patterns

    @property
    def patterns(self) -> list[re.Pattern]:
        """Just the patterns, in the same order as `term_patterns`."""
        return [pattern for _, pattern in self.term_patterns]


@dataclass
class Disease:
    id: str
    name: str
    category: str
    icd10: str
    prevalence: float
    contagious: bool
    symptoms: dict[str, float]
    key_symptoms: list[str]
    typical_duration_days: list[float]
    severity_profile: str
    severity_score: int
    self_care: list[str]
    medicines: list[str]
    routine: str
    description: str
    when_to_see_doctor: list[str]


def _normalise(text: str) -> str:
    """Lowercase and collapse punctuation/whitespace so matching is stable.

    Periods and slashes are preserved because they carry meaning: decimals
    ("39.5c"), blood pressure ("150/95") and pain scales ("8/10") all depend on
    them. Everything else ASCII and non-alphanumeric becomes a space.

    Characters outside ASCII are preserved untouched. The Indic scripts encode
    vowels as combining marks, so stripping "non-alphanumeric" characters the way
    a Latin-only rule would deletes the middle of every Hindi or Tamil word and
    leaves nothing to match against.
    """
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^a-z0-9'\-./\s\u0080-\U0010FFFF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Characters that count as "inside a word" for boundary purposes: ASCII
# alphanumerics plus the letter and combining-mark ranges of the Indic scripts.
# A boundary rule of `[a-z0-9]` alone would treat every Devanagari syllable
# boundary as a word boundary, because matras fall outside that class.
_WORD_CHARS = (
    "a-z0-9"
    "\u0300-\u036f"   # combining diacritics
    "\u0900-\u097f"   # Devanagari
    "\u0980-\u09ff"   # Bengali / Assamese
    "\u0a00-\u0a7f"   # Gurmukhi
    "\u0a80-\u0aff"   # Gujarati
    "\u0b00-\u0b7f"   # Oriya
    "\u0b80-\u0bff"   # Tamil
    "\u0c00-\u0c7f"   # Telugu
    "\u0c80-\u0cff"   # Kannada
    "\u0d00-\u0d7f"   # Malayalam
    "\u0600-\u06ff"   # Arabic (Urdu, Sindhi, Kashmiri)
    "\uabc0-\uabff"   # Meetei Mayek
    "\u1c50-\u1c7f"   # Ol Chiki
)


# Each supported script, as (name, first codepoint, last codepoint). A term
# belongs to the first script its non-ASCII characters fall in; ASCII-only terms
# are "latin".
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("arabic", 0x0600, 0x06FF),
    ("meetei", 0xABC0, 0xABFF),
    ("olchiki", 0x1C50, 0x1C7F),
)


def script_of(term: str) -> str:
    """The script a term is written in, or "latin" for ASCII-only terms.

    Used to skip patterns that cannot match the input. A Latin query cannot
    contain a Devanagari syllable, so the ~1150 Indic patterns are dead weight on
    an English request and the scanner never needs to look at them.
    """
    for ch in term:
        cp = ord(ch)
        if cp < 0x80:
            continue
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                return name
        return "other"
    return "latin"


def scripts_in(text: str) -> set[str]:
    """Every script present in `text`, always including "latin".

    Latin is always included because clinical text mixes scripts: a Hindi
    description may still write "BP 150/95" or name a drug in Latin letters, so
    the Latin patterns must always run.
    """
    found = {"latin"}
    for ch in text:
        cp = ord(ch)
        if cp < 0x80:
            continue
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                found.add(name)
                break
        else:
            found.add("other")
    return found


# English articles and filler that patients include or omit freely inside a
# multi-word symptom phrase. Matching them as optional lets one curated synonym
# cover both phrasings.
_OPTIONAL_WORDS = {"a", "an", "the", "my", "of"}


# Conjunctions across the supported languages. Two symptom words separated by
# one of these are two separate complaints, not one phrase, so the gap in a
# multi-word Indic term must not span them.
_CONJUNCTION_PATTERN = "|".join([
    "and", "or", "with", "plus",
    "और", "व", "एवं",                      # Hindi, Marathi
    "আর", "এবং", "ও",                        # Bengali, Assamese
    "ਅਤੇ", "ਵੀ",                             # Punjabi
    "અને",                                   # Gujarati
    "ଏବଂ", "ଓ",                              # Odia
    "மற்றும்",                               # Tamil
    "మరియు",                                 # Telugu
    "ಮತ್ತು",                                 # Kannada
    "കൂടാതെ", "ഉം",                          # Malayalam
    "اور", "تو",                             # Urdu, Sindhi
    "आनी",                                   # Konkani
    "आरो",                                   # Bodo
    "ꯑꯃꯁꯨꯡ",                                # Meetei Mayek
    "ᱟᱟᱨᱛᱟ",                                 # Ol Chiki
])


def _compile(synonym: str) -> re.Pattern:
    """Word-boundary pattern; hyphens in the source match spaces too.

    A trailing plural is tolerated so "body aches" matches the synonym
    "body ache", and the lookarounds still block substring hits. The plural
    suffix is only added for Latin text, where it means something.

    Indic languages inflect the noun instead of using a preposition: "chest pain"
    is written "ఛాతీలో నొప్పి" -- literally "chest-in pain" -- with the case
    marker glued to the noun, and an intensifier such as "severe" often sits
    between the two words ("ఛాతీలో తీవ్రమైన నొప్పి"). A strict, contiguous boundary
    therefore misses the symptom entirely, and for a red flag like chest pain
    that means the emergency is never detected.

    So for non-Latin terms each word may take a short trailing suffix and the
    words may be separated by up to two intervening words of the same script.
    The leading boundary still holds, so this widens what a known term matches
    without letting it fire inside an unrelated word.
    """
    if not synonym:
        return re.compile(r"(?!)")  # matches nothing

    if synonym.isascii():
        # Patients write articles inconsistently: "pain in the big toe" and
        # "pain in big toe" are the same complaint. An article inside a phrase is
        # therefore optional, so one curated synonym covers both. It is matched as
        # a whole word followed by whitespace, so "a" cannot match the start of
        # an unrelated word.
        words = synonym.split()
        pieces: list[str] = []
        for i, word in enumerate(words):
            gap = "" if i == len(words) - 1 else r"[\s\-]+"
            if word.lower() in _OPTIONAL_WORDS:
                pieces.append(rf"(?:{re.escape(word)}{gap})?")
            else:
                pieces.append(re.escape(word) + gap)
        escaped = "".join(pieces)
        if escaped and escaped[-1].isalpha():
            escaped += r"s?"
        return re.compile(rf"(?<![{_WORD_CHARS}]){escaped}(?![{_WORD_CHARS}])")

    words = synonym.split()
    parts = [rf"{re.escape(w)}[{_WORD_CHARS}]{{0,3}}" for w in words]
    if len(parts) > 1:
        # At most one intervening word, and never a conjunction. "severe headache"
        # is written "तेज़ सिरदर्द"; without the conjunction guard the pattern also
        # spans "तेज़ बुखार और सिरदर्द" ("severe fever and headache") and reports a
        # severe headache the user never mentioned.
        gap = rf"(?:[\s\-]+(?!{_CONJUNCTION_PATTERN}(?![{_WORD_CHARS}]))[{_WORD_CHARS}]+)?[\s\-]+"
        body = gap.join(parts)
    else:
        body = parts[0]
    return re.compile(rf"(?<![{_WORD_CHARS}]){body}(?![{_WORD_CHARS}])")


def _order(
    exact: list[dict], mol_prefix: list[dict], name_word: list[dict],
    name_prefix: list[dict], contains: list[dict], limit: int,
) -> list[dict]:
    """Concatenate search tiers, sorting within each so the best rows lead.

    Single-ingredient products come before combinations, live before
    discontinued, cheapest first. Without this, searching "paracetamol" leads
    with a paracetamol-plus-decongestant cold remedy, which is not what someone
    asking for paracetamol wants.

    `name_word` sits above `name_prefix` because a whole-word hit is a much
    stronger signal than a partial one. For the query "dolo" both "Dolo 650"
    (whole word) and "Doloact" (partial) start with the query, so without this
    split the cheap diclofenac injection named "Doloact" outranks the
    paracetamol brand the user actually meant.
    """
    def rank(entry: dict) -> tuple:
        price = entry.get("price_inr")
        return (
            entry.get("combination", 0),
            entry.get("discontinued", 0),
            price if price is not None else 1e9,
        )

    out: list[dict] = []
    for tier in (exact, mol_prefix, name_word, name_prefix, contains):
        out.extend(sorted(tier, key=rank))
    return out[:limit]


class KnowledgeBase:
    def __init__(self) -> None:
        symptoms_raw = _load("symptoms.json")
        diseases_raw = _load("diseases.json")
        medicines_raw = _load("medicines.json")
        routines_raw = _load("routines.json")

        self.symptoms: dict[str, Symptom] = {}
        for s in symptoms_raw["symptoms"]:
            synonyms = sorted(set(s["synonyms"]), key=len, reverse=True)
            self.symptoms[s["id"]] = Symptom(
                id=s["id"],
                label=s["label"],
                system=s["system"],
                synonyms=synonyms,
                red_flag=bool(s["red_flag"]),
                red_flag_level=s.get("red_flag_level"),
                weight=float(s["weight"]),
            )

        # ---- multilingual symptom terms ------------------------------------
        # Extra synonyms per language, keyed by symptom id. They extend the same
        # matcher, so a Hindi or Tamil description produces the same facts as the
        # English equivalent. Terms are matched with the same word-boundary
        # patterns; the Indic scripts need no case folding, and _normalise keeps
        # their combining marks because it only strips ASCII punctuation.
        self.symptom_terms_i18n: dict[str, dict[str, list[str]]] = _load(
            "symptom_terms_i18n.json", default={}
        ) or {}
        self.symptom_terms_i18n.pop("_note", None)
        for lang, table in self.symptom_terms_i18n.items():
            for sid, terms in table.items():
                symptom = self.symptoms.get(sid)
                if not symptom:
                    continue
                extra = sorted({t for t in terms if t}, key=len, reverse=True)
                symptom.synonyms = sorted(set(symptom.synonyms + extra), key=len, reverse=True)

        self.modifiers: list[dict[str, Any]] = []
        for m in symptoms_raw.get("modifiers", []):
            self.modifiers.append(
                {
                    "id": m["id"],
                    "label": m["label"],
                    "weight": float(m["weight"]),
                    # (term, pattern) pairs, compiled on first access.
                    "term_patterns": None,
                    "synonyms": sorted(m["synonyms"], key=len, reverse=True),
                }
            )

        # ---- script index --------------------------------------------------
        # The scanner otherwise runs every pattern against every request, but a
        # term in one script can never match text written in another. Indexing by
        # script lets a request touch only the buckets its text can contain, which
        # on an English description skips the ~1150 Indic patterns entirely.
        #
        # The buckets hold terms, not compiled patterns: compiling all ~2000 of
        # them at start-up costs ~1.6 s, and a request only needs the one or two
        # scripts its own text contains. `patterns_for` compiles a bucket the
        # first time it is asked for, so an English request pays for the Latin
        # bucket alone and an Indic one for Latin plus its own script.
        self._terms_by_script: dict[str, list[tuple[str, str]]] = {}
        for symptom in self.symptoms.values():
            for term in symptom.synonyms:
                self._terms_by_script.setdefault(script_of(term), []).append(
                    (symptom.id, term)
                )
        self._modifier_terms_by_script: dict[str, list[tuple[str, str]]] = {}
        for mod in self.modifiers:
            for term in mod["synonyms"]:
                self._modifier_terms_by_script.setdefault(script_of(term), []).append(
                    (mod["id"], term)
                )
        self._compiled_by_script: dict[str, list[tuple[str, re.Pattern]]] = {}
        self._compiled_modifiers_by_script: dict[str, list[tuple[str, re.Pattern]]] = {}

        self.diseases: dict[str, Disease] = {}
        for d in diseases_raw["diseases"]:
            # Drop symptom keys with zero weight (placeholders for absent data).
            symptoms = {k: float(v) for k, v in d["symptoms"].items() if v}
            self.diseases[d["id"]] = Disease(
                id=d["id"],
                name=d["name"],
                category=d["category"],
                icd10=d.get("icd10", ""),
                prevalence=float(d.get("prevalence", 0.3)),
                contagious=bool(d.get("contagious", False)),
                symptoms=symptoms,
                key_symptoms=[k for k in d.get("key_symptoms", []) if k in symptoms],
                typical_duration_days=[float(x) for x in d.get("typical_duration_days", [3, 14])],
                severity_profile=d.get("severity_profile", "mild"),
                severity_score=int(d.get("severity_score", 3)),
                self_care=list(d.get("self_care", [])),
                medicines=list(d.get("medicines", [])),
                routine=d.get("routine", "general_wellness"),
                description=d.get("description", ""),
                when_to_see_doctor=list(d.get("when_to_see_doctor", [])),
            )

        self.medicines: dict[str, dict[str, Any]] = {
            m["id"]: m for m in medicines_raw["medicines"]
        }
        self.routines: dict[str, dict[str, Any]] = routines_raw["routines"]

        # Free-text drug name -> curated medicine id. Built from the id, the
        # generic name and the display name so that a patient saying "I take
        # warfarin" or "ibuprofen" resolves to the same record the suggestion
        # engine uses. Without this, an existing medication can never be
        # compared against a new suggestion.
        self.medicine_index: dict[str, str] = {}
        for med_id, med in self.medicines.items():
            for candidate in (med_id, med.get("generic"), med.get("name")):
                key = _normalise(str(candidate or ""))
                if key:
                    self.medicine_index.setdefault(key, med_id)
            # The display name often carries a parenthetical, e.g.
            # "Paracetamol (Acetaminophen)"; index the part before it too.
            name = str(med.get("name") or "")
            if "(" in name:
                stem = _normalise(name.split("(", 1)[0])
                if stem:
                    self.medicine_index.setdefault(stem, med_id)

        # ---- Indian brand layer -------------------------------------------
        # A pharmacy catalogue merged on top of the curated medicines. It adds
        # brand names, pack sizes and prices only; it carries no clinical data,
        # so nothing here may influence *which* medicine is chosen.
        brand_data = _load("indian_brands.json", default={}) or {}
        self.brand_meta: dict[str, Any] = {
            k: v for k, v in brand_data.items() if k not in ("brands",)
        }
        self.brands: dict[str, list[dict[str, Any]]] = brand_data.get("brands", {}) or {}
        self.brand_index: dict[str, str] = {}
        for med_id, entries in self.brands.items():
            for entry in entries:
                key = _normalise(entry.get("brand", ""))
                if key:
                    self.brand_index.setdefault(key, med_id)

        # Full pharmacy product index, loaded on first search (see property).
        self._product_index: dict[str, list] | None = None
        self._product_meta: dict[str, Any] | None = None

        # Symptom id -> diseases that reference it, with the weight used.
        self.symptom_index: dict[str, list[tuple[str, float]]] = {}
        for d in self.diseases.values():
            for sid, w in d.symptoms.items():
                self.symptom_index.setdefault(sid, []).append((d.id, w))

    # ---- convenience lookups -------------------------------------------------
    def symptom_label(self, sid: str) -> str:
        s = self.symptoms.get(sid)
        return s.label if s else sid

    def is_red_flag(self, sid: str) -> bool:
        s = self.symptoms.get(sid)
        return bool(s and s.red_flag)

    def normalise(self, text: str) -> str:
        return _normalise(text)

    def script_of(self, term: str) -> str:
        return script_of(term)

    def scripts_in(self, text: str) -> set[str]:
        return scripts_in(text)

    def all_symptom_terms(self) -> list[tuple[str, str]]:
        """(id, label) for every symptom, for autocomplete endpoints."""
        return sorted(((s.id, s.label) for s in self.symptoms.values()), key=lambda x: x[1])

    # ---- lazy pattern buckets ------------------------------------------------
    def prewarm(self, scripts: Iterable[str] | None = None) -> None:
        """Compile the given script buckets ahead of the first request.

        Compilation is the single largest start-up cost, so it is deferred to the
        first request that needs each bucket. On a server that first request would
        otherwise pay ~0.5 s while the user waits; doing it in a background thread
        at start-up moves that cost off the request path entirely.

        ``None`` means every script, which is what a long-running server wants:
        any language may arrive, and the whole job is under two seconds on a
        thread nobody is waiting on.
        """
        if scripts is None:
            scripts = list(self._terms_by_script)
        for script in scripts:
            self.patterns_for(script)
            self.modifier_patterns_for(script)

    def patterns_for(self, script: str) -> list[tuple[str, re.Pattern]]:
        """(symptom_id, pattern) pairs for one script, compiled on first use."""
        compiled = self._compiled_by_script.get(script)
        if compiled is None:
            compiled = [
                (sid, _compile(term)) for sid, term in self._terms_by_script.get(script, ())
            ]
            self._compiled_by_script[script] = compiled
        return compiled

    def modifier_patterns_for(self, script: str) -> list[tuple[str, re.Pattern]]:
        """(modifier_id, pattern) pairs for one script, compiled on first use."""
        compiled = self._compiled_modifiers_by_script.get(script)
        if compiled is None:
            compiled = [
                (mid, _compile(term))
                for mid, term in self._modifier_terms_by_script.get(script, ())
            ]
            self._compiled_modifiers_by_script[script] = compiled
        return compiled

    # ---- Indian brand layer --------------------------------------------------
    def brands_for(self, med_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Indian brands of the curated medicine `med_id`, best first."""
        return list(self.brands.get(med_id, []))[:limit]

    def resolve_brand(self, name: str) -> str | None:
        """Map a brand name a user typed to its curated medicine id.

        Exact match on the normalised brand, then a whole-word containment
        fallback so "Dolo 650" finds "Dolo 650 Tablet".
        """
        key = _normalise(name)
        if not key:
            return None
        if key in self.brand_index:
            return self.brand_index[key]
        pattern = re.compile(rf"(?<![{_WORD_CHARS}]){re.escape(key)}(?![{_WORD_CHARS}])")
        for brand_key, med_id in self.brand_index.items():
            if pattern.search(brand_key):
                return med_id
        return None

    # ---- full product index --------------------------------------------------
    @property
    def product_index(self) -> dict[str, list]:
        """Every product in the pharmacy catalogue, grouped by molecule.

        Loaded on first use. The file is ~22 MB, and most requests never search
        it, so parsing it at import time would slow every start for a feature
        only the search endpoint needs. Cached after the first read.
        """
        if self._product_index is None:
            raw = _load("medicine_index.json", default={}) or {}
            self._product_index = raw.get("molecules", {}) or {}
            self._product_meta = {k: v for k, v in raw.items() if k != "molecules"}
        return self._product_index

    @property
    def product_meta(self) -> dict[str, Any]:
        if self._product_meta is None:
            _ = self.product_index
        return self._product_meta or {}

    def search_products(
        self, query: str, limit: int = 40, include_combinations: bool = True,
    ) -> list[dict[str, Any]]:
        """Search the full catalogue by product name, molecule or manufacturer.

        Ranked in tiers so the obvious answer leads: exact molecule name, then
        molecules starting with the query, then product names starting with it,
        then anything containing it. A flat "contains" scan puts "pindolol"
        above "Dolo 650" for the query "dolo", which is backwards for someone
        typing a brand they saw on a box.

        Rows inside a molecule arrive pre-sorted best-first, so each tier can
        stop as soon as it is full rather than collecting and sorting thousands.
        """
        q = _normalise(query)
        if not q or len(q) < 2:
            return []

        columns = ["name", "manufacturer", "pack", "price_inr", "form", "discontinued", "combination"]
        exact: list[dict[str, Any]] = []
        mol_prefix: list[dict[str, Any]] = []
        name_word: list[dict[str, Any]] = []
        name_prefix: list[dict[str, Any]] = []
        contains: list[dict[str, Any]] = []

        def add(bucket: list[dict[str, Any]], molecule: str, row: list) -> None:
            bucket.append(self._product_row(molecule, row, columns))

        def is_whole_word(name: str, query: str) -> bool:
            """True when the query matches a complete word of the product name."""
            return any(word == query for word in name.split())

        # Pass 1: molecule names. Only 1,560 keys, so this is near-instant and
        # answers the common case -- someone typing "paracetamol" or "metformin"
        # -- without touching the 254k product rows.
        for molecule, rows in self.product_index.items():
            if q not in molecule:
                continue
            bucket = exact if molecule == q else mol_prefix if molecule.startswith(q) else contains
            for row in rows:
                if not include_combinations and row[6]:
                    continue
                add(bucket, molecule, row)
                if len(bucket) >= limit:
                    break

        if len(exact) >= limit or len(mol_prefix) >= limit:
            return _order(exact, mol_prefix, name_word, name_prefix, contains, limit)

        # Pass 2: brand and manufacturer names, only if pass 1 left room. This is
        # the full scan, so it runs last.
        for molecule, rows in self.product_index.items():
            if q in molecule:
                continue
            for row in rows:
                if not include_combinations and row[6]:
                    continue
                name = _normalise(row[0])
                if q in name:
                    if is_whole_word(name, q):
                        add(name_word, molecule, row)
                    elif name.startswith(q):
                        add(name_prefix, molecule, row)
                    else:
                        add(contains, molecule, row)
            if len(name_word) >= limit and len(name_prefix) >= limit and len(contains) >= limit:
                break

        return _order(exact, mol_prefix, name_word, name_prefix, contains, limit)

    def _product_row(self, molecule: str, row: list, columns: list[str]) -> dict[str, Any]:
        entry = dict(zip(columns, row))
        entry["molecule"] = molecule
        entry["curated_id"] = molecule if molecule in self.medicines else None
        return entry

    def resolve_medicine(self, name: str) -> str | None:
        """Map a free-text drug name to a curated medicine id, or None.

        Tries the exact generic/display/id lexicon first, then brand names, then
        a whole-word containment pass so "taking ibuprofen tablets" still
        resolves. Returns None for a drug the formulary does not carry (for
        example warfarin, which appears only as an interaction partner); callers
        must treat None as "unknown, compare by name instead".
        """
        key = _normalise(name)
        if not key:
            return None
        if key in self.medicine_index:
            return self.medicine_index[key]

        brand = self.resolve_brand(key)
        if brand:
            return brand

        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])")
        for lex_key, med_id in self.medicine_index.items():
            if pattern.search(lex_key) or lex_key in key:
                return med_id
        return None


KB = KnowledgeBase()
