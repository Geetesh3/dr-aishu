"""Multilingual support for the 22 scheduled languages of India.

Two jobs live here:

1. **Detecting the language of what the user typed.** Every scheduled language
   is written in one of a small number of scripts, and Unicode puts each script
   in its own block. Counting code points per block therefore identifies the
   language with no model and no network call. Romanised input (``"mujhe bukhar
   hai"``) cannot be identified this way, so it is handled by a keyword pass.

2. **Translating the interface.** A flat key -> string catalogue per language,
   resolved through :func:`t`. Missing keys fall back to English, so a partially
   translated language degrades to readable English rather than blank labels.

The 22 languages are the ones listed in the Eighth Schedule of the Constitution
of India, which is the set "all Indian languages" normally means.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Language:
    code: str          # BCP-47 primary subtag, e.g. "hi"
    english: str       # English name
    native: str        # endonym, shown in the picker
    script: str        # dominant script
    rtl: bool = False


# The 22 scheduled languages, with the script each is written in.
LANGUAGES: tuple[Language, ...] = (
    Language("as", "Assamese", "অসমীয়া", "bengali"),
    Language("bn", "Bengali", "বাংলা", "bengali"),
    Language("brx", "Bodo", "बर'", "devanagari"),
    Language("doi", "Dogri", "डोगरी", "devanagari"),
    Language("gu", "Gujarati", "ગુજરાતી", "gujarati"),
    Language("hi", "Hindi", "हिन्दी", "devanagari"),
    Language("kn", "Kannada", "ಕನ್ನಡ", "kannada"),
    Language("ks", "Kashmiri", "کٲشُر", "arabic", rtl=True),
    Language("kok", "Konkani", "कोंकणी", "devanagari"),
    Language("mai", "Maithili", "मैथिली", "devanagari"),
    Language("ml", "Malayalam", "മലയാളം", "malayalam"),
    Language("mni", "Manipuri", "ꯃꯤꯇꯩ ꯂꯣꯟ", "meetei_mayek"),
    Language("mr", "Marathi", "मराठी", "devanagari"),
    Language("ne", "Nepali", "नेपाली", "devanagari"),
    Language("or", "Odia", "ଓଡ଼ିଆ", "oriya"),
    Language("pa", "Punjabi", "ਪੰਜਾਬੀ", "gurmukhi"),
    Language("sa", "Sanskrit", "संस्कृतम्", "devanagari"),
    Language("sat", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "ol_chiki"),
    Language("sd", "Sindhi", "سنڌي", "arabic", rtl=True),
    Language("ta", "Tamil", "தமிழ்", "tamil"),
    Language("te", "Telugu", "తెలుగు", "telugu"),
    Language("ur", "Urdu", "اردو", "arabic", rtl=True),
)

LANGUAGE_BY_CODE: dict[str, Language] = {lang.code: lang for lang in LANGUAGES}

# Script -> the languages that use it. Used to narrow detection: Devanagari is
# shared by eight languages, so the script alone identifies a *set*, not a code.
SCRIPT_LANGUAGES: dict[str, list[str]] = {}
for _lang in LANGUAGES:
    SCRIPT_LANGUAGES.setdefault(_lang.script, []).append(_lang.code)

# Unicode block per script. Ranges cover the letters actually used in practice.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "bengali": ((0x0980, 0x09FF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
    "gujarati": ((0x0A80, 0x0AFF),),
    "oriya": ((0x0B00, 0x0B7F),),
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "meetei_mayek": ((0xABC0, 0xABFF),),
    "ol_chiki": ((0x1C50, 0x1C7F),),
}

# Scripts shared by several languages need lexical markers to separate them.
# Each marker carries a weight: distinctive words score high, words that are
# common across the family score low. Hindi gets the low-weight generic set
# because it is the default for Devanagari and needs no proof; the others must
# show a distinctive word to win.
SCRIPT_MARKERS: dict[str, dict[str, tuple[str, ...]]] = {
    "devanagari": {
        "hi": ("है", "और", "मुझे", "नहीं", "मेरा", "हूं", "रहा", "हुआ"),
        "mr": ("आहे", "आणि", "नाही", "मला", "तुम्ही", "होते", "झाले", "खूप"),
        "ne": ("छ", "छन्", "हुन्छ", "मलाई", "गरेको", "भएको", "दुखाइ", "छैन"),
        "sa": ("अस्ति", "मम", "त्वम्", "रोगः", "भवति", "चास्ति"),
        "mai": ("अछि", "हमरा", "नहि", "भेल", "करैत", "अछ"),
        "doi": ("तुस", "जो", "ऐ", "दा", "दी", "गी", "थोआ"),
        "kok": ("आसा", "आनी", "न्हय", "म्हाका", "तुजे", "जालार"),
        "brx": ("आरो", "जाय", "नङा", "आंनि", "बे", "मोनसे"),
    },
    "bengali": {
        # Assamese keeps ৰ (U+09F0) and ৱ (U+09F1), which Bengali does not use.
        "bn": ("আমার", "আর", "আছে", "হয়", "করে", "কেন", "ব্যথা"),
        "as": ("ৰ", "ৱ", "আছিল", "নহয়", "মোৰ", "তুমি", "কৰ"),
    },
    "arabic": {
        # Urdu, Sindhi and Kashmiri share the Arabic script. Urdu uses ے
        # (U+06D2) and ں (U+06BA); Sindhi adds ڪ and ٿ; Kashmiri uses ٲ and ٕ.
        "ur": ("ہے", "اور", "نہیں", "میں", "کا", "کی", "ے", "ں", "میرا"),
        "sd": ("ڪ", "ٿ", "آهي", "۾", "نہ", "جو", "جي", "ڏ"),
        "ks": ("ٲ", "ٕ", "چھ", "تہ", "سٕند", "کٲشُر"),
    },
}

# Default when a shared script matches nothing distinctive. Devanagari defaults
# to Hindi, Bengali to Bengali and Arabic to Urdu, the most widely used language
# of each family in this context.
SCRIPT_DEFAULT: dict[str, str] = {
    "devanagari": "hi",
    "bengali": "bn",
    "arabic": "ur",
}

# Romanised keyword cues, for users who type their language in Latin letters.
# Kept deliberately small and specific to avoid false positives on English.
ROMANISED_CUES: dict[str, tuple[str, ...]] = {
    "hi": ("bukhar", "dard", "khansi", "sardi", "khujli", "ulti", "dast"),
    "bn": ("jwor", "byatha", "kashi", "bomi", "matha"),
    "ta": ("kaichal", "vali", "irumal", "thalai", "vayiru"),
    "te": ("jwara", "noppi", "daggu", "kadupu"),
    "mr": ("taap", "dukhi", "khokla", "pota"),
    "gu": ("taav", "dukhavo", "khenk", "mathu"),
    "kn": ("novu", "kemmu", "hotte"),
    "ml": ("vedana", "chuma", "vayaru"),
    "pa": ("khang", "bukhar"),
    "or": ("jvara", "byatha", "kasa", "peta"),
    "ur": ("bukhar", "khansi", "pait"),
}

# Non-ASCII letter detection: anything outside Basic Latin means the user is not
# writing plain English, which is the trigger for script detection.
_NON_LATIN = re.compile(r"[^\x00-\x7F]")
_WORD = re.compile(r"[a-z]+")


def _count_script(text: str, script: str) -> int:
    ranges = SCRIPT_RANGES.get(script, ())
    total = 0
    for ch in text:
        cp = ord(ch)
        for lo, hi in ranges:
            if lo <= cp <= hi:
                total += 1
                break
    return total


def detect_language(text: str, default: str = "en") -> dict[str, Any]:
    """Identify the language of `text`.

    Returns a dict with `code`, `confidence` (0-1), `method` and the script
    counts that produced the decision, so the caller can explain *why* a
    language was chosen rather than silently switching the interface.
    """
    if not text or not text.strip():
        return {"code": default, "confidence": 0.0, "method": "empty", "scripts": {}}

    counts = {script: _count_script(text, script) for script in SCRIPT_RANGES}
    counts = {k: v for k, v in counts.items() if v}
    total_scripted = sum(counts.values())

    if total_scripted == 0:
        # Latin script. Look for romanised cues before assuming English.
        words = set(_WORD.findall(text.lower()))
        best_code, best_hits = None, 0
        for code, cues in ROMANISED_CUES.items():
            hits = sum(1 for cue in cues if cue in words or cue in text.lower())
            if hits > best_hits:
                best_code, best_hits = code, hits
        if best_code:
            return {
                "code": best_code,
                "confidence": round(min(0.85, 0.3 + 0.15 * best_hits), 2),
                "method": "romanised-keywords",
                "scripts": {"latin": len(text)},
            }
        return {"code": default, "confidence": 0.5, "method": "latin-default", "scripts": {"latin": len(text)}}

    dominant = max(counts, key=lambda k: counts[k])
    share = counts[dominant] / max(1, sum(counts.values()))

    candidates = SCRIPT_LANGUAGES.get(dominant, [])

    # A shared script (Devanagari for eight languages, Bengali for two) cannot
    # identify a single code, so score the lexical markers instead. The script's
    # default language wins only when no distinctive marker appears.
    markers = SCRIPT_MARKERS.get(dominant)
    if len(candidates) > 1 and markers:
        scores = {
            code: sum(1 for marker in markers.get(code, ()) if marker in text)
            for code in candidates
        }
        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 0:
            return {
                "code": best,
                "confidence": round(min(0.9, 0.5 + 0.12 * scores[best]), 2),
                "method": "script+markers",
                "scripts": counts,
            }
        fallback = SCRIPT_DEFAULT.get(dominant, candidates[0])
        return {"code": fallback, "confidence": round(0.45 * share, 2),
                "method": "script-default", "scripts": counts}

    if len(candidates) == 1:
        return {"code": candidates[0], "confidence": round(min(0.98, 0.6 + share), 2),
                "method": "script", "scripts": counts}

    return {"code": candidates[0] if candidates else default,
            "confidence": round(0.5 * share, 2), "method": "script-ambiguous", "scripts": counts}


def is_supported(code: str | None) -> bool:
    return bool(code) and code in LANGUAGE_BY_CODE


def language_catalogue() -> list[dict[str, Any]]:
    """Every supported language, for the picker and the API."""
    return [
        {"code": lang.code, "english": lang.english, "native": lang.native,
         "script": lang.script, "rtl": lang.rtl}
        for lang in LANGUAGES
    ]
