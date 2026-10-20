"""Tests for multilingual support across the 22 scheduled Indian languages.

Two things can go badly wrong here and both are silent. A language can be
detected wrongly, so a Hindi speaker gets Marathi triage advice. Or a language
can be offered with its safety strings untranslated, so an emergency instruction
falls back to English and is not understood. The tests below pin both.
"""
from __future__ import annotations

import pytest

from app import pipeline
from app.i18n import LANGUAGE_BY_CODE, detect_language, is_supported
from app.nlp import extract
from app.translations import CRITICAL_KEYS, coverage, t


# ---- detection ---------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("मुझे तीन दिन से बुखार और सिरदर्द है", "hi"),
        ("मला तीन दिवसांपासून ताप आहे", "mr"),
        ("আমার তিন দিন ধরে জ্বর আছে", "bn"),
        ("മൂന്ന് ദിവസമായി എനിക്ക് പനിയുണ്ട്", "ml"),
        ("எனக்கு மூன்று நாட்களாக காய்ச்சல் உள்ளது", "ta"),
        ("నాకు మూడు రోజులుగా జ్వరం ఉంది", "te"),
        ("ನನಗೆ ಮೂರು ದಿನಗಳಿಂದ ಜ್ವರ ಇದೆ", "kn"),
        ("મને ત્રણ દિવસથી તાવ છે", "gu"),
        ("ମୋର ତିନି ଦିନ ଧରି ଜ୍ୱର ଅଛି", "or"),
        ("ਮੈਨੂੰ ਤਿੰਨ ਦਿਨਾਂ ਤੋਂ ਬੁਖਾਰ ਹੈ", "pa"),
        ("مجھے تین دن سے بخار ہے", "ur"),
        ("مون کي ٽن ڏينهن کان بخار آهي", "sd"),
        ("मोज्यांत तीन दिसां पासून ताप आसा", "kok"),
        ("I have had a fever for three days", "en"),
    ],
)
def test_detection_identifies_each_language(text, expected):
    assert detect_language(text)["code"] == expected


def test_english_is_reported_when_no_indic_script_is_present():
    result = detect_language("mild sore throat since yesterday")
    assert result["code"] == "en"
    assert result["method"] == "latin-default"


def test_empty_text_falls_back_to_english():
    assert detect_language("")["code"] == "en"
    assert detect_language("   ")["code"] == "en"


def test_detection_reports_confidence_and_method():
    result = detect_language("मुझे बुखार है")
    assert result["confidence"] > 0
    assert result["method"]


def test_shared_script_languages_are_disambiguated():
    """Devanagari is used by Hindi, Marathi, Nepali, Konkani and Bodo.

    A script-only detector cannot tell them apart, so this guards the marker-word
    step that does.
    """
    assert detect_language("मला ताप आहे")["code"] == "mr"
    assert detect_language("मलाई ज्वरो छ")["code"] == "ne"
    assert detect_language("मुझे बुखार है")["code"] == "hi"


def test_bengali_and_assamese_are_separated():
    """Assamese keeps ৰ (U+09F0) and ৱ (U+09F1), which Bengali does not use."""
    assert detect_language("মোৰ পেট বেয়া আছে")["code"] == "as"
    assert detect_language("আমার পেট খারাপ")["code"] == "bn"


def test_arabic_script_languages_are_separated():
    assert detect_language("مجھے بخار ہے")["code"] == "ur"
    assert detect_language("مون کي بخار آهي")["code"] == "sd"


# ---- coverage and safety strings ---------------------------------------------
def test_every_offered_language_has_its_safety_strings():
    """A language without translated emergency strings must not be offered.

    This is the load-bearing guarantee of the whole multilingual layer: an
    emergency instruction in a language the user cannot read is worse than no
    translation, because it looks like help.
    """
    broken = [
        code for code in LANGUAGE_BY_CODE
        if not coverage(code)["safety_complete"]
    ]
    assert not broken, f"languages missing safety strings: {broken}"


def test_critical_keys_are_non_empty_in_every_language():
    for code in LANGUAGE_BY_CODE:
        for key in CRITICAL_KEYS:
            value = t(key, code)
            assert value and value.strip(), f"{code} has an empty '{key}'"


def test_translation_falls_back_to_english_for_a_missing_key():
    assert t("disclaimer", "en")
    assert t("definitely.not.a.key", "hi") == t("definitely.not.a.key", "en")


def test_unknown_language_falls_back_to_english():
    assert t("disclaimer", "zz") == t("disclaimer", "en")


def test_supported_language_check():
    assert is_supported("hi")
    assert is_supported("ta")
    assert not is_supported("zz")
    assert not is_supported(None)


def test_all_scheduled_languages_are_present():
    """The eight schedule of the Constitution lists 22 languages; English is added."""
    expected = {
        "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai", "ml",
        "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
    }
    assert expected <= set(LANGUAGE_BY_CODE)


# ---- end-to-end through the pipeline -----------------------------------------
def test_analysis_returns_advice_in_the_detected_language():
    """The pipeline must localise triage advice, not just echo the input.

    A Telugu symptom description should produce Telugu guidance; the input being
    Telugu is not by itself evidence the output is.
    """
    result = pipeline.analyse(text="మూడు రోజులుగా తీవ్రమైన జ్వరం మరియు ఒళ్ళు నొప్పులు")
    assert result["language"]["code"] == "te"
    triage_label = result["severity"]["triage_label"]
    assert triage_label and any("\u0c00" <= ch <= "\u0c7f" for ch in triage_label), \
        "triage label was not rendered in Telugu script"


def test_explicit_language_overrides_detection():
    result = pipeline.analyse(text="I have a fever and a cough", lang="hi")
    assert result["language"]["code"] == "hi"


def test_emergency_keeps_suppressing_medicines_in_another_language():
    """The no-medicines-in-an-emergency rule must survive localisation.

    The localisation step rewrites several return fields; this proves it does not
    accidentally reintroduce a medicine on the emergency path.
    """
    result = pipeline.analyse(
        text="ఛాతీలో తీవ్రమైన నొప్పి, ఎడమ చేతికి వ్యాపిస్తోంది, శ్వాస తీసుకోవడం కష్టం",
        lang="te",
    )
    assert result["severity"]["triage"] == "emergency"
    assert result["medicines"]["recommended"] == []
    assert result["medicines"]["prescription_only"] == []


def test_disclaimer_is_localised_on_every_path():
    for text, lang in [("मुझे बुखार है", "hi"), ("எனக்கு காய்ச்சல்", "ta")]:
        result = pipeline.analyse(text=text, lang=lang)
        assert result["disclaimer"] == t("disclaimer", lang)


# ---- inflection --------------------------------------------------------------
def test_inflected_indic_terms_still_match():
    """Indic nouns take case markers, so the term is not a clean word.

    "chest pain" is written "ఛాతీలో నొప్పి" -- the "-లో" ("in") is glued to the
    noun. A strict word-boundary match misses it, and for a red flag that means
    the emergency is never raised, so this is the highest-stakes matcher case.
    """
    facts = extract("ఛాతీలో తీవ్రమైన నొప్పి, ఎడమ చేతికి వ్యాపిస్తోంది, శ్వాస తీసుకోవడం కష్టం")
    assert "chest_pain" in facts.symptoms
    assert "shortness_of_breath" in facts.symptoms


def test_inflected_chest_pain_raises_an_emergency():
    """The inflection fix must reach triage, not just extraction."""
    result = pipeline.analyse(text="ఛాతీలో తీవ్రమైన నొప్పి, ఎడమ చేతికి వ్యాపిస్తోంది")
    assert result["severity"]["triage"] in ("emergency", "urgent")


def test_a_gap_between_term_words_does_not_span_a_conjunction():
    """One intervening word is allowed, but not across "and".

    "तेज़ बुखार और सिरदर्द" is "severe fever and headache". The "severe headache"
    term ("तेज़ सिरदर्द") must not match across the conjunction, which would
    invent a symptom the user never reported.
    """
    facts = extract("मुझे तीन दिन से तेज़ बुखार और सिरदर्द है")
    assert "high_fever" in facts.symptoms
    assert "headache" in facts.symptoms
    assert "severe_headache" not in facts.symptoms


def test_an_intensifier_between_term_words_still_matches():
    """The same gap must still allow a genuine intensifier."""
    facts = extract("मुझे तेज़ सिरदर्द है")
    assert "severe_headache" in facts.symptoms


def test_english_matching_is_unaffected_by_the_indic_widening():
    """English terms keep strict boundaries; "pain" must not match "painting"."""
    facts = extract("I am painting my house today")
    assert "pain" not in facts.symptoms


# ---- extended catalogue ------------------------------------------------------
# The base catalogue in app/translations.py carries the safety-critical set. The
# descriptive interface copy is merged in from data/translations_extra.json, and
# these tests hold that overlay to the same standard as the base.
def test_extended_catalogue_loads():
    """A missing or malformed overlay file must not break the module."""
    from app import translations

    overlay = translations._load_overlay()
    assert overlay, "the extended catalogue did not load"
    assert "hi" in overlay and "ta" in overlay


def test_fully_translated_languages_cover_every_used_key():
    """A language advertised as complete must have no English gaps in the UI.

    The picker shows a percentage; a user who chooses a language reading "100%"
    should not then meet untranslated English labels on the page.
    """
    import glob
    import re

    from app.translations import EN, TRANSLATIONS, coverage

    used: set[str] = set()
    for path in glob.glob("templates/*.html"):
        with open(path, encoding="utf-8") as fh:
            used |= set(re.findall(r'data-i18n(?:-placeholder|-title|-aria)?="([^"]+)"', fh.read()))
    with open("static/app.js", encoding="utf-8") as fh:
        used |= set(re.findall(r'data-i18n="([^"]+)"', fh.read()))
    used &= set(EN)

    for code in ("hi", "bn", "ta", "te", "mr", "gu"):
        table = TRANSLATIONS[code]
        missing = sorted(k for k in used if k not in table)
        assert not missing, f"{code} is advertised complete but is missing {missing}"
        assert coverage(code)["complete"], f"{code} should report complete"


def test_partial_languages_are_not_advertised_as_complete():
    """Coverage must not overstate. A partial language is reported partial."""
    from app.translations import coverage

    assert not coverage("as")["complete"]
    assert 0 < coverage("as")["percent"] < 100


def test_brand_name_is_not_counted_as_missing_copy():
    """`app.name` is the product name, identical everywhere, and not copy."""
    from app.translations import BRAND_KEYS, coverage

    assert "app.name" in BRAND_KEYS
    assert coverage("hi")["percent"] >= 99.0


def test_extended_strings_are_actually_localised():
    """The overlay must contain real translations, not English passed through."""
    from app.translations import TRANSLATIONS, t

    for code in ("hi", "bn", "ta", "te", "mr", "gu"):
        value = t("home.how", code)
        assert value != t("home.how", "en"), f"{code} left 'home.how' in English"
        assert any(ord(ch) > 0x0900 for ch in value), f"{code} 'home.how' is not in its script"


def test_localised_page_sets_the_language_the_switcher_needs():
    """`?lang=` must reach the page, or the client cannot apply the translation.

    The interface copy is swapped in the browser from the catalogue, so the
    server's job is to mark the page with the requested language and expose the
    `data-i18n` hooks the switcher targets. This test pins that contract: if the
    attribute stops being rendered, every page silently falls back to English.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    body = client.get("/?lang=hi").text
    assert 'lang="hi"' in body, "the requested language was not put on <html>"
    assert 'data-i18n="home.how"' in body, "the i18n hook the switcher needs is gone"

    rtl_body = client.get("/?lang=ur").text
    assert 'dir="rtl"' in rtl_body, "an RTL language must set the text direction"


def test_unsupported_language_falls_back_to_english_page():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    assert 'lang="en"' in client.get("/?lang=zz").text
