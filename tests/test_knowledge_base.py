"""Knowledge-base integrity tests.

The clinical pipeline is only as good as its data. These guard the specific
defects found during debugging: symptoms wired in at weight 0 (so they could
never contribute), and load-bearing discriminators missing from key_symptoms.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.kb import KB

DATA = Path(__file__).resolve().parent.parent / "app" / "data"


@pytest.fixture(scope="module")
def raw():
    return json.loads((DATA / "diseases.json").read_text())


def test_every_disease_has_a_usable_severity_score(raw):
    diseases = raw["diseases"] if isinstance(raw, dict) else raw
    for d in diseases:
        assert 1 <= d["severity_score"] <= 10, f"{d['id']} severity_score out of range"
        assert d["self_care"] is not None
        assert d["when_to_see_doctor"], f"{d['id']} has no escalation advice"


def test_no_symptom_is_wired_in_at_zero_weight(raw):
    """A zero weight means the symptom is listed but can never score."""
    diseases = raw["diseases"] if isinstance(raw, dict) else raw
    offenders = [
        f"{d['id']}.{sid}"
        for d in diseases
        for sid, w in d["symptoms"].items()
        if w == 0
    ]
    assert offenders == [], f"symptoms with zero weight: {offenders}"


def test_every_symptom_id_resolves(raw):
    diseases = raw["diseases"] if isinstance(raw, dict) else raw
    for d in diseases:
        for sid in list(d["symptoms"]) + list(d["key_symptoms"]):
            assert sid in KB.symptoms, f"{d['id']} references unknown symptom {sid}"


def test_key_symptoms_are_also_scored(raw):
    """A key symptom that is absent from `symptoms` can never be matched."""
    diseases = raw["diseases"] if isinstance(raw, dict) else raw
    for d in diseases:
        for key in d["key_symptoms"]:
            assert key in d["symptoms"], f"{d['id']} key symptom {key} is not scored"


def test_every_disease_has_key_symptoms(raw):
    diseases = raw["diseases"] if isinstance(raw, dict) else raw
    for d in diseases:
        assert d["key_symptoms"], f"{d['id']} has no key symptoms"


def test_every_medicine_reference_exists(raw):
    diseases = raw["diseases"] if isinstance(raw, dict) else raw
    for d in diseases:
        for mid in d["medicines"]:
            assert mid in KB.medicines, f"{d['id']} references unknown medicine {mid}"


def test_every_routine_reference_exists(raw):
    diseases = raw["diseases"] if isinstance(raw, dict) else raw
    for d in diseases:
        assert d["routine"] in KB.routines, f"{d['id']} references unknown routine {d['routine']}"


def test_emergency_triage_suppresses_all_medicines():
    """The safety invariant is enforced at analysis time, for any emergency."""
    from app import pipeline

    for text in [
        "crushing chest pain spreading to my left arm with sweating and breathlessness",
        "sudden facial droop and slurred speech and weakness on one side",
        "neck stiffness with fever and severe headache and confusion",
        "swollen lips and face with difficulty breathing after eating peanuts",
    ]:
        result = pipeline.analyse(text=text)
        assert result["severity"]["triage"] == "emergency", text
        assert result["medicines"]["recommended"] == [], text
        assert result["medicines"]["prescription_only"] == [], text


def test_critical_conditions_have_a_red_flag_key_symptom():
    """A condition scoring 10 must be reachable from a red-flag symptom, so the
    severity engine can actually escalate to emergency from the symptom picture."""
    for d in KB.diseases.values():
        if d.severity_score == 10:
            red_flag_keys = [
                k for k in d.key_symptoms
                if (s := KB.symptoms.get(k)) and s.red_flag
            ]
            assert red_flag_keys, (
                f"{d.id} scores 10 but none of its key symptoms is a red flag"
            )


@pytest.mark.parametrize(
    "disease_id,key_symptom",
    [
        ("stroke", "facial_droop"),
        ("anaphylaxis", "swollen_face_lips"),
        ("covid19", "loss_of_taste"),
        ("migraine", "light_sensitivity"),
        ("fungal_skin", "ring_shaped_rash"),
    ],
)
def test_discriminating_symptoms_are_key(disease_id, key_symptom):
    assert disease_id in KB.diseases
    assert key_symptom in KB.diseases[disease_id].key_symptoms


def test_anaphylaxis_is_present_and_critical():
    assert "anaphylaxis" in KB.diseases
    assert KB.diseases["anaphylaxis"].severity_score == 10


def test_high_risk_conditions_are_not_underweighted():
    """Each of these must be able to score from its defining symptom."""
    for did, sid in [
        ("covid19", "loss_of_taste"),
        ("pneumonia", "rapid_breathing"),
        ("stroke", "facial_droop"),
        ("burns", "blister"),
    ]:
        assert KB.diseases[did].symptoms.get(sid, 0) > 0


def test_red_flag_levels_are_valid():
    for s in KB.symptoms.values():
        if s.red_flag:
            assert s.red_flag_level in {"urgent", "emergency"}, (
                f"{s.id} is a red flag with level {s.red_flag_level}"
            )


# ---- script index (performance) ----------------------------------------------
# The scanner indexes patterns by script so a request only touches the buckets its
# text can contain. That is a pure speed change: it must never alter which
# symptoms are found. These tests pin the equivalence, because a dropped match
# here would silently miss a diagnosis.
def _full_scan(text: str) -> list[tuple[str, int, int]]:
    """The pre-index behaviour: every pattern against every text."""
    normalised = KB.normalise(text)
    hits: list[tuple[str, int, int]] = []
    for symptom in KB.symptoms.values():
        for pattern in symptom.patterns:
            for m in pattern.finditer(normalised):
                hits.append((symptom.id, m.start(), m.end()))
    return sorted(hits)


def _indexed_scan(text: str) -> list[tuple[str, int, int]]:
    normalised = KB.normalise(text)
    hits: list[tuple[str, int, int]] = []
    for script in KB.scripts_in(normalised):
        for symptom_id, pattern in KB.patterns_for(script):
            for m in pattern.finditer(normalised):
                hits.append((symptom_id, m.start(), m.end()))
    return sorted(hits)


@pytest.mark.parametrize(
    "text",
    [
        "I have had a high fever and body aches for 3 days, with a dry cough.",
        "मुझे तीन दिन से बुखार और सिरदर्द है",
        "எனக்கு மூன்று நாட்களாக காய்ச்சல் உள்ளது",
        "ఛాతీలో తీవ్రమైన నొప్పి, ఎడమ చేతికి వ్యాపిస్తోంది",
        "Chest pain spreading to left arm, shortness of breath, sweating",
        "মোৰ পেট বেয়া আছে",
        "مجھے تین دن سے بخار ہے",
        "pain in the big toe with uric acid 9.1",
        "मुझे बुखार है, BP 150/95, taking metformin",
        "No fever, but severe headache for 2 days",
        "mild sore throat and runny nose since yesterday",
    ],
)
def test_script_index_finds_exactly_what_a_full_scan_finds(text):
    """Indexing by script must not change the match set, only the work done."""
    assert _indexed_scan(text) == _full_scan(text)


def test_script_detection_labels_terms_correctly():
    assert KB.script_of("fever") == "latin"
    assert KB.script_of("सिरदर्द") == "devanagari"
    assert KB.script_of("காய்ச்சல்") == "tamil"
    assert KB.script_of("జ్వరం") == "telugu"


def test_latin_is_always_scanned_even_for_indic_text():
    """Clinical text mixes scripts, so Latin terms must never be skipped."""
    assert "latin" in KB.scripts_in("मुझे बुखार है, BP 150/95")
    assert KB.scripts_in("fever") == {"latin"}


def test_every_pattern_is_indexed_somewhere():
    """A pattern that is not in any bucket would never be matched again."""
    indexed = sum(len(v) for v in KB._terms_by_script.values())
    total = sum(len(s.synonyms) for s in KB.symptoms.values())
    assert indexed == total


def test_term_patterns_stay_aligned_with_patterns():
    """The script index is built from term_patterns, so the pairing must hold."""
    for symptom in KB.symptoms.values():
        assert [p for _, p in symptom.term_patterns] == symptom.patterns


def test_pattern_buckets_are_compiled_lazily():
    """The cold-start win depends on buckets being compiled on first use only."""
    from app.kb import KnowledgeBase

    fresh = KnowledgeBase()
    assert fresh._compiled_by_script == {}
    assert all(s._term_patterns is None for s in fresh.symptoms.values())

    fresh.patterns_for("latin")
    assert set(fresh._compiled_by_script) == {"latin"}
    assert "devanagari" not in fresh._compiled_by_script


def test_lazy_buckets_match_the_eager_index():
    """Compiling per script must yield the same matches as compiling everything."""
    from app.kb import KnowledgeBase

    fresh = KnowledgeBase()
    for text in (
        "high fever and body aches for 3 days with a dry cough",
        "मुझे बुखार और सिरदर्द है",
        "मुझे बुखार है, BP 150/95",
    ):
        normalised = fresh.normalise(text)
        eager: list[tuple[str, int, int]] = []
        for symptom in fresh.symptoms.values():
            for pattern in symptom.patterns:
                for m in pattern.finditer(normalised):
                    eager.append((symptom.id, m.start(), m.end()))
        lazy: list[tuple[str, int, int]] = []
        for script in fresh.scripts_in(normalised):
            for symptom_id, pattern in fresh.patterns_for(script):
                for m in pattern.finditer(normalised):
                    lazy.append((symptom_id, m.start(), m.end()))
        assert sorted(lazy) == sorted(eager), text
