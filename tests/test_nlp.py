"""Regression tests for clinical NLP extraction.

These cover the failure modes found while debugging: punctuation-sensitive
values (decimals, blood pressure, pain scales), plural synonyms, and negation
scope across conjunctions.
"""
from __future__ import annotations

import pytest

from app.nlp import extract, extract_duration, extract_severity, extract_vitals


# ---- vitals ------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("temperature 39.2C", {"temperature_c": 39.2}),
        ("temperature 39.2c", {"temperature_c": 39.2}),
        ("temp is 38.5 celsius", {"temperature_c": 38.5}),
        ("temperature 102F", {"temperature_c": 38.9}),
        ("fever of 101 f", {"temperature_c": 38.3}),
        ("spo2 93%", {"oxygen_saturation": 93}),
        ("oxygen saturation is 97", {"oxygen_saturation": 97}),
        ("bp 150/95", {"systolic_bp": 150, "diastolic_bp": 95}),
        ("blood pressure: 120/80", {"systolic_bp": 120, "diastolic_bp": 80}),
        ("pulse 110", {"pulse": 110}),
        ("heart rate is 72", {"pulse": 72}),
    ],
)
def test_vitals_are_parsed(text, expected):
    out = extract_vitals(text)
    for key, value in expected.items():
        assert out.get(key) == value, f"{key} from {text!r} -> {out}"


def test_vitals_ignore_implausible_values():
    assert "oxygen_saturation" not in extract_vitals("spo2 12%")
    assert "pulse" not in extract_vitals("pulse 900")
    assert "temperature_c" not in extract_vitals("temperature 9C")


def test_all_vitals_parsed_from_one_sentence():
    out = extract_vitals("age 45, temperature 39.2C, spo2 93%, bp 150/95, pulse 110")
    assert out == {
        "temperature_c": 39.2,
        "oxygen_saturation": 93,
        "systolic_bp": 150,
        "diastolic_bp": 95,
        "pulse": 110,
    }


# ---- pain scale --------------------------------------------------------------
def test_numeric_pain_scale_survives_normalisation():
    assert extract_severity("pain 8/10")[0] == 8
    assert extract_severity("severity of 7 out of 10")[0] == 7
    assert extract("the pain is 9/10").severity_self_reported == 9


# ---- plurals -----------------------------------------------------------------
@pytest.mark.parametrize(
    "text,symptom",
    [
        ("bad body aches", "muscle_ache"),
        ("aching muscles", "muscle_ache"),
        ("I have chills", "chills"),
        ("red spots on my skin", "rash"),
        ("my eyes are itchy", "itching"),
    ],
)
def test_plural_synonyms_match(text, symptom):
    assert symptom in extract(text).symptoms


# ---- negation ----------------------------------------------------------------
@pytest.mark.parametrize(
    "text,present,negated",
    [
        ("no fever but severe sore throat and cough", {"sore_throat", "cough"}, {"fever"}),
        ("no nausea or vomiting", set(), {"nausea", "vomiting"}),
        ("no cough, no fever, just a headache", {"headache"}, {"cough", "fever"}),
        ("denies chest pain and shortness of breath", set(), {"chest_pain", "shortness_of_breath"}),
        ("no headache or light sensitivity", set(), {"headache", "light_sensitivity"}),
    ],
)
def test_negation_scope(text, present, negated):
    facts = extract(text)
    assert set(facts.symptoms) == present, f"{text!r} -> {sorted(facts.symptoms)}"
    assert set(facts.negated_symptoms) == negated, f"{text!r} -> {facts.negated_symptoms}"


def test_negation_does_not_leak_across_and():
    """'no energy' negates fatigue only; the fever after 'and' still stands."""
    facts = extract("no energy and fever")
    assert "fever" in facts.symptoms
    assert "fever" not in facts.negated_symptoms


def test_leading_negation_keeps_its_own_symptom():
    facts = extract("no energy")
    assert "fatigue" in facts.symptoms


def test_question_negation_is_not_an_assertion():
    facts = extract("do you have any fever or cough?")
    assert "fever" not in facts.negated_symptoms


# ---- duration ----------------------------------------------------------------
@pytest.mark.parametrize(
    "text,days",
    [
        ("for 3 days", 3.0),
        ("for 2 weeks", 14.0),
        ("3-5 days", 4.0),
        ("since yesterday", 1.0),
        ("today", 0.5),
        ("for about 10 days", 10.0),
    ],
)
def test_duration_parsing(text, days):
    assert extract_duration(text)[0] == days


def test_duration_absent_returns_none():
    assert extract_duration("I have a headache")[0] is None
