"""Regression tests for laboratory-report reading.

A printed report is numbers and abbreviations, not prose, and it lists several
results per line. These tests pin the two failure modes that matter: a value
must be read from its own field (so "Hb 9.2 g/dL, ESR 45" does not read the
haemoglobin as the ESR), and a written magnitude ("1.4 lakh", "40 thousand")
must be expanded.
"""
from __future__ import annotations

import pytest

from app.labs import lab_evidence, summarise
from app.nlp import extract, extract_labs


# ---- numeric parsing ---------------------------------------------------------
@pytest.mark.parametrize(
    "text,test_id,expected",
    [
        ("Haemoglobin 8.1 g/dL", "hemoglobin", 8.1),
        ("Hb 9.2 g/dL", "hemoglobin", 9.2),
        ("Hemoglobin: 13.5 g/dl", "hemoglobin", 13.5),
        ("Platelet Count 1.4 lakh/cumm", "platelets", 140000),
        ("Platelet count 40 thousand", "platelets", 40000),
        ("Platelet 250000", "platelets", 250000),
        ("TLC 3200 /cumm", "wbc", 3200),
        ("WBC 12,000 /cumm", "wbc", 12000),
        ("ESR 45 mm/hr", "esr", 45),
        ("CRP 62 mg/L", "crp", 62),
        ("Fasting Blood Sugar 186 mg/dL", "glucose", 186),
        ("HbA1c 8.2 %", "hba1c", 8.2),
        ("TSH 9.8 uIU/mL", "tsh", 9.8),
        ("Uric acid 9.1 mg/dL", "uric_acid", 9.1),
        ("Creatinine 1.1 mg/dL", "creatinine", 1.1),
    ],
)
def test_lab_values_are_read(text, test_id, expected):
    out = extract_labs(text)
    assert test_id in out, f"{test_id} not found in {out}"
    assert out[test_id]["value"] == pytest.approx(expected)


def test_values_do_not_bleed_across_fields():
    """The haemoglobin reading must not be taken from a later field."""
    out = extract_labs("Hb 9.2 g/dL, ESR 45 mm/hr, CRP 62 mg/L")
    assert out["hemoglobin"]["value"] == pytest.approx(9.2)
    assert out["esr"]["value"] == pytest.approx(45)
    assert out["crp"]["value"] == pytest.approx(62)


def test_full_report_line_is_parsed():
    text = ("Complete Blood Count: Haemoglobin 8.1 g/dL (13-17), "
            "TLC 3200 /cumm, Platelet Count 1.4 lakh/cumm")
    out = extract_labs(text)
    assert out["hemoglobin"]["value"] == pytest.approx(8.1)
    assert out["wbc"]["value"] == pytest.approx(3200)
    assert out["platelets"]["value"] == pytest.approx(140000)


def test_reference_range_is_not_read_as_the_value():
    """A result followed by its range keeps the result."""
    out = extract_labs("Haemoglobin 8.1 g/dL (13-17)")
    assert out["hemoglobin"]["value"] == pytest.approx(8.1)


def test_normal_values_are_flagged_normal():
    out = extract_labs("Hemoglobin 13.5 g/dl  TLC 8500 cells/cumm  Platelet 250000")
    assert out["hemoglobin"]["flag"] == "normal"
    assert out["wbc"]["flag"] == "normal"
    assert out["platelets"]["flag"] == "normal"


def test_low_and_high_flags():
    out = extract_labs("Hb 9.2 g/dL, TLC 14000 /cumm")
    assert out["hemoglobin"]["flag"] == "low"
    assert out["wbc"]["flag"] == "high"


def test_implausible_values_are_rejected():
    """A page number or a date must not become a lab value."""
    out = extract_labs("Page 12 of 30. Haemoglobin 13.0 g/dL")
    assert out["hemoglobin"]["value"] == pytest.approx(13.0)


# ---- qualitative results -----------------------------------------------------
def test_positive_dengue_ns1_is_detected():
    out = extract_labs("Dengue NS1 Antigen: POSITIVE")
    notes = [q["id"] for q in out.get("_qualitative", [])]
    assert "dengue_ns1" in notes


def test_no_labs_from_plain_prose():
    assert extract_labs("I have a headache and a mild fever") == {}


def test_empty_text_returns_nothing():
    assert extract_labs("") == {}


# ---- evidence mapping --------------------------------------------------------
def test_anemia_is_supported_by_low_haemoglobin():
    boosts, findings = lab_evidence(extract_labs("Haemoglobin 8.1 g/dL"))
    assert boosts.get("anemia", 0) > 1.0
    assert any(f["supports"] == "anemia" for f in findings)


def test_dengue_is_supported_by_positive_ns1():
    boosts, _ = lab_evidence(extract_labs("Dengue NS1 Antigen: POSITIVE"))
    assert boosts.get("dengue", 0) >= 3.0


def test_diabetes_is_supported_by_hba1c():
    boosts, _ = lab_evidence(extract_labs("HbA1c 8.2 %"))
    assert boosts.get("type2_diabetes", 0) > 1.5


def test_thyroid_is_supported_by_tsh():
    boosts, _ = lab_evidence(extract_labs("TSH 9.8 uIU/mL"))
    assert boosts.get("hypothyroidism", 0) > 1.5


def test_two_agreeing_values_reinforce():
    one = lab_evidence(extract_labs("Haemoglobin 11.5 g/dL"))[0].get("anemia", 1.0)
    two = lab_evidence(extract_labs("Haemoglobin 11.5 g/dL, Haematocrit 32%"))[0].get("anemia", 1.0)
    assert two > one


def test_no_evidence_without_values():
    assert lab_evidence({}) == ({}, [])
    assert lab_evidence(None) == ({}, [])


def test_summary_lists_abnormal_first():
    rows = summarise(extract_labs("Hemoglobin 13.5 g/dL, TLC 14000 /cumm"))
    assert rows[0]["flag"] != "normal" or rows[-1]["flag"] == "normal"


# ---- integration with the pipeline ------------------------------------------
def test_lab_report_produces_a_diagnosis():
    from app import pipeline

    result = pipeline.analyse(text="Dengue NS1 Antigen: POSITIVE", record=False)
    ids = [d["id"] for d in result["diagnoses"]]
    assert "dengue" in ids


def test_lab_report_surfaces_values_in_the_response():
    from app import pipeline

    result = pipeline.analyse(text="Haemoglobin 8.1 g/dL", record=False)
    labs = result["extracted"]["labs"]
    assert any(row["id"] == "hemoglobin" for row in labs)


def test_critical_platelet_count_is_a_red_flag():
    from app import pipeline

    result = pipeline.analyse(text="Platelet count 40 thousand", record=False)
    assert result["severity"]["red_flags"], "a low platelet count must raise a flag"
    assert result["severity"]["triage"] in ("urgent", "emergency")
