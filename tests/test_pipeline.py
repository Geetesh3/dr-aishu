"""Tests for the Dr. Aishu analysis pipeline."""
from __future__ import annotations

import pytest

from app import pipeline
from app.analyzer import assess_severity, diagnose
from app.kb import KB
from app.nlp import extract
from app.prescription import check_interactions, select_medicines
from app.routines import build_routine


# ---- knowledge base integrity -----------------------------------------------
def test_knowledge_base_loads():
    assert len(KB.diseases) >= 40
    assert len(KB.symptoms) >= 80
    assert len(KB.medicines) >= 80
    assert len(KB.routines) >= 10


def test_no_dangling_medicine_references():
    missing = {
        (d.id, mid)
        for d in KB.diseases.values()
        for mid in d.medicines
        if mid not in KB.medicines
    }
    assert not missing, f"diseases reference unknown medicines: {missing}"


def test_no_dangling_symptom_references():
    missing = {
        (d.id, sid)
        for d in KB.diseases.values()
        for sid in list(d.symptoms) + d.key_symptoms
        if sid not in KB.symptoms
    }
    assert not missing, f"diseases reference unknown symptoms: {missing}"


def test_all_medicine_records_share_a_schema():
    keysets = {frozenset(m.keys()) for m in KB.medicines.values()}
    assert len(keysets) == 1, "medicine records must all carry the same fields"


def test_every_disease_has_a_routine_template():
    for d in KB.diseases.values():
        assert d.routine in KB.routines, f"{d.id} points at missing routine {d.routine}"


def test_red_flag_levels_are_graded():
    levels = {s.red_flag_level for s in KB.symptoms.values()}
    assert levels <= {"emergency", "urgent", None}
    assert "emergency" in levels and "urgent" in levels


# ---- NLP extraction ----------------------------------------------------------
def test_extracts_symptoms_and_duration():
    facts = extract("I have had a high fever and bad body aches for 3 days with a dry cough")
    assert "fever" in facts.symptoms
    assert "muscle_ache" in facts.symptoms
    assert "cough" in facts.symptoms
    assert facts.duration_days == pytest.approx(3.0)


def test_duration_units_normalise_to_days():
    assert extract("cough for 2 weeks").duration_days == pytest.approx(14.0)
    assert extract("rash for 3 months").duration_days == pytest.approx(90.0)
    assert extract("pain for 6 hours").duration_days == pytest.approx(0.25)


def test_duration_ranges_take_the_midpoint():
    assert extract("fever for 3-5 days").duration_days == pytest.approx(4.0)


def test_negation_is_detected():
    facts = extract("I have burning when I pee but no fever and no vomiting")
    assert "painful_urination" in facts.symptoms
    assert "fever" not in facts.symptoms
    assert "vomiting" not in facts.symptoms
    assert "fever" in facts.negated_symptoms


def test_negation_does_not_leak_across_a_contrast():
    facts = extract("no fever but severe sore throat and cough")
    assert "sore_throat" in facts.symptoms
    assert "cough" in facts.symptoms
    assert "fever" not in facts.symptoms


def test_vitals_and_demographics_are_parsed():
    facts = extract("age 45, temperature 39.2C, spo2 93%, bp 150/95, pulse 110")
    assert facts.age == 45
    assert facts.temperature_c == pytest.approx(39.2)
    assert facts.oxygen_saturation == 93
    assert facts.systolic_bp == 150
    assert facts.diastolic_bp == 95
    assert facts.pulse == 110


def test_fahrenheit_is_converted():
    facts = extract("fever with temperature 103F")
    assert facts.temperature_c == pytest.approx(39.4, abs=0.1)


def test_self_reported_severity():
    assert extract("severe unbearable headache").severity_self_reported == 10
    assert extract("mild sore throat").severity_self_reported == 2
    assert extract("pain 8/10").severity_self_reported == 8


def test_history_extraction():
    facts = extract("I am diabetic and allergic to penicillin, taking metformin")
    assert "diabetes" in facts.chronic_conditions
    assert any("penicillin" in a for a in facts.allergies)
    assert facts.current_medicines


def test_skin_allergic_is_not_a_drug_allergy():
    """"skin allergic" describes a symptom, not an allergen.

    Treating it as an allergy used to capture the rest of the sentence, so
    "thyroid problem, high bp, overweight" were recorded as allergens and were
    then used to filter medicines.
    """
    facts = extract("Headache, sneezing, skin allergic, thyroid problem, high bp, overweight")
    assert facts.allergies == []
    assert "hypertension" in facts.chronic_conditions
    assert "thyroid disease" in facts.chronic_conditions


def test_allergy_stops_at_the_next_clause():
    facts = extract("I am diabetic and allergic to penicillin, taking metformin")
    assert facts.allergies == ["penicillin"]


def test_bp_abbreviation_is_a_chronic_condition():
    assert "hypertension" in extract("high bp since 2019").chronic_conditions


def test_overlapping_symptom_matches_prefer_the_specific_term():
    facts = extract("I have a high grade fever of 103F")
    assert "high_fever" in facts.symptoms


# ---- diagnosis ---------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("high fever, severe body ache and chills for 3 days with dry cough", "influenza"),
        ("burning when I pee and going very often for 2 days", "uti"),
        ("watery diarrhoea and vomiting since yesterday with stomach cramps", "gastroenteritis"),
        ("throbbing one sided headache with nausea and light sensitivity", "migraine"),
        ("very thirsty all the time, peeing a lot at night, losing weight for 3 months", "type2_diabetes"),
        ("itchy red circular rash on my arm spreading over 2 weeks", "fungal_skin"),
        ("runny nose, sneezing and itchy watery eyes", "allergic_rhinitis"),
    ],
)
def test_leading_diagnosis_is_clinically_sensible(text, expected):
    result = pipeline.analyse(text=text)
    assert result["diagnoses"], f"no diagnosis produced for: {text}"
    assert result["diagnoses"][0]["id"] == expected


def test_no_symptoms_yields_no_diagnosis():
    result = pipeline.analyse(text="hello there")
    assert result["diagnoses"] == []


def test_confidences_are_ordered_and_bounded():
    result = pipeline.analyse(text="high fever and severe body ache and chills for 3 days with dry cough")
    confs = [d["confidence"] for d in result["diagnoses"]]
    assert confs == sorted(confs, reverse=True)
    assert all(0 <= c <= 100 for c in confs)


def test_clear_case_is_more_confident_than_vague_case():
    clear = pipeline.analyse(text="burning when I pee and going very often with lower abdominal pain")
    vague = pipeline.analyse(text="I feel a bit unwell and tired")
    assert clear["diagnoses"][0]["confidence"] > vague["diagnoses"][0]["confidence"]


# ---- severity / triage -------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected_triage",
    [
        ("runny nose and sneezing since yesterday", "self_care"),
        ("mild sore throat and slight cough for 2 days", "self_care"),
        ("crushing chest pain spreading to my left arm with sweating and breathlessness", "emergency"),
        ("fever with stiff neck and severe headache and confusion", "emergency"),
        ("sudden slurred speech and numbness on one side", "emergency"),
        ("diarrhoea and vomiting with no urine output and sunken eyes", "emergency"),
    ],
)
def test_triage_levels(text, expected_triage):
    result = pipeline.analyse(text=text)
    assert result["severity"]["triage"] == expected_triage


def test_emergency_suppresses_medicine_suggestions():
    result = pipeline.analyse(text="crushing chest pain spreading to left arm with sweating, sudden onset")
    assert result["severity"]["triage"] == "emergency"
    assert result["medicines"]["recommended"] == []
    assert result["medicines"]["prescription_only"] == []
    assert result["routine"]["name"] == "Emergency Response"


def test_urgent_flag_does_not_become_an_emergency():
    # Unintentional weight loss needs prompt review, not an ambulance.
    result = pipeline.analyse(text="very thirsty, peeing a lot, losing weight for 3 months, age 52")
    assert result["severity"]["triage"] == "urgent"
    assert "Unintentional weight loss" in result["severity"]["red_flags"]


def test_low_oxygen_is_an_emergency_flag():
    result = pipeline.analyse(text="cough and fever with spo2 91%")
    assert result["severity"]["triage"] == "emergency"
    assert any("oxygen" in f.lower() for f in result["severity"]["red_flags"])


def test_high_fever_adds_an_urgent_flag():
    result = pipeline.analyse(text="fever and cough, temperature 39.5C")
    assert any("temperature" in f.lower() for f in result["severity"]["red_flags"])


def test_thin_evidence_does_not_inherit_a_severe_condition():
    """A moderate fever must not read as an emergency just because the leading
    candidate happens to be a severe disease.

    Regression: severity was taken straight from the top condition's profile, so
    "fever and chills" -- which matches malaria on one shared symptom -- scored 9
    and triaged to emergency with no red flags present, contradicting the
    documented intent that a fever of 39.5C should read urgent, not emergency.
    """
    result = pipeline.analyse(text="fever 38.5C for 3 days with chills")
    assert result["severity"]["triage"] in {"see_doctor", "urgent"}
    assert result["severity"]["triage"] != "emergency"
    assert result["severity"]["red_flags"] == []


def test_severe_condition_still_escalates_when_well_supported():
    """The discount must not silence a genuinely severe picture."""
    result = pipeline.analyse(
        text="high fever 39.8C, shaking chills, severe headache, muscle aches and vomiting for 5 days"
    )
    assert result["severity"]["triage"] == "emergency"


def test_mild_condition_is_not_inflated_by_thin_evidence():
    """Discounting a severe guess must never raise a mild one."""
    result = pipeline.analyse(text="runny nose and sneezing since yesterday")
    assert result["severity"]["triage"] == "self_care"


def test_severity_factors_are_explained():
    result = pipeline.analyse(text="high fever 39.5C with cough, age 78")
    assert result["severity"]["factors"]
    assert any("age" in f["factor"].lower() for f in result["severity"]["factors"])


def test_infant_raises_severity():
    young = pipeline.analyse(text="3 month old baby with fever and diarrhoea")
    adult = pipeline.analyse(text="30 year old with fever and diarrhoea")
    assert young["severity"]["score"] >= adult["severity"]["score"]


# ---- medicine safety ---------------------------------------------------------
def test_allergy_filters_out_the_drug():
    result = pipeline.analyse(text="sore throat with fever and swollen glands, allergic to penicillin")
    names = [m["name"] for m in result["medicines"]["recommended"]]
    names += [m["name"] for m in result["medicines"]["prescription_only"]]
    assert not any("Amoxicillin" in n or "Penicillin" in n for n in names)
    assert any("penicillin" in e["reason"].lower() for e in result["medicines"]["excluded"])


def test_pregnancy_filters_out_unsafe_drugs():
    result = pipeline.analyse(text="burning when I pee for 2 days, I am 28 and pregnant")
    excluded = " ".join(e["name"] for e in result["medicines"]["excluded"])
    assert "Nitrofurantoin" in excluded or "Trimethoprim" in excluded


def test_age_floor_filters_out_paediatric_contraindications():
    result = pipeline.analyse(text="my child is 3 years old with diarrhoea and vomiting since yesterday")
    excluded = " ".join(e["name"] for e in result["medicines"]["excluded"])
    assert "Loperamide" in excluded


def test_chronic_condition_filters_are_applied():
    result = pipeline.analyse(text="joint pain and swelling with stiffness, I have kidney disease")
    excluded = " ".join(e["name"] for e in result["medicines"]["excluded"])
    assert "Ibuprofen" in excluded or "Diclofenac" in excluded


def test_prescription_drugs_are_separated_from_otc():
    result = pipeline.analyse(text="high fever and severe body aches and chills for 3 days with dry cough")
    assert all(m["otc"] for m in result["medicines"]["recommended"])
    assert all(not m["otc"] for m in result["medicines"]["prescription_only"])


def test_prescription_drugs_hidden_when_not_requested():
    result = pipeline.analyse(text="high fever and body aches and chills for 3 days", include_rx=False)
    assert result["medicines"]["prescription_only"] == []


def test_interaction_detection_finds_known_pair():
    found = check_interactions(["ibuprofen", "warfarin"])
    assert found
    assert "ibuprofen" in " ".join(found[0]["between"]).lower() or "Warfarin" in found[0]["between"]


def test_reported_current_medicine_reaches_the_safety_layer():
    """A medicine the user says they already take must be seen by the check.

    This is the path that stops a warfarin patient being handed an NSAID, so it
    is tested through the whole pipeline, not just the interaction helper.
    """
    result = pipeline.analyse(
        text="bad headache and joint pain for 3 days",
        current_medicines=["warfarin"],
    )
    excluded = {e["id"]: e["reason"] for e in result["medicines"]["excluded"]}
    assert "ibuprofen" in excluded, "NSAID not withheld for a warfarin patient"
    assert "warfarin" in excluded["ibuprofen"]


def test_reported_current_medicine_appears_in_the_extracted_history():
    result = pipeline.analyse(
        text="sore throat for 2 days",
        current_medicines=["warfarin"],
    )
    assert "warfarin" in result["extracted"]["history"]["current_medicines"]


def test_no_current_medicines_leaves_safety_unchanged():
    """Absent a reported medicine, the NSAID must not be withheld."""
    result = pipeline.analyse(text="bad headache and joint pain for 3 days")
    excluded_ids = {e["id"] for e in result["medicines"]["excluded"]}
    assert "ibuprofen" not in excluded_ids


def test_medicines_carry_dose_and_warnings():
    result = pipeline.analyse(text="high fever and severe body aches and chills for 3 days with dry cough")
    for med in result["medicines"]["recommended"]:
        assert med["dose"]
        assert med["frequency"]


# ---- timetable ---------------------------------------------------------------
def test_timetable_is_chronological_and_grouped():
    result = pipeline.analyse(text="high fever and severe body aches and chills for 3 days with dry cough")
    slots = result["medicines"]["timetable"]
    assert slots
    order = [s["slot"] for s in slots if s["slot"] != "as_needed"]
    expected = ["early_morning", "morning", "midday", "afternoon", "evening", "night"]
    assert order == [s for s in expected if s in order]
    for slot in slots:
        times = [e["time"] for e in slot["entries"] if e.get("time")]
        assert times == sorted(times)


def test_every_medicine_appears_in_the_timetable():
    result = pipeline.analyse(text="watery diarrhoea and vomiting since yesterday with cramps")
    scheduled = {
        e["medicine"]
        for slot in result["medicines"]["timetable"]
        for e in slot["entries"]
    }
    all_meds = {m["name"] for m in result["medicines"]["recommended"]}
    assert all_meds <= scheduled


# ---- routines ----------------------------------------------------------------
def test_routine_is_chronological():
    result = pipeline.analyse(text="high fever and body aches for 3 days with dry cough")
    times = [b["time"] for b in result["routine"]["blocks"] if b["time"] not in ("NOW",)]
    assert times == sorted(times)


def test_routine_includes_medicine_blocks():
    result = pipeline.analyse(text="high fever and body aches for 3 days with dry cough")
    med_blocks = [b for b in result["routine"]["blocks"] if b["category"] == "medication"]
    assert med_blocks


def test_hydration_target_rises_for_fluid_loss_conditions():
    gastro = pipeline.analyse(text="watery diarrhoea and vomiting since yesterday")
    cold = pipeline.analyse(text="runny nose and sneezing since yesterday")
    assert gastro["routine"]["hydration_target_l"] > cold["routine"]["hydration_target_l"]


def test_children_get_more_sleep_and_less_water():
    child = pipeline.analyse(text="my 4 year old has runny nose and sneezing")
    adult = pipeline.analyse(text="my age is 30 with runny nose and sneezing")
    assert child["routine"]["sleep_target_hours"] > adult["routine"]["sleep_target_hours"]
    assert child["routine"]["hydration_target_l"] < adult["routine"]["hydration_target_l"]


def test_urgent_triage_removes_activity_blocks():
    result = pipeline.analyse(text="crushing chest pain with sweating and breathlessness, sudden")
    categories = {b["category"] for b in result["routine"]["blocks"]}
    assert "activity" not in categories


def test_routine_names_match_the_leading_condition():
    result = pipeline.analyse(text="burning when I pee and going very often for 2 days")
    assert "Urinary" in result["routine"]["name"]


# ---- full response shape -----------------------------------------------------
def test_response_contains_all_sections():
    result = pipeline.analyse(text="high fever and body aches for 3 days with dry cough")
    for key in ("disclaimer", "extracted", "diagnoses", "severity", "medicines", "routine", "inputs"):
        assert key in result


def test_disclaimer_always_present():
    result = pipeline.analyse(text="mild sore throat")
    assert "not a medical diagnosis" in result["disclaimer"].lower()


def test_empty_input_is_handled():
    result = pipeline.analyse(text="")
    assert result["diagnoses"] == []
    assert result["routine"]["blocks"]


# ---- multi-problem cases -----------------------------------------------------
MULTI_PROBLEM = "Headache, sneezing, skin allergic, thyroid problem, high bp, overweight"


def test_chronic_conditions_reach_the_differential():
    result = pipeline.analyse(text=MULTI_PROBLEM)
    names = {d["name"] for d in result["diagnoses"]}
    assert any("Hypertension" in n for n in names)
    assert any("Hypothyroid" in n for n in names)


def test_declared_condition_is_flagged():
    result = pipeline.analyse(text=MULTI_PROBLEM)
    declared = {d["name"] for d in result["diagnoses"] if d["declared_condition"]}
    assert any("Hypertension" in n for n in declared)


def test_medicines_cover_every_declared_problem():
    """The acute complaint must not crowd the chronic ones out of the plan.

    "Headache, sneezing, ... high bp, ... thyroid problem" used to return
    antihistamines only, because the ranked differential put allergic rhinitis
    first and the medicine step looked at the leading diagnoses alone.
    """
    result = pipeline.analyse(text=MULTI_PROBLEM)
    plan = result["medicines"]
    for_conditions = {
        c
        for med in plan["recommended"] + plan["prescription_only"]
        for c in med.get("for_conditions", [])
    }
    assert any("Hypertension" in c for c in for_conditions)
    assert any("Hypothyroid" in c for c in for_conditions)


def test_unspecified_thyroid_does_not_mix_opposite_drugs():
    result = pipeline.analyse(text=MULTI_PROBLEM)
    names = {m["name"] for m in result["medicines"]["recommended"] + result["medicines"]["prescription_only"]}
    # Levothyroxine treats hypothyroidism, carbimazole the opposite. Both in one
    # plan would be a serious contradiction, so only the more likely is treated.
    assert not ({"Levothyroxine", "Carbimazole"} <= names)
