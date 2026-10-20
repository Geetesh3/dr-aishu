"""Turn laboratory findings into diagnostic evidence.

A lab report is numbers, not symptoms. "Haemoglobin 8.1 g/dL" and "Platelets
1.4 lakh" say nothing the symptom lexicon can match, so on their own they
produced no diagnoses at all. This module bridges that gap: it reads the values
parsed by :func:`app.nlp.extract_labs` and returns a multiplicative boost for
each condition the pattern supports, plus a plain-language list of the findings
for the result page.

The boosts are deliberately moderate. A laboratory value is supporting
evidence, not a diagnosis: it reorders conditions that the clinical picture
already raised, and can lift a condition into view when the numbers are
strongly specific (a positive dengue NS1, a platelet count of 40 thousand). It
never creates a recommendation on its own, and it cannot suppress the emergency
path, which is decided before medicines are chosen.
"""
from __future__ import annotations

from typing import Any

# Each rule reads one test and applies when the value falls in a band. `below`
# and `above` are inclusive-exclusive thresholds in the test's own unit, so the
# rule reads the same way as the report.
#
# (test_id, direction, threshold, condition, boost, plain-language reason)
_NUMERIC_RULES: list[tuple[str, str, float, str, float, str]] = [
    # --- anaemia ---------------------------------------------------------
    ("hemoglobin", "below", 11.0, "anemia", 2.6, "Haemoglobin is well below normal"),
    ("hemoglobin", "below", 12.0, "anemia", 1.8, "Haemoglobin is below normal"),
    ("hematocrit", "below", 33.0, "anemia", 1.6, "Haematocrit is low"),
    # --- dengue / vector-borne ------------------------------------------
    ("platelets", "below", 100000, "dengue", 2.4, "Platelet count is very low"),
    ("platelets", "below", 150000, "dengue", 1.8, "Platelet count is low"),
    ("platelets", "below", 150000, "typhoid", 1.2, "Platelet count is low"),
    ("wbc", "below", 4000, "dengue", 1.5, "White cell count is low, typical of a viral fever"),
    ("wbc", "below", 4000, "covid19", 1.3, "White cell count is low"),
    ("wbc", "below", 4000, "influenza", 1.2, "White cell count is low"),
    # --- bacterial infection --------------------------------------------
    ("wbc", "above", 11000, "pneumonia", 1.5, "White cell count is raised"),
    ("wbc", "above", 11000, "uti", 1.4, "White cell count is raised"),
    ("wbc", "above", 11000, "cellulitis", 1.4, "White cell count is raised"),
    ("wbc", "above", 11000, "tonsillitis", 1.3, "White cell count is raised"),
    ("wbc", "above", 11000, "typhoid", 1.3, "White cell count is raised"),
    ("wbc", "above", 15000, "pneumonia", 1.4, "White cell count is markedly raised"),
    # --- inflammation ----------------------------------------------------
    ("crp", "above", 5, "pneumonia", 1.3, "C-reactive protein is raised"),
    ("crp", "above", 5, "cellulitis", 1.3, "C-reactive protein is raised"),
    ("crp", "above", 5, "uti", 1.2, "C-reactive protein is raised"),
    ("crp", "above", 50, "pneumonia", 1.4, "C-reactive protein is markedly raised"),
    ("esr", "above", 30, "rheumatoid_arthritis", 1.3, "ESR is raised"),
    # --- metabolic -------------------------------------------------------
    ("glucose", "above", 200, "type2_diabetes", 2.6, "Blood glucose is in the diabetic range"),
    ("glucose", "above", 126, "type2_diabetes", 1.9, "Fasting glucose is above the diabetic threshold"),
    ("hba1c", "above", 6.5, "type2_diabetes", 2.8, "HbA1c is in the diabetic range"),
    ("hba1c", "above", 5.7, "type2_diabetes", 1.6, "HbA1c is in the pre-diabetic range"),
    ("tsh", "above", 4.0, "hypothyroidism", 2.6, "TSH is raised, suggesting an underactive thyroid"),
    ("tsh", "below", 0.4, "hyperthyroidism", 2.6, "TSH is suppressed, suggesting an overactive thyroid"),
    # --- gout / renal / liver -------------------------------------------
    ("uric_acid", "above", 7.2, "gout", 2.2, "Uric acid is raised"),
    ("creatinine", "above", 1.3, "kidney_stones", 1.2, "Creatinine is raised, so kidney function needs review"),
    ("bilirubin", "above", 1.2, "malaria", 1.2, "Bilirubin is raised, which can occur in malaria"),
]

# A named positive result is far more specific than a number, so these carry
# more weight. (condition, boost, reason)
_QUALITATIVE_RULES: dict[str, list[tuple[str, float, str]]] = {
    "dengue_ns1": [("dengue", 3.0, "Dengue NS1 antigen is positive")],
    "dengue_igm": [("dengue", 2.8, "Dengue antibodies are positive")],
    "dengue_word": [("dengue", 2.2, "The report names dengue")],
    "malaria_positive": [("malaria", 3.0, "Malaria parasites were detected")],
    "widal_positive": [("typhoid", 2.6, "The Widal test is positive")],
    "typhoid_word": [("typhoid", 2.0, "The report names typhoid")],
    "urine_infection": [("uti", 2.4, "Urine shows pus cells or leucocytes")],
    "covid_positive": [("covid19", 3.0, "COVID-19 test is positive")],
}


def _band(value: float, direction: str, threshold: float) -> bool:
    return value < threshold if direction == "below" else value > threshold


def lab_evidence(lab_values: dict[str, Any] | None) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Return (per-condition boost, findings) for a set of lab values.

    The boost for each condition is the product of every rule that fired, so two
    agreeing values (a low haemoglobin and a low haematocrit) reinforce each
    other while a single value stays modest. Findings are ordered strongest
    first, which is the order the result page shows them in.
    """
    if not lab_values:
        return {}, []

    boosts: dict[str, float] = {}
    findings: list[dict[str, Any]] = []

    for test_id, entry in lab_values.items():
        if test_id == "_qualitative" or not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if value is None:
            continue
        for rule_test, direction, threshold, condition, boost, reason in _NUMERIC_RULES:
            if rule_test != test_id or not _band(float(value), direction, threshold):
                continue
            boosts[condition] = boosts.get(condition, 1.0) * boost
            findings.append({
                "test": test_id,
                "label": entry.get("label", test_id),
                "value": value,
                "unit": entry.get("unit", ""),
                "flag": entry.get("flag", "abnormal"),
                "supports": condition,
                "reason": reason,
                "weight": "strong" if boost >= 2.0 else "supporting",
            })

    for item in lab_values.get("_qualitative", []) or []:
        for condition, boost, reason in _QUALITATIVE_RULES.get(item.get("id", ""), []):
            boosts[condition] = boosts.get(condition, 1.0) * boost
            findings.append({
                "test": item.get("id", ""),
                "label": item.get("note", ""),
                "value": None,
                "unit": "",
                "flag": "positive",
                "supports": condition,
                "reason": reason,
                "weight": "strong",
            })

    findings.sort(key=lambda f: (f["weight"] != "strong", f["test"]))
    return boosts, findings


def summarise(lab_values: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The values themselves, for display, without the evidence mapping."""
    if not lab_values:
        return []
    rows = [
        {"id": test_id, **entry}
        for test_id, entry in lab_values.items()
        if test_id != "_qualitative" and isinstance(entry, dict)
    ]
    rows.sort(key=lambda r: (r.get("flag") == "normal", r.get("label", "")))
    for item in lab_values.get("_qualitative", []) or []:
        rows.append({
            "id": item.get("id", ""), "label": item.get("note", ""),
            "value": None, "unit": "", "flag": "positive",
        })
    return rows
