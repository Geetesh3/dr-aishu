"""Tests for the Dr. Aishu data store, PII scrubbing and learning module."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app import learning, store


# ---- PII scrubbing -----------------------------------------------------------
@pytest.mark.parametrize(
    "text,must_vanish",
    [
        ("email me at priya.sharma@example.com about my fever", ["priya.sharma@example.com"]),
        ("call 9876543210 if it gets worse", ["9876543210"]),
        ("my name is Priya Sharma and I have a cough", ["Priya", "Sharma"]),
        ("Name: Anita Desai, fever for 2 days", ["Anita", "Desai"]),
        ("Patient: Ramesh Kumar with chest pain", ["Ramesh", "Kumar"]),
        ("I live at 42 MG Road, PIN 560001", ["42 MG Road", "560001"]),
        ("MRN: AB12345 admitted yesterday", ["AB12345"]),
        ("follow me @priya_s on instagram", ["@priya_s"]),
        ("see https://example.com/report for details", ["example.com"]),
    ],
)
def test_direct_identifiers_are_removed(text, must_vanish):
    scrubbed, count = store.scrub_pii(text)
    for secret in must_vanish:
        assert secret not in scrubbed, f"'{secret}' survived scrubbing: {scrubbed}"
    assert count >= 1


@pytest.mark.parametrize(
    "text",
    [
        "I have had a high fever and body aches for 3 days",
        "I am 30 years old with a sore throat",
        "no fever but severe cough and headache",
        "temperature 39.5C, bp 150/95, spo2 93%",
        "burning when I pee and going very often",
        "I am tired and have a headache since 2 days",
        "pain 8/10 in my lower back for a week",
    ],
)
def test_clinical_content_is_preserved(text):
    scrubbed, count = store.scrub_pii(text)
    assert scrubbed.strip(), "scrubbing must not empty a clinical description"
    # Clinical numbers and measurements must survive.
    for token in ("39.5", "150/95", "93", "8/10"):
        if token in text:
            assert token in scrubbed, f"clinical value {token} was destroyed"


def test_scrubbing_never_raises_on_empty_input():
    assert store.scrub_pii("") == ("", 0)
    assert store.scrub_pii(None) == ("", 0)


# ---- case storage ------------------------------------------------------------
def _result(text="high fever and body aches for 3 days", triage="urgent", top="influenza"):
    """A pipeline result object, without persisting it.

    These tests exercise save_case directly, so the fixture must not write a
    case of its own; `assess` is the pure half of the pipeline.
    """
    from app import pipeline
    return pipeline.assess(text=text)


def test_case_is_stored_when_consent_given():
    res = _result()
    out = store.save_case("high fever and body aches for 3 days", res, consent_share=True)
    assert out["stored_for_training"] is True
    assert out["case_id"]
    stats = store.stats()
    assert stats["cases_total"] == 1
    assert stats["cases_consented"] == 1


def test_case_text_is_dropped_without_consent():
    res = _result()
    out = store.save_case("high fever and body aches", res, consent_share=False)
    assert out["stored_for_training"] is False
    assert out["consented"] is False
    # The row exists for analytics, but the narrative must not be retained.
    examples = store.training_examples()
    assert examples, "structured symptoms still train the classifier"
    assert all(e["text"] == "" for e in examples)
    assert examples[0]["symptoms"]


def test_stored_text_is_scrubbed_of_pii():
    res = _result()
    store.save_case(
        "My name is Priya Sharma, phone 9876543210. High fever for 3 days.",
        res, consent_share=True,
    )
    examples = store.training_examples()
    assert len(examples) == 1
    text = examples[0]["text"]
    assert "Priya" not in text and "Sharma" not in text and "9876543210" not in text
    assert "fever" in text


def test_feedback_requires_a_known_case():
    with pytest.raises(KeyError):
        store.save_feedback("does-not-exist", is_correct=True)


def test_feedback_rating_range_is_enforced():
    res = _result()
    case_id = store.save_case("fever", res, consent_share=True)["case_id"]
    with pytest.raises(ValueError):
        store.save_feedback(case_id, rating=9)
    store.save_feedback(case_id, rating=4)
    assert store.stats()["feedback_total"] == 1


def test_correction_overrides_the_label_and_is_upweighted():
    res = _result()
    case_id = store.save_case("high fever with body aches", res, consent_share=True)["case_id"]
    store.save_feedback(case_id, correct_diagnosis="covid19")

    examples = store.training_examples()
    assert len(examples) == 1
    assert examples[0]["label"] == "covid19"
    assert examples[0]["origin"] == "feedback"
    assert examples[0]["weight"] == 2.0


def test_feedback_comment_is_scrubbed():
    res = _result()
    case_id = store.save_case("fever", res, consent_share=True)["case_id"]
    store.save_feedback(case_id, comment="my name is Priya Sharma and the dose was wrong")
    with store._connect() as conn:
        row = conn.execute("SELECT comment FROM feedback").fetchone()
    assert "Priya" not in row["comment"]
    assert "Sharma" not in row["comment"]


def test_right_to_erasure_removes_case_and_feedback():
    res = _result()
    case_id = store.save_case("fever and cough", res, consent_share=True)["case_id"]
    store.save_feedback(case_id, correct_diagnosis="covid19")

    assert store.delete_case(case_id) is True
    assert store.stats()["cases_total"] == 0
    assert store.stats()["feedback_total"] == 0
    assert store.delete_case(case_id) is False


def test_min_confidence_filter_excludes_weak_cases():
    low = {"extracted": {}, "severity": {}, "diagnoses": [{"id": "influenza", "confidence": 12.0}]}
    high = {"extracted": {}, "severity": {}, "diagnoses": [{"id": "influenza", "confidence": 88.0}]}
    store.save_case("vague feeling of being unwell", low, consent_share=True)
    store.save_case("high fever and severe body aches", high, consent_share=True)

    assert len(store.training_examples(min_confidence=0.0)) == 2
    assert len(store.training_examples(min_confidence=50.0)) == 1


# ---- learning ----------------------------------------------------------------
def test_document_builder_includes_symptom_tokens():
    doc = learning.build_document("I have a fever", ["fever", "cough"])
    assert "fever" in doc
    assert "sym_fever" in doc and "sym_cough" in doc


def test_training_and_prediction_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(learning, "TFIDF_PATH", tmp_path / "tfidf.joblib")
    monkeypatch.setattr(learning, "META_PATH", tmp_path / "meta.json")

    examples = [
        {"text": "high fever with chills and severe body aches and dry cough", "label": "influenza", "weight": 1.0},
        {"text": "fever with shivering and aching muscles and cough", "label": "influenza", "weight": 1.0},
        {"text": "burning when i pee and going very often", "label": "uti", "weight": 1.0},
        {"text": "painful urination with urgency and frequency", "label": "uti", "weight": 1.0},
        {"text": "throbbing one sided headache with nausea and light sensitivity", "label": "migraine", "weight": 1.0},
        {"text": "severe headache with vomiting and photophobia", "label": "migraine", "weight": 1.0},
    ]
    report = learning.train_tfidf(examples, min_examples_per_label=1)
    assert report.n_labels == 3
    assert learning.is_ready()

    preds = learning.predict("high fever with chills and body aches", top_k=3)
    assert preds
    assert preds[0]["label"] == "influenza"
    assert 0.0 <= preds[0]["probability"] <= 1.0


def test_training_rejects_too_little_data(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(learning, "TFIDF_PATH", tmp_path / "tfidf.joblib")
    monkeypatch.setattr(learning, "META_PATH", tmp_path / "meta.json")
    with pytest.raises(ValueError):
        learning.train_tfidf([{"text": "fever", "label": "influenza", "weight": 1.0}], min_examples_per_label=2)


def test_predict_returns_empty_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(learning, "TFIDF_PATH", tmp_path / "none.joblib")
    monkeypatch.setattr(learning, "TRANSFORMER_DIR", tmp_path / "none")
    monkeypatch.setattr(learning, "_state", {"backend": None, "model": None, "meta": {}})
    assert learning.load_model() is None
    assert learning.predict("fever and cough") == []


# ---- seed data generator -----------------------------------------------------
def test_seed_generator_produces_balanced_varied_data():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.seed_training_data import generate

    rows = generate(per_disease=5, seed=3)
    labels = {}
    for r in rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
        assert r["text"].strip()
        assert r["symptoms"], "each example must carry at least one symptom id"

    from app.kb import KB
    assert set(labels) == set(KB.diseases), "every condition should be represented"
    assert min(labels.values()) >= 5
    # Descriptions must vary, not be the same string repeated.
    assert len({r["text"] for r in rows}) == len(rows)


def test_seed_labels_all_exist_in_the_knowledge_base():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.kb import KB
    from scripts.seed_training_data import generate

    for r in generate(per_disease=3, seed=11):
        assert r["label"] in KB.diseases
        for sid in r["symptoms"]:
            assert sid in KB.symptoms
