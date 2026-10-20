"""API tests for the Dr. Aishu FastAPI application."""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from app import store
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_index_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Dr. Aishu" in res.text


def test_health_reports_knowledge_base_and_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["assistant"] == "Dr. Aishu"
    kb = body["knowledge_base"]
    assert kb["diseases"] >= 40
    assert kb["symptoms"] >= 80
    assert kb["medicines"] >= 80
    assert "model" in body


def test_diseases_endpoint(client):
    body = client.get("/api/diseases").json()
    assert len(body["diseases"]) >= 40
    sample = body["diseases"][0]
    for key in ("id", "name", "category", "symptoms", "self_care", "when_to_see_doctor"):
        assert key in sample


def test_symptoms_endpoint_exposes_graded_red_flags(client):
    body = client.get("/api/symptoms").json()
    levels = {s["red_flag_level"] for s in body["symptoms"] if s["red_flag"]}
    assert levels <= {"urgent", "emergency"}


def test_medicines_endpoint(client):
    body = client.get("/api/medicines").json()
    assert len(body["medicines"]) >= 80


def test_analyze_returns_full_payload(client):
    res = client.post("/api/analyze", json={"text": "high fever and body aches for 3 days with dry cough"})
    assert res.status_code == 200
    body = res.json()
    assert body["diagnoses"]
    assert body["severity"]["triage"] in {"self_care", "see_doctor", "urgent", "emergency"}
    assert body["routine"]["blocks"]
    assert "case" in body and body["case"]["case_id"]


def test_analyze_rejects_empty_input(client):
    assert client.post("/api/analyze", json={"text": "   "}).status_code == 400


def test_analyze_validates_ranges(client):
    assert client.post("/api/analyze", json={"text": "fever", "age": 500}).status_code == 422
    assert client.post("/api/analyze", json={"text": "fever", "top_n": 99}).status_code == 422


def test_consent_flag_controls_training_storage(client):
    client.post("/api/analyze", json={"text": "fever and cough for 2 days", "consent_share": True})
    client.post("/api/analyze", json={"text": "sore throat for 2 days", "consent_share": False})
    stats = client.get("/api/data/stats").json()
    assert stats["cases_total"] == 2
    assert stats["cases_consented"] == 1


def test_export_returns_scrubbed_text_and_symptom_only_rows(client, operator_headers):
    """Export is an operator function, so it needs an operator key.

    A non-consented case still contributes a labelled row, but only through its
    canonical symptom ids -- its free text must never appear.
    """
    headers = operator_headers
    client.post("/api/analyze", json={
        "text": "My name is Priya Sharma and my phone is 9876543210. High fever for 3 days.",
        "consent_share": True,
    })
    client.post("/api/analyze", json={"text": "mild rash on arm", "consent_share": False})

    body = client.get("/api/export/training-data", headers=headers).json()
    assert body["count"] == 2
    texts = [e["text"] for e in body["examples"]]
    assert "Priya" not in " ".join(texts) and "9876543210" not in " ".join(texts)
    assert "rash" not in " ".join(texts)
    # The consented row keeps its scrubbed narrative.
    assert any("fever" in t for t in texts)


def test_anonymous_demo_key_cannot_export(client):
    """The browser demo key must not be able to exfiltrate stored cases."""
    client.post("/api/analyze", json={"text": "fever for 3 days", "consent_share": True})
    assert client.get("/api/export/training-data").status_code == 403


def test_feedback_flow(client, operator_headers):
    case_id = client.post("/api/analyze", json={
        "text": "high fever and body aches for 3 days", "consent_share": True,
    }).json()["case"]["case_id"]

    res = client.post("/api/feedback", json={
        "case_id": case_id, "correct_diagnosis": "covid19", "rating": 5, "comment": "lost my smell too",
    })
    assert res.status_code == 200

    examples = client.get("/api/export/training-data", headers=operator_headers).json()["examples"]
    assert examples[0]["label"] == "covid19"
    assert examples[0]["weight"] == 2.0


def test_feedback_rejects_unknown_case(client):
    res = client.post("/api/feedback", json={"case_id": "nope", "rating": 3})
    assert res.status_code == 404


def test_feedback_rejects_bad_rating(client):
    case_id = client.post("/api/analyze", json={"text": "fever for 2 days"}).json()["case"]["case_id"]
    assert client.post("/api/feedback", json={"case_id": case_id, "rating": 11}).status_code == 422


def test_model_status(client):
    body = client.get("/api/model/status").json()
    assert "model" in body and "data" in body
    assert "available_backends" in body["model"]


def test_case_deletion(client):
    case_id = client.post("/api/analyze", json={
        "text": "fever and cough for 2 days", "consent_share": True,
    }).json()["case"]["case_id"]

    assert client.delete(f"/api/data/case/{case_id}").status_code == 200
    assert client.delete(f"/api/data/case/{case_id}").status_code == 404
    assert client.get("/api/data/stats").json()["cases_total"] == 0


def _png_bytes() -> bytes:
    """A small valid PNG, generated without external fixtures."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (48, 48), (200, 60, 60)).save(buf, format="PNG")
    return buf.getvalue()


def test_image_analysis_endpoint(client):
    res = client.post(
        "/api/analyze-image",
        files={"image": ("rash.png", _png_bytes(), "image/png")},
        data={"text": "itchy red rash on my arm for a week", "consent_share": "false"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["image_analysis"]["ok"] is True
    assert "visual_features" in body["image_analysis"]


def test_image_endpoint_rejects_wrong_type(client):
    res = client.post(
        "/api/analyze-image",
        files={"image": ("notes.txt", b"just text", "text/plain")},
    )
    assert res.status_code == 415


def test_image_endpoint_rejects_empty_file(client):
    res = client.post(
        "/api/analyze-image",
        files={"image": ("empty.png", b"", "image/png")},
    )
    assert res.status_code == 400


def test_emergency_analysis_returns_no_medicines(client):
    body = client.post("/api/analyze", json={
        "text": "crushing chest pain spreading to my left arm with sweating and breathlessness, sudden onset",
    }).json()
    assert body["severity"]["triage"] == "emergency"
    assert body["medicines"]["recommended"] == []
    assert body["medicines"]["prescription_only"] == []
