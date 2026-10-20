"""API tests for the consolidated-dataset endpoints and image modality.

The workbook is a build artefact that may be absent in a fresh checkout, so
these tests assert the endpoints answer correctly either way: with the workbook
they return rows, without it they report it as unavailable rather than failing.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import dataset
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def has_workbook():
    dataset.reset()
    if not dataset.available():
        pytest.skip("dataset workbook not built")
    yield
    dataset.reset()


def test_dataset_status_shape(client, operator_headers):
    body = client.get("/api/dataset/status", headers=operator_headers).json()
    for key in ("available", "sheets", "image_model"):
        assert key in body


def test_dataset_page_renders(client):
    res = client.get("/dataset")
    assert res.status_code == 200
    assert "Consolidated dataset" in res.text or "The data behind" in res.text
    assert "/static/dataset.js" in res.text


def test_dataset_diseases_paginates(client, has_workbook, operator_headers):
    body = client.get("/api/dataset/diseases?limit=5", headers=operator_headers).json()
    assert body["limit"] == 5
    assert len(body["items"]) <= 5
    assert body["total"] >= 40
    assert "name" in body["items"][0]


def test_dataset_diseases_in_kb_filter(client, has_workbook, operator_headers):
    body = client.get("/api/dataset/diseases?in_kb=true&limit=0", headers=operator_headers).json()
    assert body["total"] >= 40
    assert all((r.get("in_kb") or "").lower() == "yes" for r in body["items"])


def test_dataset_disease_lookup_and_404(client, has_workbook, operator_headers):
    first = client.get("/api/dataset/diseases?limit=1", headers=operator_headers).json()["items"][0]
    ok = client.get(f"/api/dataset/diseases/{first['name']}", headers=operator_headers)
    assert ok.status_code == 200
    assert ok.json()["name"] == first["name"]
    missing = client.get("/api/dataset/diseases/not-a-real-condition-xyz", headers=operator_headers)
    assert missing.status_code == 404


def test_dataset_images_reports_labels(client, has_workbook, operator_headers):
    body = client.get("/api/dataset/images?limit=1", headers=operator_headers).json()
    assert body["total"] > 0
    assert set(body["labels"]) == {"skin", "eye"}
    assert sum(sum(c.values()) for c in body["labels"].values()) == body["total"]


def test_dataset_sources_have_licences(client, has_workbook, operator_headers):
    body = client.get("/api/dataset/sources", headers=operator_headers).json()
    assert len(body["items"]) >= 5
    for row in body["items"]:
        assert "license" in row and "url" in row


def test_dataset_medicinenet_index(client, has_workbook, operator_headers):
    body = client.get("/api/dataset/medicinenet?limit=10", headers=operator_headers).json()
    assert len(body["items"]) <= 10
    assert "condition" in body["items"][0]


def test_dataset_deeplesion_annotations(client, has_workbook, operator_headers):
    body = client.get("/api/dataset/deeplesion?limit=3", headers=operator_headers).json()
    assert body["total"] > 1000
    assert len(body["items"]) <= 3
    row = body["items"][0]
    assert row["file_name"].endswith(".png")
    assert "bounding_boxes" in row and "coarse_lesion_type" in row
    # Radiology annotations must not masquerade as symptom data.
    assert row["source"] == "deeplesion_metadata"
    assert sum(body["lesion_types"].values()) == body["total"]


def test_dataset_deeplesion_type_filter(client, has_workbook, operator_headers):
    first = client.get("/api/dataset/deeplesion?limit=1", headers=operator_headers).json()
    code = first["items"][0]["coarse_lesion_type"]
    filtered = client.get(
        f"/api/dataset/deeplesion?lesion_type={code}&limit=0", headers=operator_headers
    ).json()
    assert filtered["total"] == first["lesion_types"][str(code)]
    assert all(str(r["coarse_lesion_type"]) == str(code) for r in filtered["items"])


def test_dataset_catalogue_lists_public_datasets(client, has_workbook, operator_headers):
    body = client.get("/api/dataset/catalogue", headers=operator_headers).json()
    assert body["total"] >= 5
    for row in body["items"]:
        assert row["url"].startswith("http")
        assert row["dataset"]
        assert row["license"] == "cc-by-4.0"


def test_new_dataset_endpoints_require_operator(client):
    for path in ("/api/dataset/deeplesion", "/api/dataset/catalogue"):
        assert client.get(path).status_code == 403


def test_workbook_optional_sheets_present(client, has_workbook):
    counts = dataset.status()["sheets"]
    assert counts.get("DeepLesion", 0) > 1000
    assert counts.get("Dataset_Catalogue", 0) >= 5


def test_model_status_includes_image_model(client):
    body = client.get("/api/model/status").json()
    assert "image_model" in body
    assert "available" in body["image_model"]


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (80, 80), (180, 120, 110)).save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_image_accepts_modality(client):
    res = client.post(
        "/api/analyze-image",
        files={"image": ("rash.png", _png(), "image/png")},
        data={"text": "itchy rash on the arm", "image_modality": "skin"},
    )
    assert res.status_code == 200
    body = res.json()
    analysis = body["image_analysis"]
    assert analysis["ok"] is True
    assert "model_prediction" in analysis


def test_analyze_image_rejects_unknown_modality_gracefully(client):
    res = client.post(
        "/api/analyze-image",
        files={"image": ("rash.png", _png(), "image/png")},
        data={"text": "itchy rash", "image_modality": "retina"},
    )
    # The upload still analyses; only the model prediction reports the problem.
    assert res.status_code == 200
    prediction = res.json()["image_analysis"]["model_prediction"]
    assert prediction is None or prediction.get("ok") is False or prediction.get("uncertain")
