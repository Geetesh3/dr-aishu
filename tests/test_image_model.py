"""Tests for the trained image classifier and its dataset preparation.

The model itself is a build artefact and may not exist in a fresh checkout, so
these tests cover the parts that must hold either way: the feature vector is
stable and finite, the trainer and the server compute the same vector, and the
prediction API never names a class when no model is loaded or the modality is
unknown. When a model *is* present it is exercised on a real image.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from app import image_model

DATASETS = image_model.MODEL_DIR.parent.parent / "data" / "external" / "datasets"


def _png(colour=(190, 140, 130), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def _gradient() -> bytes:
    img = Image.new("RGB", (96, 96))
    for x in range(96):
        for y in range(96):
            img.putpixel((x, y), (x * 2 % 256, y * 2 % 256, (x + y) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_feature_vector_has_declared_dimension():
    feats = image_model.extract_features(_png())
    assert len(feats) == image_model.FEATURE_DIM


def test_feature_vector_is_finite_and_normalised():
    feats = image_model.extract_features(_gradient())
    assert all(isinstance(v, float) for v in feats)
    assert all(v == v for v in feats), "no NaN"
    assert all(abs(v) <= 5 for v in feats)


def test_same_image_same_features():
    data = _gradient()
    assert image_model.extract_features(data) == image_model.extract_features(data)


def test_different_images_differ():
    a = image_model.extract_features(_png((200, 30, 30)))
    b = image_model.extract_features(_png((30, 30, 200)))
    assert a != b


def test_predict_returns_none_without_model(monkeypatch):
    monkeypatch.setattr(image_model, "IMAGE_MODEL_PATH", image_model.MODEL_DIR / "nope.joblib")
    image_model.reset()
    try:
        assert image_model.predict(_png()) is None
    finally:
        image_model.reset()


def test_predict_rejects_unknown_modality(monkeypatch):
    class _Fake:
        def predict_proba(self, x):
            import numpy as np

            return np.asarray([[0.9, 0.1]])

    bundle = {"models": {"skin": {"model": _Fake(), "classes": ["A", "B"]}}, "meta": {}}
    monkeypatch.setattr(image_model, "load", lambda: bundle)
    out = image_model.predict(_png(), modality="retina")
    assert out is not None and out["ok"] is False
    assert "retina" in out["error"]


def test_predict_without_modality_does_not_pick_a_winner(monkeypatch):
    import numpy as np

    class _Fake:
        def __init__(self, classes):
            self.classes = classes

        def predict_proba(self, x):
            return np.asarray([[0.99, 0.01]])

    bundle = {
        "models": {
            "skin": {"model": _Fake(["Eczema", "X"]), "classes": ["Eczema", "X"]},
            "eye": {"model": _Fake(["Cataract", "Y"]), "classes": ["Cataract", "Y"]},
        },
        "meta": {},
    }
    monkeypatch.setattr(image_model, "load", lambda: bundle)
    out = image_model.predict(_png())
    assert out is not None
    # Both models are reported and no single label is chosen, because the eye
    # model is overconfident outside its domain.
    assert set(out["predictions"]) == {"skin", "eye"}
    assert out["best"] is None and out["label"] is None and out["uncertain"] is True


def test_pinned_modality_names_a_class_when_confident(monkeypatch):
    import numpy as np

    class _Fake:
        def predict_proba(self, x):
            return np.asarray([[0.95, 0.05]])

    bundle = {"models": {"skin": {"model": _Fake(), "classes": ["Eczema", "X"]}}, "meta": {}}
    monkeypatch.setattr(image_model, "load", lambda: bundle)
    out = image_model.predict(_png(), modality="skin")
    assert out["label"] == "Eczema" and out["uncertain"] is False


def test_pinned_modality_reports_uncertain_below_threshold(monkeypatch):
    import numpy as np

    class _Fake:
        def predict_proba(self, x):
            return np.asarray([[0.40, 0.35, 0.25]])

    bundle = {"models": {"skin": {"model": _Fake(), "classes": ["A", "B", "C"]}}, "meta": {}}
    monkeypatch.setattr(image_model, "load", lambda: bundle)
    out = image_model.predict(_png(), modality="skin")
    assert out["uncertain"] is True and out["label"] is None


def test_status_reports_not_available_without_model(monkeypatch):
    monkeypatch.setattr(image_model, "IMAGE_MODEL_PATH", image_model.MODEL_DIR / "nope.joblib")
    image_model.reset()
    try:
        assert image_model.status()["available"] is False
    finally:
        image_model.reset()


@pytest.mark.skipif(not (DATASETS / "eye").exists(), reason="image datasets not prepared")
def test_real_model_if_trained():
    if not image_model.is_ready():
        pytest.skip("no trained image model")
    cls_dir = DATASETS / "eye" / "Cataract"
    files = sorted(cls_dir.glob("*.jpg"))[:1]
    if not files:
        pytest.skip("no cataract images")
    out = image_model.predict(files[0].read_bytes(), modality="eye")
    assert out["ok"] is True
    assert out["label"] in image_model.status()["modalities"]["eye"]["classes"]


@pytest.mark.skipif(not (DATASETS / "skin").exists(), reason="image datasets not prepared")
def test_vision_includes_model_prediction():
    from app import vision

    if not image_model.is_ready():
        pytest.skip("no trained image model")
    files = sorted((DATASETS / "skin" / "Eczema").glob("*.jpg"))[:1]
    if not files:
        pytest.skip("no eczema images")
    out = vision.analyse_image(files[0].read_bytes(), modality="skin")
    assert out["ok"] is True
    assert "model_prediction" in out
    assert out["model_prediction"]["ok"] is True
