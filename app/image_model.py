"""Trained image classifier for skin and eye photographs.

`app/vision.py` reads text out of an image and derives descriptive colour
observations. This module adds the other half: a model actually trained on
labelled clinical photographs, so an upload of a rash or an eye can be compared
against real examples rather than only described.

Two models are trained separately because the classes do not overlap:

    skin  6 classes  - Atopic Dermatitis, Contact Dermatitis, Eczema, Scabies,
                       Seborrheic Dermatitis, Tinea Corporis
    eye   5 classes  - Cataract, Conjunctivitis, Eyelid, Uveitis, Normal

The feature pipeline lives here and is used by both the trainer and the server,
which is the point: training and serving must compute the same vector or the
model silently degrades. Features are colour statistics and a downsampled
grayscale thumbnail, so no deep-learning runtime is needed and inference costs a
few milliseconds.

Safety: a prediction from this module is a *suggestion with a confidence*, never
a diagnosis. It is surfaced to the user as such and is only folded into the
differential as low-weight evidence; it can never trigger or suppress the
emergency gate, which stays the responsibility of `app/pipeline.py`.
"""
from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
IMAGE_MODEL_PATH = MODEL_DIR / "image_model.joblib"

_lock = threading.Lock()
_state: dict[str, Any] = {"model": None, "meta": None, "loaded": False}

# Feature vector layout. Kept explicit so the trainer and the server agree and a
# change is visible rather than buried in a magic 1106.
THUMB = 32          # grayscale thumbnail edge
H_BINS = 16         # hue histogram bins
S_BINS = 8
V_BINS = 8
RGB_BINS = 8
FEATURE_DIM = (
    (THUMB * THUMB)          # grayscale thumbnail
    + H_BINS + S_BINS + V_BINS  # HSV histograms
    + (RGB_BINS * 3)         # RGB histograms
    + 12                     # rgb means, rgb stddev, hsv means, hsv stddev
    + 4                      # red / yellow / dark / pale ratios
    + 4                      # edge density, colour variance, brightness, stddev
)


def _pixels(img):
    """Pixel accessor that avoids Pillow's `getdata` deprecation."""
    if hasattr(img, "get_flattened_data"):
        return list(img.get_flattened_data())
    return list(img.getdata())


def _features_from_image(img) -> list[float]:
    """Compute the feature vector. Pure function of the pixels."""
    from PIL import ImageFilter, ImageStat

    rgb = img.convert("RGB")
    hsv = rgb.convert("HSV")
    grey = rgb.convert("L")

    feats: list[float] = []

    # 1. Grayscale thumbnail: coarse shape and texture, scale-invariant.
    thumb = grey.copy()
    thumb.thumbnail((THUMB, THUMB))
    canvas = thumb.resize((THUMB, THUMB))
    feats.extend(v / 255.0 for v in _pixels(canvas))

    # 2. HSV histograms: the colour signature that separates erythema from
    #    scaly white plaques from normal skin.
    for channel, bins in ((0, H_BINS), (1, S_BINS), (2, V_BINS)):
        hist = hsv.getchannel(channel).histogram()
        band = len(hist) // bins
        counts = [sum(hist[i * band:(i + 1) * band]) for i in range(bins)]
        total = sum(counts) or 1
        feats.extend(c / total for c in counts)

    # 3. RGB histograms.
    for channel in range(3):
        hist = rgb.getchannel(channel).histogram()
        band = len(hist) // RGB_BINS
        counts = [sum(hist[i * band:(i + 1) * band]) for i in range(RGB_BINS)]
        total = sum(counts) or 1
        feats.extend(c / total for c in counts)

    # 4. Summary statistics.
    small = rgb.copy()
    small.thumbnail((320, 320))
    pixels = _pixels(small)
    n = len(pixels) or 1
    r_mean = sum(p[0] for p in pixels) / n
    g_mean = sum(p[1] for p in pixels) / n
    b_mean = sum(p[2] for p in pixels) / n
    feats.extend((r_mean / 255, g_mean / 255, b_mean / 255))

    stat_rgb = ImageStat.Stat(small)
    feats.extend(v / 255 for v in stat_rgb.stddev)
    stat_hsv = ImageStat.Stat(hsv)
    feats.extend(v / 255 for v in stat_hsv.mean)
    feats.extend(v / 255 for v in stat_hsv.stddev)

    red = yellow = dark = pale = 0
    for r, g, b in pixels:
        mx, mn = max(r, g, b), min(r, g, b)
        if r > 110 and r - g > 32 and r - b > 32:
            red += 1
        elif r > 150 and g > 140 and b < 130 and abs(r - g) < 45:
            yellow += 1
        if mx < 62:
            dark += 1
        if mn > 200 and mx - mn < 28:
            pale += 1
    feats.extend((red / n, yellow / n, dark / n, pale / n))

    edges = grey.filter(ImageFilter.FIND_EDGES)
    edge_density = min(1.0, (ImageStat.Stat(edges).mean[0] / 255.0) * 2.2)
    feats.append(edge_density)
    feats.append(sum(stat_rgb.stddev) / 765.0)
    feats.append(sum(stat_rgb.mean) / 765.0)
    feats.append(min(1.0, stat_rgb.stddev[0] / 128.0))

    return feats


def extract_features(source: bytes | str | Path) -> list[float]:
    """Feature vector for an image given as bytes or a path."""
    from PIL import Image, ImageOps

    if isinstance(source, (str, Path)):
        img = Image.open(source)
    else:
        img = Image.open(io.BytesIO(source))
    img.load()
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 - optional correction
        pass
    return _features_from_image(img)


# ---------------------------------------------------------------------------
# model lifecycle
# ---------------------------------------------------------------------------
def load() -> dict[str, Any] | None:
    """Load the trained image model once. Returns None when none is trained."""
    with _lock:
        if _state["loaded"]:
            return _state["model"]
        _state["loaded"] = True
        if not IMAGE_MODEL_PATH.exists():
            return None
        try:
            import joblib

            bundle = joblib.load(IMAGE_MODEL_PATH)
            _state["model"] = bundle
            _state["meta"] = bundle.get("meta", {})
        except Exception as exc:  # noqa: BLE001 - a corrupt model must not 500
            _state["model"] = None
            _state["meta"] = {"error": f"{type(exc).__name__}: {exc}"}
        return _state["model"]


def reset() -> None:
    with _lock:
        _state.update({"model": None, "meta": None, "loaded": False})


def is_ready() -> bool:
    return load() is not None


def status() -> dict[str, Any]:
    bundle = load()
    if bundle is None:
        return {
            "available": False,
            "path": str(IMAGE_MODEL_PATH.relative_to(Path(__file__).resolve().parent.parent))
            if IMAGE_MODEL_PATH.exists() else None,
            "hint": "Run: python scripts/train_image_model.py",
        }
    meta = bundle.get("meta", {})
    return {
        "available": True,
        "path": str(IMAGE_MODEL_PATH.relative_to(Path(__file__).resolve().parent.parent)),
        "trained_at": meta.get("trained_at"),
        "modalities": {
            name: {
                "classes": info.get("classes", []),
                "n_train": info.get("n_train"),
                "accuracy": info.get("accuracy"),
                "cv_accuracy": info.get("cv_accuracy"),
                "cv_std": info.get("cv_std"),
                "split": info.get("split"),
            }
            for name, info in (meta.get("models") or {}).items()
        },
    }


# Below this top-class probability the result is reported as uncertain rather
# than named. The skin model sits near 0.40 held-out accuracy, so a confident-
# sounding label would be misleading on roughly three of five images.
MIN_CONFIDENCE = 0.55


def predict(image_bytes: bytes, modality: str | None = None) -> dict[str, Any] | None:
    """Classify an image. Returns None when no model is trained.

    `modality` pins the model ("skin" or "eye"). When it is omitted every model
    runs and every result is returned under `predictions`, with `best` left
    None. That is deliberate: the two class sets do not overlap, and the models
    are overconfident outside their domain — the eye model scores 1.00 on skin
    photographs — so choosing a winner by confidence would name a cataract for a
    rash. The person uploading knows which body part the photo shows, so the
    caller passes the modality when it is known and otherwise shows both.

    `best` is only populated when the modality is pinned, and only when the
    top-class probability clears `MIN_CONFIDENCE`; `uncertain` says when it did
    not. HGB probabilities are not calibrated, so this is a floor against
    nonsense, not a claim of reliability.
    """
    bundle = load()
    if bundle is None:
        return None

    try:
        vector = extract_features(image_bytes)
    except Exception as exc:  # noqa: BLE001 - a bad upload must not 500
        return {"ok": False, "error": f"feature extraction failed: {exc}"}

    import numpy as np

    models: dict[str, Any] = bundle.get("models", {})
    if modality is not None and modality not in models:
        return {"ok": False, "error": f"no model for modality {modality!r}",
                "available": sorted(models)}
    candidates = {modality: models[modality]} if modality else models

    predictions: dict[str, Any] = {}
    for name, entry in candidates.items():
        clf = entry["model"]
        classes = entry["classes"]
        x = np.asarray([vector], dtype="float32")
        proba = clf.predict_proba(x)[0]
        order = proba.argsort()[::-1]
        predictions[name] = {
            "label": classes[order[0]],
            "confidence": round(float(proba[order[0]]), 4),
            "top": [
                {"label": classes[i], "confidence": round(float(proba[i]), 4)}
                for i in order[:3]
            ],
            "accuracy": entry.get("accuracy"),
        }

    result: dict[str, Any] = {
        "ok": True,
        "predictions": predictions,
        "best": None,
        "modality": None,
        "label": None,
        "uncertain": True,
        "min_confidence": MIN_CONFIDENCE,
        "note": "Model suggestion, not a diagnosis.",
    }
    if modality:
        top = predictions[modality]
        if top["confidence"] >= MIN_CONFIDENCE:
            result.update({
                "best": {"modality": modality, **top},
                "modality": modality,
                "label": top["label"],
                "confidence": top["confidence"],
                "uncertain": False,
            })
        else:
            result["confidence"] = top["confidence"]
    else:
        result["note"] += " Pass modality='skin' or 'eye' to select one."
    return result
