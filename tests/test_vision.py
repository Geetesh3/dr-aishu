"""Tests for image analysis: visual features and OCR of reports.

OCR tests are skipped when the Tesseract binary is unavailable, so the suite
still runs in a bare environment.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from app import vision


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def _font(size: int = 24):
    """PIL's built-in bitmap font is too coarse for reliable digit OCR."""
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _solid(colour, size=(64, 64)) -> bytes:
    return _png(Image.new("RGB", size, colour))


def _report_image(lines: list[str], size=(700, 300)) -> bytes:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = _font()
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black", font=font)
        y += 46
    return _png(img)


# ---- visual features ---------------------------------------------------------
def test_red_image_is_detected_as_reddish():
    features = vision.analyse_pixels(Image.open(io.BytesIO(_solid((210, 40, 40)))))
    assert features.dominant_hue in {"red/pink", "red"}
    assert features.redness_ratio > 0.5


def test_pale_image_is_detected_as_pale():
    features = vision.analyse_pixels(Image.open(io.BytesIO(_solid((245, 245, 240)))))
    assert features.dominant_hue in {"white/pale", "neutral"}


def test_yellow_image_is_detected():
    features = vision.analyse_pixels(Image.open(io.BytesIO(_solid((230, 220, 40)))))
    assert "yellow" in features.dominant_hue


def test_analyse_image_reports_all_sections():
    result = vision.analyse_image(_solid((200, 60, 60)))
    assert result["ok"] is True
    assert "visual_features" in result
    assert "ocr" in result
    assert "observations" in result


def test_red_image_produces_observations_and_synthetic_symptoms():
    """A strongly red image must yield observations that reach the pipeline.

    Regression: ``observations`` was reported as an empty list because the
    pixel analyser built the list but never attached it to the returned object,
    so image findings silently stopped feeding the differential diagnosis.
    """
    result = vision.analyse_image(_solid((210, 40, 40)))
    assert result["observations"], "red image should raise at least one observation"
    assert result["synthetic_symptoms"], "observations should map to symptoms"
    assert any(o["observation"] for o in result["observations"])


def test_analyse_pixels_attaches_observations_to_features():
    features = vision.analyse_pixels(Image.open(io.BytesIO(_solid((210, 40, 40)))))
    assert features.observations
    assert features.synthetic_symptoms


# ---- document detection ------------------------------------------------------
def _text_page(lines: int, size=(720, 1000)) -> bytes:
    """A page of dense printed text, as a lab report or prescription would be."""
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    for i in range(lines):
        draw.text((20, 16 + i * 24), "Hemoglobin 9.2 g/dL WBC 12400 /uL", fill="black")
    return _png(img)


def test_dense_text_page_is_detected_as_document():
    features = vision.analyse_pixels(Image.open(io.BytesIO(_text_page(45))))
    assert features.is_likely_document is True


def test_sparse_text_page_is_detected_as_document():
    features = vision.analyse_pixels(Image.open(io.BytesIO(_text_page(8))))
    assert features.is_likely_document is True


def test_blank_page_is_not_a_document():
    """A blank sheet has no clinical content, so it must not claim to be one."""
    features = vision.analyse_pixels(Image.open(io.BytesIO(_solid((255, 255, 255), (720, 1000)))))
    assert features.is_likely_document is False


@pytest.mark.parametrize("colour", [(190, 140, 125), (230, 205, 195), (205, 150, 135)])
def test_skin_toned_photos_are_not_documents(colour):
    """Photographs of skin carry saturation, unlike greyscale print."""
    features = vision.analyse_pixels(Image.open(io.BytesIO(_solid(colour, (720, 720)))))
    assert features.is_likely_document is False


# ---- preprocessing -----------------------------------------------------------
def test_small_images_are_upscaled_for_ocr():
    prepared, info = vision._preprocess_for_ocr(_solid((255, 255, 255), size=(200, 100)))
    assert info["preprocessed"] is True
    assert info["ocr_size"][0] >= info["original_size"][0]
    assert prepared


def test_preprocessing_falls_back_on_invalid_bytes():
    prepared, info = vision._preprocess_for_ocr(b"not an image")
    assert info["preprocessed"] is False
    assert prepared == b"not an image"


# ---- OCR ---------------------------------------------------------------------
requires_tesseract = pytest.mark.skipif(
    not vision.TESSERACT, reason="tesseract binary not installed"
)


@requires_tesseract
def test_ocr_reads_a_lab_report():
    image = _report_image([
        "LAB REPORT",
        "Hb 9.2 g/dL",
        "WBC 12000 /uL",
        "Reported fever for 4 days",
    ])
    text, meta = vision.ocr_text(image)
    assert meta["available"] is True
    assert meta.get("preprocessed") is True
    assert "fever" in text.lower()
    assert "9.2" in text


@requires_tesseract
def test_ocr_confidence_is_a_percentage():
    """Regression: the TSV parser once read word text as the confidence value."""
    text, meta = vision.ocr_text(_report_image(["WBC 12000 /uL", "Platelets 1.4 lakh"]))
    confidence = meta["confidence"]
    assert confidence is not None
    assert 0.0 <= confidence <= 100.0, f"confidence {confidence} is not a percentage"


@requires_tesseract
def test_ocr_handles_an_image_with_no_text():
    text, meta = vision.ocr_text(_solid((120, 130, 140)))
    assert text.strip() == ""
    assert meta["available"] is True


# ---- TSV reconstruction ------------------------------------------------------
# Tesseract is run once (tsv) and the text rebuilt from it, halving the cost of
# the slowest step in an image request. These tests hold the reconstruction to
# Tesseract's own `txt` output, so the two never drift apart.
def test_text_from_tsv_joins_words_lines_and_paragraphs():
    tsv = "\n".join([
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        "5\t1\t1\t1\t1\t1\t0\t0\t0\t0\t96.0\tHello",
        "5\t1\t1\t1\t1\t2\t0\t0\t0\t0\t95.0\tworld",
        "5\t1\t1\t1\t2\t1\t0\t0\t0\t0\t94.0\tsecond",
        "5\t1\t1\t2\t1\t1\t0\t0\t0\t0\t93.0\tparagraph",
    ])
    assert vision._text_from_tsv(tsv) == "Hello world\nsecond\n\nparagraph"


def test_text_from_tsv_ignores_blank_words():
    tsv = "\n".join([
        "level\tblock_num\tpar_num\tline_num\ttext\tconf",
        "5\t1\t1\t1\tHi\t90.0",
        "5\t1\t1\t1\t\t-1",
        "5\t1\t1\t1\tthere\t90.0",
    ])
    assert vision._text_from_tsv(tsv) == "Hi there"


def test_text_from_tsv_handles_missing_header_columns():
    """A different Tesseract build must degrade to no text, not to a crash."""
    assert vision._text_from_tsv("level\ttext\n5\thello\n") == ""
    assert vision._text_from_tsv("") == ""


@requires_tesseract
def test_tsv_reconstruction_matches_tesseract_plain_text():
    """The single-invocation path must reproduce the two-invocation output."""
    import subprocess
    import tempfile
    from pathlib import Path

    image = _report_image([
        "CITY DIAGNOSTIC LABORATORY",
        "Haemoglobin: 8.1 g/dL",
        "Platelet Count: 1.4 lakh/cumm",
        "Uric Acid: 9.1 mg/dL",
    ])
    prepared, _ = vision._preprocess_for_ocr(image)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "u.png"
        path.write_bytes(prepared)
        tsv = subprocess.run(
            [vision.TESSERACT, str(path), "stdout", "-l", "eng", "--psm", "6", "tsv"],
            check=True, capture_output=True,
        ).stdout.decode()
        plain = subprocess.run(
            [vision.TESSERACT, str(path), "stdout", "-l", "eng", "--psm", "6"],
            check=True, capture_output=True,
        ).stdout.decode()
    assert vision._text_from_tsv(tsv).strip() == plain.strip()
