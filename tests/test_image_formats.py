"""Tests for universal image support.

Every format Pillow can encode is fed through the same path, because the point
of normalisation is that the rest of the app never learns which format arrived.
The optional decoders (HEIF for iPhone photos, PDF for scanned reports) are
tested only when installed; their absence must degrade to a clear message rather
than a crash, and that degradation is tested too.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from app import vision


def _encode(fmt: str, size=(320, 120), colour="white", **save_kwargs) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


# Formats Pillow itself both writes and reads, so no optional dependency is
# needed for these. The app must treat them all identically.
NATIVE_FORMATS = ["PNG", "JPEG", "GIF", "BMP", "TIFF", "WEBP", "PPM", "TGA", "PCX"]


@pytest.mark.parametrize("fmt", NATIVE_FORMATS)
def test_every_native_format_is_detected_and_supported(fmt):
    data = _encode(fmt)
    assert vision.detect_format(data) is not None
    assert vision.is_supported(data) is True


@pytest.mark.parametrize("fmt", NATIVE_FORMATS)
def test_every_native_format_normalises_to_rgb_png(fmt):
    normalised, meta = vision.normalise_image(_encode(fmt))
    assert normalised.startswith(b"\x89PNG")
    img = Image.open(io.BytesIO(normalised))
    assert img.mode == "RGB"
    assert img.size == (320, 120)
    assert meta["normalised_size"] == [320, 120]


@pytest.mark.parametrize("fmt", NATIVE_FORMATS)
def test_every_native_format_survives_full_analysis(fmt):
    result = vision.analyse_image(_encode(fmt))
    assert result["ok"] is True
    assert "visual_features" in result


def test_multi_page_tiff_is_stacked_not_truncated():
    """A scanned multi-page report must not lose its later pages to OCR."""
    frames = [Image.new("RGB", (300, 100), c) for c in ("white", "red", "blue")]
    buf = io.BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])

    normalised, meta = vision.normalise_image(buf.getvalue())
    assert meta["frames"] == 3
    assert meta["stacked"] == 3
    # Stacked vertically, so height grows while width is preserved.
    assert meta["normalised_size"][0] == 300
    assert meta["normalised_size"][1] > 100


def test_animated_gif_keeps_its_frames():
    frames = [Image.new("RGB", (60, 60), c) for c in ("red", "green", "blue")]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100)
    _, meta = vision.normalise_image(buf.getvalue())
    assert meta["frames"] >= 2


def test_unknown_bytes_are_rejected_cleanly():
    """Random bytes must produce a reportable failure, never a traceback."""
    result = vision.analyse_image(b"this is definitely not an image")
    assert result["ok"] is False
    assert "error" in result


def test_empty_bytes_are_rejected_cleanly():
    result = vision.analyse_image(b"")
    assert result["ok"] is False


def test_grayscale_and_palette_inputs_become_rgb():
    """Mode conversion is the normaliser's job, not each consumer's."""
    grey = io.BytesIO()
    Image.new("L", (100, 100), 128).save(grey, format="PNG")
    normalised, _ = vision.normalise_image(grey.getvalue())
    assert Image.open(io.BytesIO(normalised)).mode == "RGB"

    palette = io.BytesIO()
    Image.new("P", (100, 100)).save(palette, format="PNG")
    normalised, _ = vision.normalise_image(palette.getvalue())
    assert Image.open(io.BytesIO(normalised)).mode == "RGB"


def test_transparency_is_flattened_so_ocr_sees_dark_text_on_light():
    """RGBA text on a transparent background would otherwise OCR as nothing."""
    img = Image.new("RGBA", (200, 60), (255, 255, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    normalised, _ = vision.normalise_image(buf.getvalue())
    assert Image.open(io.BytesIO(normalised)).mode == "RGB"


def test_very_large_images_are_downscaled_for_ocr():
    """A 12 MP phone photo must not be handed to Tesseract at full size."""
    data = _encode("JPEG", size=(4000, 3000))
    _, meta = vision.normalise_image(data, max_side=2400)
    assert max(meta["normalised_size"]) <= 2400


# ---- optional decoders -------------------------------------------------------
def test_heif_is_supported_when_the_plugin_is_installed():
    try:
        data = _encode("HEIF", size=(160, 90))
    except Exception:
        pytest.skip("pillow-heif cannot encode HEIF here")
    if "HEIF" not in vision.SUPPORTED_FORMATS:
        assert vision.is_supported(data) is False
        return
    normalised, meta = vision.normalise_image(data)
    assert meta["detected_format"] == "HEIF"
    assert normalised.startswith(b"\x89PNG")


def test_pdf_is_supported_when_pdfium_is_installed():
    try:
        data = _encode("PDF", size=(300, 150))
    except Exception:
        pytest.skip("Pillow cannot write PDF here")
    if "PDF" not in vision.SUPPORTED_FORMATS:
        assert vision.is_supported(data) is False
        return
    _, meta = vision.normalise_image(data)
    assert meta["detected_format"] == "PDF"
    assert meta["frames"] >= 1


def test_missing_decoder_is_reported_as_unsupported(monkeypatch):
    """Without the PDF decoder, a PDF must read as unsupported, not as broken."""
    data = _encode("PDF")
    monkeypatch.setattr(vision, "_pdfium", None)
    monkeypatch.setattr(vision, "SUPPORTED_FORMATS", vision.SUPPORTED_FORMATS - {"PDF"})
    assert vision.is_supported(data) is False
    result = vision.analyse_image(data)
    assert result["ok"] is False
