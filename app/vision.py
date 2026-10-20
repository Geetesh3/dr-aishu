"""Image analysis: OCR text extraction plus visual feature heuristics.

Two complementary signals are produced from an uploaded image:

1. Text - any clinical text in the image (a lab report, a prescription label, a
   symptom note) is read with Tesseract OCR and fed into the same NLP pipeline
   used for typed input.
2. Visual features - colour and texture statistics are computed on the pixel
   data to derive coarse observations such as "prominent red area",
   "yellow/white discharge", "dark or blackened region". These become synthetic
   symptom observations that feed the differential diagnosis.

The visual layer is deliberately conservative: it reports descriptive,
non-diagnostic observations and always defers the final interpretation to the
knowledge-base scoring, because a photograph alone cannot confirm a diagnosis.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat

# HEIC/HEIF and AVIF are common on modern phones and are not always compiled
# into Pillow. Register the plugins when the optional packages are present; if
# they are missing the format is simply reported as unsupported by name.
try:  # pragma: no cover - optional dependency
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # noqa: BLE001 - optional dependency
    pass

try:  # pragma: no cover - optional dependency
    import pillow_avif  # noqa: F401
except Exception:  # noqa: BLE001 - optional dependency
    pass

try:  # pragma: no cover - optional dependency
    import pypdfium2 as _pdfium
except Exception:  # noqa: BLE001 - optional dependency
    _pdfium = None

TESSERACT = shutil.which("tesseract")


def _pillow_formats() -> set[str]:
    try:
        Image.init()
        return {fmt.upper() for fmt in Image.registered_extensions().values()}
    except Exception:  # noqa: BLE001
        return set()


SUPPORTED_FORMATS = _pillow_formats()
if _pdfium is not None:  # pragma: no cover - optional dependency
    SUPPORTED_FORMATS.add("PDF")

# Magic-byte signatures for the formats whose content type is worth trusting
# even when the client sends a generic one (or none at all).
_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"BM", "BMP"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
    (b"RIFF", "WEBP"),
    (b"\x00\x00\x01\x00", "ICO"),
    (b"8BPS", "PSD"),
    (b"qoif", "QOI"),
    (b"DDS ", "DDS"),
    (b"\x00\x00\x00\x0cjP", "JP2"),
    (b"II+\x00", "DNG"),
    (b"farbfeld", "FARBFELD"),
    (b"%PDF", "PDF"),
]
_HEIF_BRANDS = (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"avif", b"avis")


def detect_format(data: bytes) -> str | None:
    """Identify an image by its leading bytes.

    The uploaded content type is attacker-controlled and frequently wrong (a
    phone may send `application/octet-stream` for a HEIC), so the format used for
    decoding is the one sniffed from the data, not the header the client sent.
    """
    if not data:
        return None
    head = data[:32]
    for magic, fmt in _MAGIC:
        if head.startswith(magic):
            if fmt == "WEBP" and head[8:12] != b"WEBP":
                continue
            return fmt
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12].lower()
        if any(brand.startswith(b) for b in _HEIF_BRANDS):
            return "HEIF"
    try:
        with Image.open(io.BytesIO(data)) as img:
            return (img.format or "").upper() or None
    except Exception:  # noqa: BLE001
        return None


def is_supported(data: bytes) -> bool:
    fmt = detect_format(data)
    if fmt is None:
        return False
    if fmt in ("HEIF", "HEIC", "AVIF"):
        return "HEIF" in SUPPORTED_FORMATS or "AVIF" in SUPPORTED_FORMATS
    if fmt == "PDF":
        return _pdfium is not None
    return fmt in SUPPORTED_FORMATS or fmt in {"JPEG", "PNG", "GIF", "BMP", "TIFF", "WEBP"}


def _decode(data: bytes) -> tuple[list[Image.Image], dict[str, Any]]:
    """Decode any supported upload into one or more RGB frames.

    Multi-page documents (TIFF, animated GIF, PDF) are returned as separate
    frames so a photographed multi-page report is not silently truncated to its
    first page.
    """
    meta: dict[str, Any] = {}
    fmt = detect_format(data)
    meta["detected_format"] = fmt or "unknown"

    if fmt == "PDF":
        if _pdfium is None:  # pragma: no cover - guarded by is_supported
            raise ValueError("PDF support requires pypdfium2")
        pdf = _pdfium.PdfDocument(data)
        pages: list[Image.Image] = []
        for i in range(min(len(pdf), 10)):
            bitmap = pdf[i].render(scale=2)
            pages.append(bitmap.to_pil().convert("RGB"))
        meta["pages"] = len(pages)
        meta["frames"] = len(pages)
        if not pages:
            raise ValueError("PDF has no renderable pages")
        return pages, meta

    img = Image.open(io.BytesIO(data))
    frames: list[Image.Image] = []
    n_frames = getattr(img, "n_frames", 1)
    for i in range(min(n_frames, 10)):
        try:
            img.seek(i)
        except EOFError:  # pragma: no cover - malformed frame table
            break
        frames.append(img.convert("RGB").copy())
    meta["frames"] = len(frames)
    return frames, meta


def normalise_image(data: bytes, *, max_side: int = 2400) -> tuple[bytes, dict[str, Any]]:
    """Decode an upload of any format and re-encode it as a plain RGB PNG.

    Every downstream consumer (pixel statistics, OCR) then works on one format
    with predictable channels, so adding a decoder is the only change needed to
    support a new input type. Multi-page inputs are stacked vertically so a
    single OCR pass reads the whole document.
    """
    frames, meta = _decode(data)
    if not frames:
        raise ValueError("no image frames could be decoded")

    if len(frames) == 1:
        canvas = frames[0]
    else:
        gap = 16
        width = max(f.width for f in frames)
        height = sum(f.height for f in frames) + gap * (len(frames) - 1)
        canvas = Image.new("RGB", (width, height), "white")
        y = 0
        for f in frames:
            canvas.paste(f, (0, y))
            y += f.height + gap
        meta["stacked"] = len(frames)

    if max(canvas.size) > max_side:
        scale = max_side / max(canvas.size)
        canvas = canvas.resize(
            (max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))),
            Image.LANCZOS,
        )
        meta["downscaled"] = True

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    meta["normalised_size"] = list(canvas.size)
    return buf.getvalue(), meta



@dataclass
class VisualFeatures:
    width: int
    height: int
    mean_rgb: tuple[float, float, float]
    dominant_hue: str
    redness_ratio: float
    yellowness_ratio: float
    dark_ratio: float
    pallor_ratio: float
    saturation_mean: float
    brightness_mean: float
    edge_density: float
    colour_variance: float
    is_likely_document: bool
    observations: list[dict[str, Any]] = field(default_factory=list)
    synthetic_symptoms: list[dict[str, Any]] = field(default_factory=list)


def _classify_hue(r: float, g: float, b: float) -> str:
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 18:
        if mx < 60:
            return "dark/black"
        if mx > 205:
            return "white/pale"
        return "grey/neutral"
    if r >= g and r >= b:
        if g > b + 25 and r > 150:
            return "yellow/orange"
        if b > g + 15:
            return "pink/magenta"
        return "red"
    if g >= r and g >= b:
        return "green"
    if b >= r and b >= g:
        return "blue/purple"
    return "mixed"


def analyse_pixels(img: Image.Image) -> VisualFeatures:
    rgb = img.convert("RGB")
    small = rgb.copy()
    small.thumbnail((320, 320))
    w, h = small.size
    # `getdata` is deprecated; `get_flattened_data` is the supported replacement
    # when available, with a fallback for older Pillow releases.
    if hasattr(small, "get_flattened_data"):
        pixels = list(small.get_flattened_data())
    else:
        pixels = list(small.getdata())
    n = len(pixels) or 1

    total_r = sum(p[0] for p in pixels) / n
    total_g = sum(p[1] for p in pixels) / n
    total_b = sum(p[2] for p in pixels) / n

    red = yellow = dark = pale = 0
    sat_sum = 0.0
    bright_sum = 0.0
    for r, g, b in pixels:
        mx, mn = max(r, g, b), min(r, g, b)
        bright_sum += (r + g + b) / 3
        sat_sum += 0 if mx == 0 else (mx - mn) / mx
        # Skin-flush / inflammation: red clearly dominant and reasonably saturated.
        if r > 110 and r - g > 32 and r - b > 32:
            red += 1
        # Discharge / pus / crust: yellow-white dominance.
        elif r > 150 and g > 140 and b < 130 and abs(r - g) < 45:
            yellow += 1
        # Necrosis / eschar / very dark lesion.
        if mx < 62:
            dark += 1
        # Pallor / blanching / white coating.
        if mn > 200 and mx - mn < 28:
            pale += 1

    grey = small.convert("L")
    edges = grey.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    edge_density = min(1.0, (stat.mean[0] / 255.0) * 2.2)

    stat_rgb = ImageStat.Stat(small)
    variance = sum(stat_rgb.stddev) / 3.0

    features = VisualFeatures(
        width=img.width,
        height=img.height,
        mean_rgb=(round(total_r, 1), round(total_g, 1), round(total_b, 1)),
        dominant_hue=_classify_hue(total_r, total_g, total_b),
        redness_ratio=round(red / n, 4),
        yellowness_ratio=round(yellow / n, 4),
        dark_ratio=round(dark / n, 4),
        pallor_ratio=round(pale / n, 4),
        saturation_mean=round(sat_sum / n, 4),
        brightness_mean=round(bright_sum / n, 1),
        edge_density=round(edge_density, 4),
        colour_variance=round(variance, 2),
        # A page of text is near-white, fully desaturated (printed text on paper
        # is greyscale) and has some edge content. Thresholds are calibrated
        # against the fixtures in tests/test_vision.py: text pages measure
        # saturation 0.00 and edge density 0.06-0.23, while photographs of skin
        # measure saturation 0.15-0.40, so saturation is the reliable separator
        # and edge density only rules out a blank page.
        is_likely_document=(
            sat_sum / n < 0.08 and bright_sum / n > 150 and edge_density > 0.05
        ),
    )

    obs: list[dict[str, Any]] = []
    synth: list[dict[str, Any]] = []

    def add(observation: str, confidence: float, symptom_id: str | None, detail: str = "") -> None:
        obs.append({"observation": observation, "confidence": round(confidence, 2), "detail": detail})
        if symptom_id:
            synth.append({"symptom_id": symptom_id, "source": "image", "observation": observation,
                          "confidence": round(confidence, 2)})

    if features.is_likely_document:
        add("Image appears to be a document or report rather than a photograph of a body area",
            0.85, None, "Text extraction will carry the clinical detail.")

    if features.redness_ratio > 0.22:
        add("Prominent red area consistent with inflammation, flushing or erythema",
            min(0.9, 0.4 + features.redness_ratio), "skin_redness",
            f"{features.redness_ratio * 100:.0f}% of sampled pixels are strongly red")
    elif features.redness_ratio > 0.10:
        add("Localised redness present", min(0.75, 0.35 + features.redness_ratio), "skin_redness")

    if features.yellowness_ratio > 0.12:
        add("Yellow or creamy area suggestive of pus, exudate or discharge",
            min(0.85, 0.4 + features.yellowness_ratio), "pus_discharge",
            f"{features.yellowness_ratio * 100:.0f}% of sampled pixels are yellow-toned")

    if features.dark_ratio > 0.18:
        add("Dark or blackened region present - requires clinical assessment",
            min(0.85, 0.4 + features.dark_ratio), "dark_skin_patch",
            f"{features.dark_ratio * 100:.0f}% of sampled pixels are very dark")
    elif features.dark_ratio > 0.08:
        add("Darker area noted within the image", 0.45, None)

    if features.pallor_ratio > 0.30 and not features.is_likely_document:
        add("Large pale or blanched area", 0.5, None)

    if features.colour_variance > 62 and features.edge_density > 0.35 and not features.is_likely_document:
        add("Mottled or irregular surface texture consistent with a rash or lesion",
            0.55, "rash", f"colour variance {features.colour_variance}, edge density {features.edge_density}")

    if features.saturation_mean > 0.45 and features.redness_ratio <= 0.10 and not features.is_likely_document:
        add("High colour saturation overall", 0.35, None)

    if not obs:
        add("No distinctive colour or texture abnormality detected in the image",
            0.4, None, "Absence of a visual abnormality does not rule out illness.")

    features.observations = obs
    features.synthetic_symptoms = synth
    return features


def _preprocess_for_ocr(image_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    """Prepare an upload for OCR: greyscale, upscale small images, boost contrast.

    Phone photos of lab reports are often low resolution or dim. Tesseract does
    markedly better on a larger, contrast-normalised greyscale image, so this
    runs before recognition and degrades gracefully if the bytes are unusable.
    """
    info: dict[str, Any] = {}
    try:
        normalised, decode_meta = normalise_image(image_bytes)
        info.update(decode_meta)
        with Image.open(io.BytesIO(normalised)) as img:
            original = img.size
            work = img.convert("L")
            # Upscale small images so glyphs reach a height Tesseract can read.
            target_width = 1800
            if work.width < target_width:
                scale = target_width / work.width
                work = work.resize(
                    (target_width, max(1, int(work.height * scale))),
                    Image.LANCZOS,
                )
            work = ImageOps.autocontrast(work, cutoff=1)
            buf = io.BytesIO()
            work.save(buf, format="PNG")
            info["preprocessed"] = True
            info["original_size"] = list(original)
            info["ocr_size"] = list(work.size)
            return buf.getvalue(), info
    except Exception as exc:  # noqa: BLE001 - fall back to the raw bytes
        info["preprocessed"] = False
        info["preprocess_error"] = str(exc)[:120]
        return image_bytes, info


def _text_from_tsv(tsv: str) -> str:
    """Rebuild Tesseract's plain-text layout from its TSV output.

    The TSV rows carry the words plus the block/paragraph/line they belong to,
    which is everything needed to reassemble the text the `txt` output would
    have produced: words joined by a space within a line, lines separated by a
    newline, and a blank line between paragraphs.
    """
    rows = [r.split("\t") for r in tsv.splitlines()]
    if not rows:
        return ""
    header = rows[0]
    try:
        text_i = header.index("text")
        line_i = header.index("line_num")
        block_i = header.index("block_num")
        par_i = header.index("par_num")
    except ValueError:
        return ""
    needed = max(text_i, line_i, block_i, par_i)

    lines: dict[tuple[str, str, str], list[str]] = {}
    for row in rows[1:]:
        if len(row) <= needed:
            continue
        word = row[text_i].strip()
        if word:
            lines.setdefault((row[block_i], row[par_i], row[line_i]), []).append(word)

    out: list[str] = []
    last_par: tuple[str, str] | None = None
    for key in sorted(lines, key=lambda k: (int(k[0]), int(k[1]), int(k[2]))):
        par = (key[0], key[1])
        if last_par is not None and par != last_par:
            out.append("")
        out.append(" ".join(lines[key]))
        last_par = par
    return "\n".join(out)


def ocr_text(image_bytes: bytes, lang: str = "eng") -> tuple[str, dict[str, Any]]:
    """Run Tesseract OCR. Returns (text, metadata).

    Tesseract is invoked once, with `tsv` output, and the plain text is rebuilt
    from the TSV rows. The obvious implementation runs it twice -- once for text
    and once for confidence -- which doubles the cost of the slowest step in an
    image request. The reconstruction is checked against Tesseract's own `txt`
    output in the tests, so the two stay equivalent.
    """
    meta: dict[str, Any] = {"engine": "tesseract", "available": bool(TESSERACT), "confidence": None}
    if not TESSERACT:
        meta["error"] = "tesseract binary not found on PATH"
        return "", meta

    prepared, pre_meta = _preprocess_for_ocr(image_bytes)
    meta.update(pre_meta)

    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "upload.png"
        with open(img_path, "wb") as fh:
            fh.write(prepared)
        try:
            proc = subprocess.run(
                [TESSERACT, str(img_path), "stdout", "-l", lang, "--psm", "6", "tsv"],
                check=True, capture_output=True, timeout=60,
            )
        except subprocess.CalledProcessError as exc:
            meta["error"] = f"tesseract failed: {exc.stderr.decode(errors='ignore')[:200]}"
            return "", meta
        except subprocess.TimeoutExpired:
            meta["error"] = "tesseract timed out"
            return "", meta

        tsv = proc.stdout.decode(errors="ignore")
        text = _text_from_tsv(tsv)
        meta["char_count"] = len(text)
        meta["word_count"] = len(text.split())

        # Mean word confidence, read from the same TSV. The column order is taken
        # from the header rather than assumed: `conf` is column 10 in Tesseract 5,
        # and reading the wrong index silently reports word text as confidence.
        rows = tsv.splitlines()
        confs: list[float] = []
        if rows:
            header = rows[0].split("\t")
            if "conf" in header:
                idx = header.index("conf")
                for line in rows[1:]:
                    parts = line.split("\t")
                    if len(parts) > idx and parts[idx] not in ("", "-1"):
                        try:
                            value = float(parts[idx])
                        except ValueError:
                            continue
                        if 0.0 <= value <= 100.0:
                            confs.append(value)
        if confs:
            meta["confidence"] = round(sum(confs) / len(confs), 1)
            meta["word_count_scored"] = len(confs)
    return text.strip(), meta


def analyse_image(image_bytes: bytes, lang: str = "eng", modality: str | None = None) -> dict[str, Any]:
    """Full image analysis: OCR text, visual features and a trained prediction.

    The upload is normalised to RGB PNG first, so any format Pillow (plus the
    optional HEIF/PDF plugins) can read is analysed the same way. Formats that
    cannot be decoded produce a clear `ok: False` rather than a traceback.

    `modality` selects the trained classifier ("skin" or "eye"). It is left to
    the caller because only the person uploading knows what the photo shows; when
    it is None both models run and both results are returned rather than one
    being picked by confidence, which would mislabel a rash as a cataract.
    """
    try:
        normalised, decode_meta = normalise_image(image_bytes)
    except Exception as exc:  # noqa: BLE001 - a bad upload must not 500
        fmt = detect_format(image_bytes) or "unknown"
        return {
            "ok": False,
            "error": f"Could not read image ({fmt}): {exc}",
            "detected_format": fmt,
            "supported_formats": sorted(SUPPORTED_FORMATS),
        }

    try:
        img = Image.open(io.BytesIO(normalised))
        img.load()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Could not read image: {exc}"}

    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    features = analyse_pixels(img)
    text, ocr_meta = ocr_text(normalised, lang=lang)

    # Trained classifier over the labelled skin/eye datasets. Absent when no
    # model has been trained, so the rest of the response is unaffected.
    try:
        from . import image_model

        prediction = image_model.predict(image_bytes, modality=modality)
    except Exception as exc:  # noqa: BLE001 - vision must never fail on this
        prediction = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        "image": {
            "width": features.width,
            "height": features.height,
            "format": decode_meta.get("detected_format"),
            "mode": img.mode,
            **{k: v for k, v in decode_meta.items() if k != "detected_format"},
        },
        "visual_features": {
            "mean_rgb": features.mean_rgb,
            "dominant_hue": features.dominant_hue,
            "redness_ratio": features.redness_ratio,
            "yellowness_ratio": features.yellowness_ratio,
            "dark_ratio": features.dark_ratio,
            "pallor_ratio": features.pallor_ratio,
            "saturation_mean": features.saturation_mean,
            "brightness_mean": features.brightness_mean,
            "edge_density": features.edge_density,
            "colour_variance": features.colour_variance,
            "is_likely_document": features.is_likely_document,
        },
        "observations": features.observations,
        "synthetic_symptoms": features.synthetic_symptoms,
        "ocr": {"text": text, **ocr_meta},
        "model_prediction": prediction,
    }
