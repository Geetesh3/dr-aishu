"""Persistence for Dr. Aishu.

Stores anonymised case records and clinician/user feedback so the model can be
retrained on real usage. Three rules govern everything here:

1. Structured clinical signals (symptom ids, diagnosis ids, severity, triage,
   duration, demographics, image presence) are always recorded -- they carry no
   direct identifier and they are the bulk of the training signal.
2. Narrative data (free text and the uploaded image itself) is recorded
   according to the retention policy. The default requires explicit consent per
   request; `DR_AISHU_RETENTION=all` keeps it for every request instead. The
   user's actual choice is always recorded separately in `consent_share`, so a
   row retained under the policy is never misreported as consented.
3. Free text is scrubbed of direct identifiers before it touches the database,
   whatever the retention policy says.

Scrubbing is best-effort, not a guarantee, so retention remains the primary
control. The database file lives outside the served static directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The database location is configurable so hosts with a persistent volume can
# point at it. On ephemeral filesystems the data resets on redeploy, which is
# acceptable for the demo tiers but should be a mounted disk in production.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR = Path(os.environ.get("DR_AISHU_DATA_DIR", _DEFAULT_DATA_DIR))
DB_PATH = DATA_DIR / "dr_aishu.db"
# Consented uploads live beside the database, never under the served static
# directory. They are training material, not web assets.
UPLOAD_DIR = DATA_DIR / "uploads"

# How narrative data (free text and uploaded images) is retained.
#   "consented" -- only when the caller opts in for that request (default).
#   "all"       -- for every request, still scrubbed of identifiers first.
def parse_retention(value: str | None) -> str:
    """Map a configured retention value onto one of the supported policies.

    An unrecognised value falls back to the stricter default rather than being
    accepted, so a typo cannot silently widen what is stored.
    """
    policy = (value or "").strip().lower()
    return policy if policy in ("consented", "all") else "consented"


RETENTION = parse_retention(os.environ.get("DR_AISHU_RETENTION"))


def retains_all() -> bool:
    """True when the deployment keeps narrative data without per-request consent."""
    return RETENTION == "all"


_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    text            TEXT NOT NULL,
    age             REAL,
    sex             TEXT,
    pregnant        INTEGER DEFAULT 0,
    chronic         TEXT,
    allergies       TEXT,
    symptoms        TEXT,
    duration_days   REAL,
    temperature_c   REAL,
    severity_score  INTEGER,
    triage          TEXT,
    top_diagnosis   TEXT,
    confidence      REAL,
    diagnoses       TEXT,
    source          TEXT,
    image_features  TEXT,
    consent_share   INTEGER DEFAULT 0,
    model_version   TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id                TEXT PRIMARY KEY,
    case_id           TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    correct_diagnosis TEXT,
    is_correct        INTEGER,
    rating            INTEGER,
    comment           TEXT,
    source            TEXT,
    FOREIGN KEY (case_id) REFERENCES cases (id)
);

CREATE INDEX IF NOT EXISTS idx_cases_created ON cases (created_at);
CREATE INDEX IF NOT EXISTS idx_cases_top_dx  ON cases (top_diagnosis);
CREATE INDEX IF NOT EXISTS idx_fb_case       ON feedback (case_id);

-- One row per analysis request, whether or not the user consented to storing
-- their free text. It holds structured clinical signals only (symptom ids,
-- diagnosis ids, severity, triage, duration) and never the narrative, so it
-- supports analytics and training without weakening the privacy promise the UI
-- makes: free text is still retained only when consent_share is set.
CREATE TABLE IF NOT EXISTS search_events (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    case_id         TEXT,
    source          TEXT NOT NULL,
    symptoms        TEXT,
    symptom_count   INTEGER DEFAULT 0,
    top_diagnosis   TEXT,
    diagnoses       TEXT,
    confidence      REAL,
    severity_score  INTEGER,
    triage          TEXT,
    duration_days   REAL,
    temperature_c   REAL,
    age             REAL,
    sex             TEXT,
    had_image       INTEGER DEFAULT 0,
    consent_share   INTEGER DEFAULT 0,
    is_demo         INTEGER DEFAULT 0,
    model_version   TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_created ON search_events (created_at);
CREATE INDEX IF NOT EXISTS idx_events_top_dx  ON search_events (top_diagnosis);
CREATE INDEX IF NOT EXISTS idx_events_demo    ON search_events (is_demo);

-- One row per consented image upload. The bytes are written to UPLOAD_DIR and
-- only the relative path is stored here, so the database stays small and the
-- images can be shipped to a training run as a folder. `ocr_text` holds the
-- scrubbed text read from the image, which is what the classifier actually
-- learns from; the image itself is kept for vision work and re-OCR.
CREATE TABLE IF NOT EXISTS uploads (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    case_id       TEXT,
    path          TEXT NOT NULL,
    content_type  TEXT,
    bytes         INTEGER,
    sha256        TEXT,
    ocr_text      TEXT,
    source        TEXT,
    consent_share INTEGER DEFAULT 1,
    model_version TEXT
);

CREATE INDEX IF NOT EXISTS idx_uploads_created ON uploads (created_at);
CREATE INDEX IF NOT EXISTS idx_uploads_case    ON uploads (case_id);
"""

# --- PII patterns -------------------------------------------------------------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_LONG_ID = re.compile(r"\b(?:\d[ -]?){9,}\b")
_URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+")
_HANDLE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,}")
_NAME_STRONG = re.compile(
    # Explicit name introducers: whatever follows is a name, whatever its case.
    r"\b(?:my name is|name[:\s]+|patient[:\s]+|pt[:\s]+)\s+"
    r"([A-Za-z][A-Za-z'\-]{1,}(?:\s+[A-Za-z][A-Za-z'\-]{1,}){0,3})",
    re.IGNORECASE,
)
_NAME_WEAK = re.compile(
    # "I am", "I'm", "this is" are ambiguous, so the name itself must look like a
    # proper noun (capitalised) even though the introducer may appear in any case.
    r"(?i:\b(?:i am|i'm|this is))\s+"
    r"([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})",
)
_ADDRESS_HINT = re.compile(
    r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+"
    r"(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|nagar|colony|sector)\b\.?",
    re.IGNORECASE,
)
_PINCODE = re.compile(r"\b(?:pin|zip|postal)\s*(?:code)?[:\s]*\d{5,6}\b", re.IGNORECASE)
_MR_NUMBER = re.compile(r"\b(?:mrn|patient id|uhid|hospital no|reg no)[:\s#]*[\w/-]{4,}\b", re.IGNORECASE)

REDACTION = "[redacted]"


def _redact_name(match: re.Match) -> str:
    """Keep the introducer phrase, redact only the captured name."""
    start = match.start(1) - match.start(0)
    return match.group(0)[:start] + REDACTION


def scrub_pii(text: str) -> tuple[str, int]:
    """Remove direct identifiers from free text.

    Returns the scrubbed text and the count of redactions, so callers can decide
    whether a record is clean enough to keep for training.
    """
    if not text:
        return "", 0
    count = 0
    out = text

    def sub(pattern: re.Pattern, replacement, s: str) -> str:
        nonlocal count
        new, n = pattern.subn(replacement, s)
        count += n
        return new

    out = sub(_EMAIL, REDACTION, out)
    out = sub(_URL, REDACTION, out)
    out = sub(_MR_NUMBER, REDACTION, out)
    out = sub(_PINCODE, REDACTION, out)
    out = sub(_ADDRESS_HINT, REDACTION, out)
    out = sub(_LONG_ID, REDACTION, out)
    out = sub(_PHONE, REDACTION, out)
    out = sub(_HANDLE, REDACTION, out)
    out = sub(_NAME_STRONG, _redact_name, out)
    out = sub(_NAME_WEAK, _redact_name, out)

    out = re.sub(r"\s{2,}", " ", out).strip()
    return out, count


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # WAL with synchronous=NORMAL is the documented pairing: a commit no longer
    # waits on an fsync per write, which is most of the cost of recording one
    # analysis, and the database still survives a process crash. The trade is
    # that a power loss can lose the last few committed rows -- acceptable for
    # usage counters and audit rows, not for anything the user must not lose.
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(SCHEMA)


def save_case(
    text: str,
    result: dict[str, Any],
    consent_share: bool,
    source: str = "web",
    sex: str | None = None,
) -> dict[str, Any]:
    """Persist one analysed case. Returns the stored record summary.

    Structured fields are always written. Free text is kept when the user
    consented or when the deployment's retention policy keeps everything; the
    `consent_share` column records what the *user* chose, so a row retained
    under `DR_AISHU_RETENTION=all` is still distinguishable from one the user
    opted into.
    """
    extracted = result.get("extracted", {}) or {}
    dem = extracted.get("demographics", {}) or {}
    hist = extracted.get("history", {}) or {}
    vit = extracted.get("vitals", {}) or {}
    sev = result.get("severity", {}) or {}
    dxs = result.get("diagnoses", []) or []
    top = dxs[0] if dxs else {}

    case_id = uuid.uuid4().hex[:16]
    created = _now()

    keep_text = consent_share or retains_all()
    if keep_text:
        clean_text, redactions = scrub_pii(text)
    else:
        clean_text, redactions = "", 0

    record = {
        "id": case_id,
        "created_at": created,
        "text": clean_text,
        "age": dem.get("age"),
        "sex": sex,
        "pregnant": 1 if dem.get("pregnant") else 0,
        "chronic": json.dumps(hist.get("chronic_conditions", [])),
        "allergies": json.dumps(hist.get("allergies", [])),
        "symptoms": json.dumps([s.get("id") for s in extracted.get("symptoms", [])]),
        "duration_days": (extracted.get("duration") or {}).get("days"),
        "temperature_c": vit.get("temperature_c"),
        "severity_score": sev.get("score"),
        "triage": sev.get("triage"),
        "top_diagnosis": top.get("id"),
        "confidence": top.get("confidence"),
        "diagnoses": json.dumps([{"id": d.get("id"), "confidence": d.get("confidence")} for d in dxs]),
        "source": source,
        "image_features": json.dumps(
            (result.get("image_analysis") or {}).get("visual_features") or {}
        ),
        "consent_share": 1 if consent_share else 0,
        "model_version": result.get("model_version"),
    }

    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO cases (id, created_at, text, age, sex, pregnant, chronic, allergies,
                   symptoms, duration_days, temperature_c, severity_score, triage, top_diagnosis,
                   confidence, diagnoses, source, image_features, consent_share, model_version)
               VALUES (:id, :created_at, :text, :age, :sex, :pregnant, :chronic, :allergies,
                   :symptoms, :duration_days, :temperature_c, :severity_score, :triage,
                   :top_diagnosis, :confidence, :diagnoses, :source, :image_features,
                   :consent_share, :model_version)""",
            record,
        )

    return {
        "case_id": case_id,
        "stored_for_training": bool(keep_text),
        "retained_under_policy": bool(keep_text and not consent_share),
        "consented": bool(consent_share),
        "redactions": redactions,
        "created_at": created,
    }


_SAFE_EXT = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/webp": ".webp", "image/bmp": ".bmp", "image/tiff": ".tiff",
    "image/gif": ".gif", "image/heic": ".heic", "image/heif": ".heif",
    "image/avif": ".avif", "application/pdf": ".pdf",
}
# Sniffed-format fallbacks, used when the client's content type is generic
# (`application/octet-stream`) or missing. Keyed on the format name that
# `vision.detect_format` returns.
_FORMAT_EXT = {
    "PNG": ".png", "JPEG": ".jpg", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tiff",
    "WEBP": ".webp", "HEIF": ".heic", "AVIF": ".avif", "PDF": ".pdf",
    "ICO": ".ico", "PSD": ".psd", "QOI": ".qoi", "DDS": ".dds", "JP2": ".jp2",
    "FARBFELD": ".ff", "DNG": ".dng",
}


def _upload_extension(content_type: str | None, image_bytes: bytes) -> str:
    """Pick a file extension from the bytes first, then the declared type.

    The declared type is only a fallback because it is client-supplied and often
    wrong; the stored file must have an extension that matches its real format.
    """
    from app import vision

    fmt = vision.detect_format(image_bytes)
    if fmt and fmt in _FORMAT_EXT:
        return _FORMAT_EXT[fmt]
    return _SAFE_EXT.get((content_type or "").lower(), ".bin")


def save_upload(
    image_bytes: bytes,
    *,
    case_id: str | None = None,
    content_type: str | None = None,
    ocr_text: str = "",
    source: str = "web",
    consent_share: bool = True,
    retain: bool = False,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Persist an image upload for training.

    The bytes are written only when the user consented or the deployment retains
    everything (`DR_AISHU_RETENTION=all`); the caller decides which by passing
    `consent_share`/`retain`. The bytes land in UPLOAD_DIR under a random name --
    the original filename is discarded because it frequently contains the
    patient's name -- and the extracted text is scrubbed of identifiers before it
    is stored, exactly as free text is.

    The file write and the row insert are independent: a full disk must not fail
    the analysis the user is waiting for, so a write error is reported in the
    return value rather than raised.
    """
    if not image_bytes:
        raise ValueError("empty upload")
    if not (consent_share or retain):
        return {"stored": False, "reason": "no consent"}

    upload_id = uuid.uuid4().hex[:16]
    ext = _upload_extension(content_type, image_bytes)
    rel_path = f"{upload_id}{ext}"
    target = UPLOAD_DIR / rel_path

    clean_text, redactions = scrub_pii(ocr_text or "")
    digest = hashlib.sha256(image_bytes).hexdigest()

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        # Write to a temp name then rename, so a crash mid-write cannot leave a
        # half-image that a training run would happily read.
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(image_bytes)
        tmp.replace(target)
    except OSError as exc:
        return {"stored": False, "error": f"could not write upload: {exc}"}

    record = {
        "id": upload_id,
        "created_at": _now(),
        "case_id": case_id,
        "path": rel_path,
        "content_type": content_type,
        "bytes": len(image_bytes),
        "sha256": digest,
        "ocr_text": clean_text,
        "source": source,
        "consent_share": 1 if consent_share else 0,
        "model_version": model_version,
    }
    try:
        with _lock, _connect() as conn:
            conn.execute(
                """INSERT INTO uploads (id, created_at, case_id, path, content_type, bytes,
                       sha256, ocr_text, source, consent_share, model_version)
                   VALUES (:id, :created_at, :case_id, :path, :content_type, :bytes,
                       :sha256, :ocr_text, :source, :consent_share, :model_version)""",
                record,
            )
    except sqlite3.Error as exc:
        # Roll the file back too, so an unindexed image cannot accumulate on disk.
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        return {"stored": False, "error": f"could not index upload: {exc}"}

    return {
        "stored": True,
        "upload_id": upload_id,
        "path": rel_path,
        "bytes": len(image_bytes),
        "redactions": redactions,
        "consented": bool(consent_share),
    }


def upload_examples(min_confidence: float = 0.0) -> list[dict[str, Any]]:
    """Labelled examples built from consented uploads.

    An upload contributes its scrubbed OCR text as the document and the case's
    top diagnosis as the label, so a photographed lab report becomes a training
    row without the free-text consent form being involved twice.
    """
    with _lock, _connect() as conn:
        rows = conn.execute(
            """SELECT u.*, c.top_diagnosis AS label, c.confidence AS confidence,
                      c.symptoms AS symptoms
               FROM uploads u JOIN cases c ON c.id = u.case_id
               WHERE c.top_diagnosis IS NOT NULL"""
        ).fetchall()

    examples: list[dict[str, Any]] = []
    for r in rows:
        text = (r["ocr_text"] or "").strip()
        if not text:
            continue
        if r["confidence"] is not None and r["confidence"] < min_confidence:
            continue
        examples.append({
            "text": text,
            "symptoms": json.loads(r["symptoms"] or "[]"),
            "label": r["label"],
            "weight": 1.5,
            "origin": "upload",
        })
    return examples


def upload_stats() -> dict[str, Any]:
    with _lock, _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n, COALESCE(SUM(bytes), 0) AS total,
                      SUM(CASE WHEN ocr_text != '' THEN 1 ELSE 0 END) AS with_text
               FROM uploads"""
        ).fetchone()
    return {
        "uploads_total": row["n"],
        "uploads_bytes": row["total"],
        "uploads_with_text": row["with_text"] or 0,
        "upload_dir": str(UPLOAD_DIR),
    }


def save_feedback(
    case_id: str,
    correct_diagnosis: str | None = None,
    is_correct: bool | None = None,
    rating: int | None = None,
    comment: str | None = None,
    source: str = "web",
) -> dict[str, Any]:
    if rating is not None and not 1 <= rating <= 5:
        raise ValueError("rating must be between 1 and 5")
    clean_comment, redactions = scrub_pii(comment or "")

    fb_id = uuid.uuid4().hex[:16]
    with _lock, _connect() as conn:
        exists = conn.execute("SELECT 1 FROM cases WHERE id = ?", (case_id,)).fetchone()
        if not exists:
            raise KeyError(f"unknown case_id: {case_id}")
        conn.execute(
            """INSERT INTO feedback (id, case_id, created_at, correct_diagnosis, is_correct,
                   rating, comment, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (fb_id, case_id, _now(), correct_diagnosis,
             None if is_correct is None else (1 if is_correct else 0),
             rating, clean_comment, source),
        )
    return {"feedback_id": fb_id, "case_id": case_id, "redactions": redactions}


def training_examples(min_confidence: float = 0.0, include_feedback: bool = True) -> list[dict[str, Any]]:
    """Return labelled (text, label) pairs suitable for supervised training.

    Labels come from the reported top diagnosis. When a user has told us the
    correct diagnosis, that overrides the model's guess.

    Every labelled case is eligible: the free text is present only when it was
    retained (consent or policy), and a case whose text was dropped still
    contributes its canonical symptom ids, exactly as a search event does.
    """
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM cases WHERE top_diagnosis IS NOT NULL"
        ).fetchall()
        fb_rows = conn.execute(
            "SELECT * FROM feedback WHERE correct_diagnosis IS NOT NULL AND correct_diagnosis != ''"
        ).fetchall() if include_feedback else []

    corrections: dict[str, str] = {}
    for fb in fb_rows:
        corrections[fb["case_id"]] = fb["correct_diagnosis"]

    examples: list[dict[str, Any]] = []
    for r in rows:
        label = corrections.get(r["id"], r["top_diagnosis"])
        text = r["text"] or ""
        symptom_ids = json.loads(r["symptoms"] or "[]")
        if not text and not symptom_ids:
            continue
        if r["confidence"] is not None and r["confidence"] < min_confidence and r["id"] not in corrections:
            continue
        examples.append({
            "text": text,
            "symptoms": symptom_ids,
            "label": label,
            "weight": 2.0 if r["id"] in corrections else 1.0,
            "origin": "feedback" if r["id"] in corrections else "case",
        })
    return examples


def event_training_examples(min_confidence: float = 0.0) -> list[dict[str, Any]]:
    """Labelled pairs derived from recorded search events.

    Events carry no free text, so the document is rebuilt from the canonical
    symptom ids plus the numeric observations. This lets searches that were made
    *without* text consent still contribute to the classifier as structured
    signal, which is exactly the data the consent-free design was meant to allow.
    """
    with _lock, _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM search_events
               WHERE top_diagnosis IS NOT NULL AND symptom_count > 0"""
        ).fetchall()

    examples: list[dict[str, Any]] = []
    for r in rows:
        if r["confidence"] is not None and r["confidence"] < min_confidence:
            continue
        symptom_ids = json.loads(r["symptoms"] or "[]")
        if not symptom_ids:
            continue
        bits = [f"sym_{sid}" for sid in symptom_ids]
        if r["temperature_c"] is not None:
            bits.append("fever" if float(r["temperature_c"]) >= 38 else "afebrile")
        if r["duration_days"] is not None:
            days = float(r["duration_days"])
            bits.append("acute" if days <= 3 else "subacute" if days <= 14 else "chronic")
        examples.append({
            "text": " ".join(bits),
            "symptoms": symptom_ids,
            "label": r["top_diagnosis"],
            "weight": 1.0 if r["is_demo"] else 1.5,
            "origin": "demo" if r["is_demo"] else "event",
        })
    return examples


def record_event(
    result: dict[str, Any],
    source: str = "web",
    case_id: str | None = None,
    consent_share: bool = False,
    sex: str | None = None,
    is_demo: bool = False,
) -> str:
    """Record one search as a structured, non-identifying event.

    Called for every analysis, consenting or not. Only symptom/diagnosis ids and
    numeric observations are kept, never the user's own words, so this can power
    the insights dashboard and the training set without touching the free-text
    consent promise.
    """
    extracted = result.get("extracted", {}) or {}
    dem = extracted.get("demographics", {}) or {}
    vit = extracted.get("vitals", {}) or {}
    sev = result.get("severity", {}) or {}
    dxs = result.get("diagnoses", []) or []
    top = dxs[0] if dxs else {}
    symptom_ids = [s.get("id") for s in extracted.get("symptoms", []) if s.get("id")]

    event_id = uuid.uuid4().hex[:16]
    record = {
        "id": event_id,
        "created_at": _now(),
        "case_id": case_id,
        "source": source,
        "symptoms": json.dumps(symptom_ids),
        "symptom_count": len(symptom_ids),
        "top_diagnosis": top.get("id"),
        "diagnoses": json.dumps([d.get("id") for d in dxs if d.get("id")]),
        "confidence": top.get("confidence"),
        "severity_score": sev.get("score"),
        "triage": sev.get("triage"),
        "duration_days": (extracted.get("duration") or {}).get("days"),
        "temperature_c": vit.get("temperature_c"),
        "age": dem.get("age"),
        "sex": sex,
        "had_image": 1 if result.get("image_analysis") else 0,
        "consent_share": 1 if consent_share else 0,
        "is_demo": 1 if is_demo else 0,
        "model_version": result.get("model_version"),
    }

    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO search_events (id, created_at, case_id, source, symptoms,
                   symptom_count, top_diagnosis, diagnoses, confidence, severity_score,
                   triage, duration_days, temperature_c, age, sex, had_image,
                   consent_share, is_demo, model_version)
               VALUES (:id, :created_at, :case_id, :source, :symptoms, :symptom_count,
                   :top_diagnosis, :diagnoses, :confidence, :severity_score, :triage,
                   :duration_days, :temperature_c, :age, :sex, :had_image,
                   :consent_share, :is_demo, :model_version)""",
            record,
        )
    return event_id


def record_analysis(
    text: str,
    result: dict[str, Any],
    *,
    consent_share: bool,
    source: str = "web",
    sex: str | None = None,
    is_demo: bool = False,
) -> dict[str, Any]:
    """Persist one analysis: its case row plus its structured search event.

    The single place that knows an analysis produces both. Callers that ran the
    pipeline themselves -- the HTTP handler and the demo generator -- use this
    rather than wiring the two writes by hand, so neither can forget one.

    Each write is attempted independently: an analytics failure must never cost
    the user their result, and a case-storage failure must not lose the event.
    """
    case: dict[str, Any] = {"stored_for_training": False}
    case_id: str | None = None
    try:
        case = save_case(
            text=text, result=result, consent_share=consent_share, source=source, sex=sex,
        )
        case_id = case.get("case_id")
    except Exception as exc:  # never fail an analysis because logging failed
        case = {"stored_for_training": False, "error": str(exc)}

    event_id: str | None = None
    try:
        event_id = record_event(
            result, source=source, case_id=case_id, consent_share=consent_share,
            sex=sex, is_demo=is_demo,
        )
    except Exception as exc:
        case["event_error"] = str(exc)

    return {"case": case, "case_id": case_id, "event_id": event_id}


def stats() -> dict[str, Any]:
    with _lock, _connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM cases").fetchone()["c"]
        consented = conn.execute("SELECT COUNT(*) c FROM cases WHERE consent_share = 1").fetchone()["c"]
        with_text = conn.execute("SELECT COUNT(*) c FROM cases WHERE text != ''").fetchone()["c"]
        fb = conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]
        corrections = conn.execute(
            "SELECT COUNT(*) c FROM feedback WHERE correct_diagnosis IS NOT NULL AND correct_diagnosis != ''"
        ).fetchone()["c"]
        by_dx = conn.execute(
            """SELECT top_diagnosis, COUNT(*) c FROM cases
               WHERE top_diagnosis IS NOT NULL
               GROUP BY top_diagnosis ORDER BY c DESC LIMIT 12"""
        ).fetchall()
        by_triage = conn.execute(
            "SELECT triage, COUNT(*) c FROM cases GROUP BY triage"
        ).fetchall()
        events = conn.execute("SELECT COUNT(*) c FROM search_events").fetchone()["c"]
        demo_events = conn.execute(
            "SELECT COUNT(*) c FROM search_events WHERE is_demo = 1"
        ).fetchone()["c"]
        uploads = conn.execute("SELECT COUNT(*) c FROM uploads").fetchone()["c"]
    return {
        "cases_total": total,
        "cases_consented": consented,
        "cases_with_text": with_text,
        "feedback_total": fb,
        "corrections": corrections,
        "top_diagnoses": {r["top_diagnosis"]: r["c"] for r in by_dx},
        "by_triage": {r["triage"]: r["c"] for r in by_triage if r["triage"]},
        "search_events_total": events,
        "search_events_demo": demo_events,
        "uploads_total": uploads,
        "retention": RETENTION,
        "db_path": str(DB_PATH),
    }


def delete_case(case_id: str) -> bool:
    """Purge a case, its feedback and its uploads (right-to-erasure support)."""
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM feedback WHERE case_id = ?", (case_id,))
        uploads = conn.execute(
            "SELECT path FROM uploads WHERE case_id = ?", (case_id,)
        ).fetchall()
        conn.execute("DELETE FROM uploads WHERE case_id = ?", (case_id,))
        cur = conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        removed = cur.rowcount > 0
    for row in uploads:
        try:
            (UPLOAD_DIR / row["path"]).unlink(missing_ok=True)
        except OSError:
            pass
    return removed


def purge_all() -> dict[str, int]:
    with _lock, _connect() as conn:
        f = conn.execute("DELETE FROM feedback").rowcount
        c = conn.execute("DELETE FROM cases").rowcount
        u = conn.execute("DELETE FROM uploads").rowcount
    # Erasure is only complete once the bytes are gone, not just the index.
    if UPLOAD_DIR.exists():
        for path in UPLOAD_DIR.iterdir():
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
    return {"cases_deleted": c, "feedback_deleted": f, "uploads_deleted": u}


init_db()
