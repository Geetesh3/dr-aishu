"""Upload persistence: consented images must be stored, and nothing else.

These tests use the throwaway database and upload directory from the shared
`isolated_db` fixture, so they can assert on real files on disk.
"""
from __future__ import annotations

import hashlib

from app import pipeline, store

# Smallest valid PNG header plus padding; the vision layer is not exercised here.
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def test_upload_is_not_written_without_consent():
    out = store.save_upload(PNG, consent_share=False, content_type="image/png")
    assert out["stored"] is False
    assert out["reason"] == "no consent"
    assert store.stats()["uploads_total"] == 0
    assert not store.UPLOAD_DIR.exists() or not any(store.UPLOAD_DIR.iterdir())


def test_upload_is_written_when_consent_is_given():
    out = store.save_upload(PNG, content_type="image/png", ocr_text="Haemoglobin 8.1 g/dL")
    assert out["stored"] is True
    stored = store.UPLOAD_DIR / out["path"]
    assert stored.is_file()
    assert stored.read_bytes() == PNG
    assert store.stats()["uploads_total"] == 1


def test_stored_filename_does_not_leak_the_original_name():
    """The caller's filename is discarded; it often carries the patient's name."""
    out = store.save_upload(PNG, content_type="image/png")
    assert "priya" not in out["path"].lower()
    assert out["path"].endswith(".png")


def test_ocr_text_is_scrubbed_before_it_is_stored():
    out = store.save_upload(
        PNG, content_type="image/png",
        ocr_text="Patient: Ramesh Kumar, call 9876543210, Hb 8.1 g/dL",
    )
    assert out["redactions"] >= 1
    with store._lock, store._connect() as conn:
        row = conn.execute("SELECT ocr_text FROM uploads WHERE id = ?", (out["upload_id"],)).fetchone()
    assert "Ramesh" not in row["ocr_text"]
    assert "9876543210" not in row["ocr_text"]
    assert "8.1" in row["ocr_text"], "clinical values must survive scrubbing"


def test_image_bytes_are_hashed_for_deduplication():
    out = store.save_upload(PNG, content_type="image/png")
    with store._lock, store._connect() as conn:
        row = conn.execute("SELECT sha256 FROM uploads WHERE id = ?", (out["upload_id"],)).fetchone()
    assert row["sha256"] == hashlib.sha256(PNG).hexdigest()


def test_pipeline_stores_a_consented_image_against_its_case():
    """The end-to-end path the HTTP handler uses."""
    result = pipeline.analyse(
        text="photograph of a lab report showing high fever workup",
        image_bytes=PNG,
        image_content_type="image/png",
        consent_share=True,
        source="test",
    )
    assert result["upload"]["stored"] is True
    assert result["case"]["stored_for_training"] is True
    case_id = result["case"]["case_id"]
    with store._lock, store._connect() as conn:
        row = conn.execute("SELECT case_id FROM uploads").fetchone()
    assert row["case_id"] == case_id, "the upload must be linked to its label"


def test_pipeline_keeps_no_image_without_consent():
    result = pipeline.analyse(
        text="high fever and body aches for 3 days",
        image_bytes=PNG,
        image_content_type="image/png",
        consent_share=False,
        source="test",
    )
    assert "upload" not in result
    assert store.stats()["uploads_total"] == 0


def test_upload_examples_join_the_case_label():
    result = pipeline.analyse(
        text="throbbing one sided headache with nausea and light sensitivity",
        image_bytes=PNG,
        image_content_type="image/png",
        consent_share=True,
        source="test",
    )
    # The OCR text of this synthetic image is empty, so no example is produced;
    # seed one directly to check the join.
    store.save_upload(
        PNG, case_id=result["case"]["case_id"], content_type="image/png",
        ocr_text="throbbing one sided headache with nausea and light sensitivity",
    )
    examples = store.upload_examples()
    assert examples, "a labelled upload with text should yield an example"
    assert all(e["origin"] == "upload" for e in examples)
    label = result["diagnoses"][0]["id"]
    assert {e["label"] for e in examples} == {label}, "label must come from the case"


def test_deleting_a_case_removes_its_upload_files():
    result = pipeline.analyse(
        text="high fever and body aches for 3 days",
        image_bytes=PNG,
        image_content_type="image/png",
        consent_share=True,
        source="test",
    )
    case_id = result["case"]["case_id"]
    path = store.UPLOAD_DIR / result["upload"]["path"]
    assert path.is_file()

    assert store.delete_case(case_id) is True
    assert not path.exists(), "erasure must delete the bytes, not just the row"
    assert store.stats()["uploads_total"] == 0


def test_purge_all_clears_the_upload_directory():
    store.save_upload(PNG, content_type="image/png")
    store.save_upload(PNG, content_type="image/png")
    out = store.purge_all()
    assert out["uploads_deleted"] == 2
    assert not any(store.UPLOAD_DIR.iterdir())


def test_upload_stats_reports_size_and_text_coverage():
    store.save_upload(PNG, content_type="image/png", ocr_text="fever and cough")
    store.save_upload(PNG, content_type="image/png")
    stats = store.upload_stats()
    assert stats["uploads_total"] == 2
    assert stats["uploads_bytes"] == len(PNG) * 2
    assert stats["uploads_with_text"] == 1


def test_empty_upload_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        store.save_upload(b"", content_type="image/png")
