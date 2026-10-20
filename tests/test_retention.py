"""Tests for the data-retention policy.

Two promises have to hold at once:

* the default deployment keeps narrative data only when the user consents; and
* a deployment configured to keep everything still records what each user
  actually chose, so a policy-retained row is never mislabelled as consented.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from app import store


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), (180, 90, 90)).save(buf, format="PNG")
    return buf.getvalue()


def _result(label: str = "influenza") -> dict:
    return {
        "extracted": {"symptoms": [{"id": "fever"}, {"id": "muscle_ache"}]},
        "severity": {"score": 4, "triage": "routine"},
        "diagnoses": [{"id": label, "confidence": 0.7}],
    }


@pytest.fixture
def retain_all(monkeypatch):
    monkeypatch.setattr(store, "RETENTION", "all")


@pytest.fixture
def retain_consented(monkeypatch):
    monkeypatch.setattr(store, "RETENTION", "consented")


# ---- default policy ----------------------------------------------------------
def test_default_policy_drops_narrative_without_consent(retain_consented):
    out = store.save_case("high fever for 3 days", _result(), consent_share=False)
    assert out["stored_for_training"] is False
    assert out["consented"] is False
    assert out["retained_under_policy"] is False


def test_default_policy_keeps_narrative_with_consent(retain_consented):
    out = store.save_case("high fever for 3 days", _result(), consent_share=True)
    assert out["stored_for_training"] is True
    assert out["consented"] is True
    assert out["retained_under_policy"] is False


def test_default_policy_does_not_write_upload_without_consent(retain_consented):
    out = store.save_upload(_png(), case_id=None, content_type="image/png",
                            consent_share=False, retain=False)
    assert out["stored"] is False
    assert out["reason"] == "no consent"


# ---- retention=all -----------------------------------------------------------
def test_retain_all_keeps_narrative_without_consent(retain_all):
    out = store.save_case("high fever for 3 days", _result(), consent_share=False)
    assert out["stored_for_training"] is True
    # The user did not consent, and the record must say so even though it was kept.
    assert out["consented"] is False
    assert out["retained_under_policy"] is True


def test_retain_all_still_scrubs_identifiers(retain_all):
    store.save_case(
        "My name is Priya Sharma, call 9876543210, fever for 3 days",
        _result(), consent_share=False,
    )
    rows = store.training_examples()
    text = " ".join(e["text"] for e in rows)
    assert "Priya" not in text
    assert "9876543210" not in text
    assert "fever" in text


def test_retain_all_writes_upload_but_records_no_consent(retain_all):
    out = store.save_upload(_png(), case_id=None, content_type="image/png",
                            consent_share=False, retain=True)
    assert out["stored"] is True
    assert out["consented"] is False


def test_retain_all_makes_cases_available_to_training(retain_all):
    store.save_case("high fever and body aches", _result(), consent_share=False)
    examples = store.training_examples()
    assert examples
    assert examples[0]["label"] == "influenza"
    assert "fever" in examples[0]["text"]


# ---- provenance reporting ----------------------------------------------------
def test_retention_is_reported_in_stats(retain_all):
    stats = store.stats()
    assert stats["retention"] == "all"
    assert "cases_with_text" in stats


@pytest.mark.parametrize("value,expected", [
    ("all", "all"),
    ("ALL", "all"),
    ("  all  ", "all"),
    ("consented", "consented"),
    ("", "consented"),
    (None, "consented"),
    ("everything", "consented"),
])
def test_parse_retention_only_accepts_known_policies(value, expected):
    """A typo must not widen retention; it falls back to the stricter default."""
    assert store.parse_retention(value) == expected


def test_retains_all_helper_reflects_policy(monkeypatch):
    monkeypatch.setattr(store, "RETENTION", "all")
    assert store.retains_all() is True
    monkeypatch.setattr(store, "RETENTION", "consented")
    assert store.retains_all() is False
