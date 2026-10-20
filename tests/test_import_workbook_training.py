"""Tests for turning the workbook into narrative training examples.

The point of this script is that it produces *prose*, not symptom lists, and
only in the knowledge base's own label and symptom vocabulary. A regression that
emitted raw ids or Indic-script synonyms would silently train the classifier on
the wrong signal, so those properties are asserted directly.
"""
from __future__ import annotations

import random

import pytest

from app import dataset
from app.kb import KB
from scripts import import_workbook_training as gen


@pytest.fixture
def has_workbook():
    dataset.reset()
    if not dataset.available():
        pytest.skip("dataset workbook not built")
    yield
    dataset.reset()


def _latin(text: str) -> bool:
    return all(ord(ch) < 0x250 for ch in text)


def test_phrase_never_returns_indic_script():
    rng = random.Random(0)
    for sid in KB.symptoms:
        for _ in range(20):
            assert _latin(gen._phrase(sid, rng)), sid


def test_build_returns_prose_in_kb_vocabulary(has_workbook):
    rows = [r for r in dataset.diseases(in_kb_only=True) if r.get("disease_id") in KB.diseases]
    assert rows, "workbook has no in-KB conditions"
    rng = random.Random(1)
    built = 0
    for row in rows:
        out = gen.build(row, rng)
        if not out:
            continue
        text, symptoms = out
        built += 1
        assert isinstance(text, str) and text.endswith(".")
        assert symptoms, row.get("disease_id")
        assert all(s in KB.symptoms for s in symptoms)
        # Raw ids carry underscores; prose must not leak them.
        assert "_" not in text
        assert _latin(text), text
    assert built >= 50


def test_generate_is_deterministic(has_workbook):
    a = gen.generate(3, seed=5)
    b = gen.generate(3, seed=5)
    assert a == b
    assert a, "generator produced nothing"


def test_generate_labels_are_valid_and_balanced(has_workbook):
    rows = gen.generate(4, seed=2)
    assert rows
    for r in rows:
        assert r["label"] in KB.diseases
        assert r["origin"] == "workbook"
        assert r["weight"] == 1.0
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    assert min(counts.values()) >= 3


def test_main_refuses_without_workbook(monkeypatch, tmp_path):
    monkeypatch.setattr(gen.dataset, "available", lambda path=None: False)
    monkeypatch.setattr(
        "sys.argv", ["import_workbook_training.py", "--out", str(tmp_path / "x.jsonl")]
    )
    assert gen.main() == 2
    assert not (tmp_path / "x.jsonl").exists()
