"""Tests for the external dataset importer.

The importer turns a wide disease x symptom matrix into labelled training rows.
What matters is that it maps only what it can justify, drops what it cannot, and
never invents a label outside the knowledge base's vocabulary.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import import_symptom_dataset as imp  # noqa: E402

from app.kb import KB  # noqa: E402


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.fixture
def out_paths(tmp_path, monkeypatch):
    """Keep the importer's outputs inside the test's temp directory."""
    train = tmp_path / "training"
    ext = tmp_path / "external"
    monkeypatch.setattr(imp, "TRAIN_DIR", train)
    monkeypatch.setattr(imp, "REPORT_PATH", ext / "report.json")
    monkeypatch.setattr(imp, "RATES_PATH", ext / "rates.json")
    return train, ext


# ---- alias integrity ---------------------------------------------------------
def test_every_disease_alias_points_at_a_real_kb_disease():
    bad = {k: v for k, v in imp.DISEASE_ALIASES.items() if v not in KB.diseases}
    assert not bad, f"aliases point at unknown diseases: {bad}"


def test_every_symptom_alias_points_at_a_real_kb_symptom():
    bad = {k: v for k, v in imp.SYMPTOM_ALIASES.items() if v not in KB.symptoms}
    assert not bad, f"aliases point at unknown symptoms: {bad}"


# ---- mapping -----------------------------------------------------------------
def test_disease_maps_by_id_name_and_alias():
    index = imp.build_disease_index()
    assert imp.map_disease("influenza", index) == "influenza"
    assert imp.map_disease("Influenza (Flu)", index) == "influenza"
    assert imp.map_disease("flu", index) == "influenza"
    assert imp.map_disease("a condition that does not exist", index) is None


def test_symptom_maps_by_column_name():
    index = imp.build_symptom_index()
    assert imp.map_symptom("fever", index) == "fever"
    assert imp.map_symptom("shortness_of_breath", index) == "shortness_of_breath"
    assert imp.map_symptom("some_unmapped_column", index) is None


def test_normalisation_is_case_and_punctuation_insensitive():
    assert imp.norm("Heart Failure") == imp.norm("heart_failure")
    assert imp.norm("otitis externa (swimmer's ear)") == imp.norm("otitis externa (swimmers ear)")


# ---- import ------------------------------------------------------------------
def test_import_writes_only_kb_mappable_rows(tmp_path, out_paths):
    train, _ = out_paths
    csv_path = _write_csv(
        tmp_path / "d.csv",
        ["diseases", "fever", "cough", "some_unknown_symptom"],
        [
            ["flu", "1", "0", "0"],
            ["a condition that does not exist", "1", "1", "0"],
        ],
    )
    report = imp.import_dataset(csv_path)
    assert report["examples_written"] == 1
    assert report["rows_skipped_unmapped_disease"] == 1
    assert report["diseases_in_kb_matched"] == 1
    assert (train / "mendeley_2023.jsonl").exists()


def test_imported_labels_are_all_in_the_knowledge_base(tmp_path, out_paths):
    train, _ = out_paths
    csv_path = _write_csv(
        tmp_path / "d.csv",
        ["diseases", "fever", "cough"],
        [["flu", "1", "1"], ["pneumonia", "1", "1"], ["unknown disease xyz", "1", "0"]],
    )
    imp.import_dataset(csv_path)
    import json

    for line in (train / "mendeley_2023.jsonl").read_text().splitlines():
        assert json.loads(line)["label"] in KB.diseases


def test_import_collapses_repeated_symptoms(tmp_path, out_paths):
    """Several dataset columns can map to one KB symptom; count it once."""
    train, _ = out_paths
    csv_path = _write_csv(
        tmp_path / "d.csv",
        ["diseases", "hip_pain", "ankle_pain", "wrist_pain"],
        [["osteoarthritis", "1", "1", "1"]],
    )
    imp.import_dataset(csv_path)
    import json

    row = json.loads((train / "mendeley_2023.jsonl").read_text().splitlines()[0])
    assert row["symptoms"] == ["joint_pain"]


def test_import_respects_the_per_disease_cap(tmp_path, out_paths):
    train, _ = out_paths
    rows = [["flu", "1", "0"] for _ in range(10)]
    csv_path = _write_csv(tmp_path / "d.csv", ["diseases", "fever", "cough"], rows)
    report = imp.import_dataset(csv_path, max_per_disease=3)
    assert report["examples_written"] == 3


def test_import_skips_rows_with_no_known_symptom(tmp_path, out_paths):
    csv_path = _write_csv(
        tmp_path / "d.csv",
        ["diseases", "some_unknown_symptom"],
        [["flu", "1"]],
    )
    report = imp.import_dataset(csv_path)
    assert report["examples_written"] == 0


def test_import_records_rates_and_report(tmp_path, out_paths):
    train, ext = out_paths
    csv_path = _write_csv(
        tmp_path / "d.csv",
        ["diseases", "fever", "cough"],
        [["flu", "1", "0"], ["flu", "1", "1"], ["flu", "0", "1"], ["flu", "0", "0"]],
    )
    report = imp.import_dataset(csv_path)
    import json

    rates = json.loads((ext / "rates.json").read_text())
    assert rates["influenza"]["fever"] == 0.5
    assert rates["influenza"]["cough"] == 0.5
    assert json.loads((ext / "report.json").read_text())["rows_read"] == 4
    assert report["examples_path"].endswith("mendeley_2023.jsonl")


def test_imported_examples_carry_a_sub_unit_weight(tmp_path, out_paths):
    """Imported rows must not outweigh observed cases in training."""
    train, _ = out_paths
    csv_path = _write_csv(tmp_path / "d.csv", ["diseases", "fever"], [["flu", "1"]])
    imp.import_dataset(csv_path)
    import json

    row = json.loads((train / "mendeley_2023.jsonl").read_text().splitlines()[0])
    assert row["weight"] < 1.0
    assert row["origin"] == "mendeley_2023"
