"""Contract tests for the external-source fetcher.

`scripts/fetch_sources.py` records provenance for every dataset Dr. Aishu is
built from, so the shape of that manifest is a contract: every source must
declare a licence (even if the answer is "unknown"), and every handler a source
names must exist. These tests read the source table directly rather than the
downloaded manifest, so they hold on a fresh clone with nothing fetched yet.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def fetch_sources():
    spec = importlib.util.spec_from_file_location(
        "fetch_sources", ROOT / "scripts" / "fetch_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_source_declares_a_licence(fetch_sources):
    # An empty or missing licence is the failure this guards: it would let a
    # source be reused without anyone having checked its terms.
    for src in fetch_sources.SOURCES:
        assert src.get("license"), f"{src['id']} has no licence recorded"


def test_every_kind_has_a_handler(fetch_sources):
    known = set(fetch_sources.HANDLERS) | {"note"}
    for src in fetch_sources.SOURCES:
        assert src["kind"] in known, f"{src['id']} uses unhandled kind {src['kind']}"


def test_source_ids_are_unique(fetch_sources):
    ids = [s["id"] for s in fetch_sources.SOURCES]
    assert len(ids) == len(set(ids))


def test_requested_sources_are_registered(fetch_sources):
    ids = {s["id"] for s in fetch_sources.SOURCES}
    assert {
        "deeplesion_metadata",
        "pmc_ids",
        "pmc_ml_disease_diagnosis_review",
        "mendeley_disease_symptoms_2023",
        "google_health_ai",
    } <= ids


def test_deeplesion_does_not_fetch_the_image_archive(fetch_sources):
    """The 238 GB image set must stay out of the fetch list."""
    src = next(s for s in fetch_sources.SOURCES if s["id"] == "deeplesion_metadata")
    names = [name for _, name in src["files"]]
    assert "DL_info.csv" in names
    assert not any("Images_png" in n or "Key_slices" in n or n.endswith(".zip") for n in names)


def test_reference_only_sources_have_no_filename(fetch_sources):
    for src in fetch_sources.SOURCES:
        if src["kind"] == "note":
            assert not src.get("filename"), f"{src['id']} is a note but names a file"


def test_box_shared_source_pins_a_token(fetch_sources):
    src = next(s for s in fetch_sources.SOURCES if s["kind"] == "box_shared")
    assert src["shared_name"]
    assert src["files"]


def test_manifest_shape_when_present(fetch_sources):
    manifest = ROOT / "data" / "external" / "sources" / "sources.json"
    if not manifest.exists():
        pytest.skip("manifest not fetched yet")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert "sources" in data
    for sid, entry in data["sources"].items():
        assert entry.get("license"), f"{sid} fetched without a licence recorded"
        assert entry.get("status")
