"""Tests for the full pharmacy product catalogue.

The product index is a search layer: it answers "does this product exist, what is
it called and what does it cost". It carries no clinical data, so the tests that
matter most are the ones proving it cannot influence which medicine is
suggested, and that its reported size is the size it actually has.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.kb import KB

INDEX = Path(__file__).resolve().parent.parent / "app" / "data" / "medicine_index.json"


@pytest.fixture(scope="module")
def raw():
    return json.loads(INDEX.read_text())


def test_index_reports_the_size_it_holds(raw):
    """The advertised count must match the rows actually stored.

    A catalogue feature is easy to inflate by quoting the row count while
    storing a sample. This pins the two together.
    """
    stored = sum(len(rows) for rows in raw["molecules"].values())
    assert raw["count"] == stored, "reported count must equal stored rows"
    assert raw["count"] > 200_000, "the full catalogue should be indexed"


def test_every_molecule_row_has_the_documented_columns(raw):
    columns = raw["columns"]
    for molecule, rows in raw["molecules"].items():
        for row in rows:
            assert len(row) == len(columns), f"{molecule} row width mismatch"
            assert row[0], f"{molecule} has a row with no product name"


def test_rows_are_pre_sorted_single_ingredient_first(raw):
    """Search relies on the file being sorted; if that breaks, ranking breaks.

    It must not re-sort at request time, so the ordering has to hold on disk.
    """
    for molecule, rows in raw["molecules"].items():
        keys = [(r[6], r[5], r[3] if r[3] is not None else 1e9) for r in rows]
        assert keys == sorted(keys), f"{molecule} rows are not best-first"


def test_search_finds_a_generic_by_molecule():
    results = KB.search_products("paracetamol", limit=5)
    assert results, "paracetamol should be in the catalogue"
    assert all(r["molecule"] == "paracetamol" for r in results[:3])


def test_search_prefers_the_plain_product_over_a_combination():
    """Someone typing "paracetamol" wants paracetamol, not a cold remedy.

    The catalogue lists combination products first by insertion order, so this
    guards the ranking that pushes single-ingredient rows ahead.
    """
    results = KB.search_products("paracetamol", limit=3)
    assert results
    assert results[0]["combination"] == 0, "a combination led the results"


def test_search_ranks_exact_molecule_above_substring():
    """A flat substring scan puts "pindolol" above "Dolo" for the query "dolo"."""
    results = KB.search_products("dolo", limit=5)
    assert results
    assert not any(r["molecule"] == "pindolol" for r in results[:3])


def test_whole_word_brand_beats_a_cheaper_partial_match():
    """Typing a brand name must find that brand, not a cheaper near-namesake.

    "dolo" is a whole word of "Dolo 650 Tablet" but only a prefix of "Doloact",
    a diclofenac injection. Price sorts within a tier, so without the whole-word
    tier the injection leads and the paracetamol brand the user meant sits
    eleventh.
    """
    results = KB.search_products("dolo", limit=4)
    names = [r["name"] for r in results]
    assert any(n.lower().startswith("dolo ") for n in names), \
        f"no whole-word Dolo product in the top 4: {names}"


def test_search_is_case_and_punctuation_insensitive():
    lower = KB.search_products("azithral", limit=3)
    upper = KB.search_products("AZITHRAL", limit=3)
    assert [r["name"] for r in lower] == [r["name"] for r in upper]


def test_short_query_returns_nothing():
    """A one-character query would match most of the catalogue."""
    assert KB.search_products("a", limit=5) == []
    assert KB.search_products("", limit=5) == []


def test_unknown_query_returns_nothing():
    assert KB.search_products("zzzznotadrug", limit=5) == []


def test_search_can_exclude_combinations():
    results = KB.search_products("paracetamol", limit=5, include_combinations=False)
    assert results
    assert all(r["combination"] == 0 for r in results)


def test_curated_medicines_are_linked_to_their_catalogue_rows():
    """A search hit for a curated molecule must carry its curated id.

    That link is what lets the UI attach safe, curated guidance to a raw
    catalogue row instead of showing an unexplained product name.
    """
    results = KB.search_products("paracetamol", limit=5)
    assert results[0]["curated_id"] == "paracetamol"


def test_uncurated_molecule_has_no_curated_id():
    """A real molecule with no curated entry must not claim one."""
    results = KB.search_products("pantoprazole", limit=3)
    assert results
    for r in results:
        if r["molecule"] not in KB.medicines:
            assert r["curated_id"] is None


def test_product_index_is_lazy_and_cached():
    """The 22 MB file must not be parsed until search needs it, then only once."""
    fresh = type(KB)()
    assert fresh._product_index is None, "index should not load at construction"
    first = fresh.product_index
    assert fresh._product_index is not None
    assert fresh.product_index is first, "second read should hit the cache"


def test_product_metadata_reports_provenance():
    meta = KB.product_meta
    assert meta["count"] > 200_000
    assert meta["molecule_count"] > 1_000
    assert "github" in meta["source"]
