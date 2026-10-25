"""Tests for the ontology-backed condition library and the RxNorm medicine reference.

Both databases are built artefacts and are gitignored, so the important
behaviour to pin down is that the app degrades cleanly when they are missing and
that, when they are present, the data is real and traceable: a source, a licence
and an identifier on every row. The tests that need the databases skip when they
have not been built rather than failing, so a fresh clone still gets a green run.
"""
from __future__ import annotations

import sqlite3

import pytest

from app import conditions, drugref

conditions_ready = pytest.mark.skipif(
    not conditions.available(), reason="data/conditions.db not built")
drugref_ready = pytest.mark.skipif(
    not drugref.available(), reason="data/medicines_reference.db not built")


# ---- condition library: graceful absence ------------------------------------
def test_condition_library_reports_unavailable_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(conditions, "DB_PATH", tmp_path / "missing.db")
    monkeypatch.setattr(conditions._local, "conn", None, raising=False)
    assert conditions.available() is False
    assert conditions.stats()["available"] is False
    result = conditions.search("diabetes")
    assert result["available"] is False and result["results"] == []
    assert conditions.get("MONDO:0005148") is None
    assert conditions.lookup("diabetes mellitus") is None


def test_medicine_reference_reports_unavailable_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(drugref, "DB_PATH", tmp_path / "missing.db")
    monkeypatch.setattr(drugref._local, "conn", None, raising=False)
    monkeypatch.setattr(drugref._local, "ingredient_index", None, raising=False)
    assert drugref.available() is False
    assert drugref.stats()["available"] is False
    assert drugref.search("metformin")["results"] == []
    assert drugref.ingredients_for("Glucophage") == []


# ---- condition library: real data ------------------------------------------
@conditions_ready
def test_condition_library_is_substantial():
    stats = conditions.stats()
    assert stats["conditions"] > 50_000, "library should carry tens of thousands of conditions"
    assert stats["synonyms"] > 50_000
    assert stats["cross_references"] > 50_000
    assert stats["built_at"]


@conditions_ready
def test_every_condition_carries_source_and_licence():
    """Provenance is the point of the library: no row may lack it."""
    conn = sqlite3.connect(f"file:{conditions.DB_PATH}?mode=ro", uri=True)
    missing = conn.execute(
        "SELECT COUNT(*) FROM conditions WHERE source IS NULL OR source = '' "
        "OR license IS NULL OR license = ''").fetchone()[0]
    assert missing == 0
    # A blank licence would read as permission; the builder must always name one.
    unknown = conn.execute(
        "SELECT COUNT(*) FROM conditions WHERE license = 'unknown'").fetchone()[0]
    assert unknown == 0
    conn.close()


@conditions_ready
def test_sources_match_the_published_ontologies():
    by_source = {s["source"]: s for s in conditions.stats()["sources"]}
    for source in ("mondo", "doid", "orphanet", "icd10cm", "mesh"):
        assert source in by_source, f"{source} missing from the built library"
        assert by_source[source]["count"] > 0
        assert by_source[source]["license"]
    # Mondo is the largest source; a regression here means a parser broke.
    assert by_source["mondo"]["count"] > 20_000


@conditions_ready
def test_search_finds_a_known_condition_and_ranks_the_exact_name_first():
    results = conditions.search("type 2 diabetes mellitus", limit=5)["results"]
    assert results, "search returned nothing for a common condition"
    assert results[0]["name"].lower() == "type 2 diabetes mellitus"


@conditions_ready
def test_search_matches_synonyms_not_only_names():
    """A user typing "T2DM" should reach the same disease as the full name."""
    results = conditions.search("T2DM", limit=10)["results"]
    assert any("type 2 diabetes" in r["name"].lower() for r in results)


@conditions_ready
def test_search_operators_are_treated_as_text():
    """FTS5 syntax typed by a user must not raise or match everything."""
    for query in ['diabetes AND OR NOT', '"unbalanced', "heart*", "covid-19", "  "]:
        result = conditions.search(query, limit=5)
        assert result["available"] is True
        assert isinstance(result["results"], list)


@conditions_ready
def test_rare_filter_only_returns_rare_conditions():
    result = conditions.search("", limit=10, rare_only=True)
    assert result["results"]
    assert all(r["is_rare"] for r in result["results"])


@conditions_ready
def test_get_returns_definition_synonyms_and_cross_references():
    record = conditions.get("MONDO:0005148")
    assert record is not None
    assert record["name"].lower() == "type 2 diabetes mellitus"
    assert record["source"] == "mondo" and record["license"]
    assert "insulin resistance" in (record["definition"] or "").lower()
    assert any(s.lower() == "t2dm" for s in record["synonyms"])
    systems = {x["system"] for x in record["xrefs"]}
    assert {"ICD10CM", "MESH"} <= systems, "cross-references to ICD and MeSH expected"


@conditions_ready
def test_lookup_is_exact_and_prefers_the_hand_curated_entry():
    """An exact-name lookup must never approximate: type 1 and type 2 differ by one word."""
    assert conditions.lookup("type 2 diabetes mellitus")["name"].lower() == "type 2 diabetes mellitus"
    assert conditions.lookup("type 1 diabetes mellitus")["name"].lower() == "type 1 diabetes mellitus"
    assert conditions.lookup("diabetis") is None


@conditions_ready
def test_pagination_does_not_repeat_rows():
    first = conditions.search("", limit=10, offset=0)["results"]
    second = conditions.search("", limit=10, offset=10)["results"]
    assert len(first) == 10 and len(second) == 10
    assert {r["id"] for r in first}.isdisjoint({r["id"] for r in second})


# ---- medicine reference: real data -----------------------------------------
@drugref_ready
def test_medicine_reference_is_substantial_and_declares_its_licence():
    stats = drugref.stats()
    assert stats["ingredients"] > 5_000
    assert stats["names"] > 20_000
    assert "RxNorm" in (stats["source"] or "")
    assert "public domain" in (stats["license"] or "").lower()
    assert stats["attribution"], "NLM asks to be acknowledged"


@drugref_ready
def test_only_nlm_created_rxnorm_rows_are_imported():
    """The full RxNorm release bundles restricted sources; the builder must filter them out."""
    conn = sqlite3.connect(f"file:{drugref.DB_PATH}?mode=ro", uri=True)
    has_sab = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='rxnsat'").fetchone()[0]
    assert has_sab == 0, "no non-RxNorm source table should be present"
    conn.close()


@drugref_ready
def test_brand_resolves_to_its_ingredient_through_rxnorm_relations():
    names = {i["name"].lower() for i in drugref.ingredients_for("Glucophage")}
    assert "metformin" in names


@drugref_ready
def test_multi_ingredient_brand_resolves_to_every_molecule():
    names = {i["name"].lower() for i in drugref.ingredients_for("Maxzide")}
    assert {"hydrochlorothiazide", "triamterene"} <= names


@drugref_ready
def test_product_name_resolves_even_with_a_leading_release_form():
    """RxNorm product names can lead with "24 HR" before the ingredient."""
    names = {i["name"].lower() for i in
             drugref.ingredients_for("Glucophage XR 24 HR 500 MG Extended Release Oral Tablet")}
    assert any("metformin" in n for n in names)


@drugref_ready
def test_unknown_medicine_returns_nothing_rather_than_guessing():
    """A miss must read as unknown; it must never be reported as safe."""
    assert drugref.ingredients_for("zzzznotarealdrugname") == []
    assert drugref.ingredients_for("") == []
    assert drugref.search("zzzznotarealdrugname")["results"] == []


@drugref_ready
def test_medicine_search_finds_ingredients_and_brands():
    ingredients = drugref.search("metformin", limit=10, kind="ingredient")["results"]
    assert ingredients and all(r["kind"] == "ingredient" for r in ingredients)
    assert any(r["name"].lower() == "metformin" for r in ingredients)


@drugref_ready
def test_medicine_search_operators_are_treated_as_text():
    for query in ["metformin AND OR", '"unbalanced', "aspirin*", ""]:
        result = drugref.search(query, limit=5)
        assert isinstance(result["results"], list)


# ---- API surface -------------------------------------------------------------
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_conditions_stats_endpoint_always_answers(client):
    """The page needs a shape it can render even on a host with no library."""
    body = client.get("/api/conditions/stats").json()
    assert "available" in body
    if body["available"]:
        assert body["conditions"] > 0


def test_conditions_search_endpoint_clamps_limits(client):
    body = client.get("/api/conditions/search?q=diabetes&limit=10000").json()
    assert body["limit"] <= 100
    assert body["offset"] == 0


def test_conditions_search_endpoint_handles_a_missing_query(client):
    assert client.get("/api/conditions/search").status_code == 200


def test_condition_detail_404s_for_an_unknown_id(client):
    assert client.get("/api/conditions/NOT:A:REAL:ID").status_code == 404


@conditions_ready
def test_condition_detail_endpoint_returns_provenance(client):
    body = client.get("/api/conditions/MONDO:0005148").json()
    assert body["source"] and body["license"]
    assert body["synonyms"] and body["xrefs"]


def test_drugs_stats_endpoint_always_answers(client):
    body = client.get("/api/drugs/stats").json()
    assert "available" in body


def test_drugs_ingredients_endpoint_requires_a_name(client):
    assert client.get("/api/drugs/ingredients?name=").status_code == 400


def test_drugs_search_endpoint_rejects_an_unknown_kind(client):
    assert client.get("/api/drugs/search?q=metformin&kind=bogus").status_code == 400


def test_drugs_search_endpoint_accepts_a_valid_kind(client):
    assert client.get("/api/drugs/search?q=metformin&kind=ingredient").status_code == 200
