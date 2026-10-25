"""Tests for the world health statistics module and its API surface.

The numbers on the Insights page come from the WHO Global Health Observatory and
World Bank Open Data. Nothing on a request path may call the network, so these
tests use a temporary cache file and never hit the real APIs. What matters is
that a cached figure keeps its source, indicator code and reference year, and
that a missing cache degrades to "unavailable" instead of inventing values.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import worlddata
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    path = tmp_path / "world_health.json"
    monkeypatch.setattr(worlddata, "CACHE", path)
    monkeypatch.setattr(worlddata, "CACHE_DIR", tmp_path)
    return path


SAMPLE = {
    "country": "IND",
    "fetched_at": "2026-01-02T03:04:05+00:00",
    "who": [
        {"code": "WHOSIS_000001", "label": "Life expectancy at birth", "unit": "years",
         "sex": "SEX_BTSX", "source": "WHO GHO",
         "points": [{"year": 2022, "value": 70.1}, {"year": 2023, "value": 71.0}],
         "latest": {"year": 2023, "value": 71.0}},
    ],
    "world_bank": [
        {"code": "SH.XPD.CHEX.GD.ZS", "label": "Current health expenditure",
         "unit": "% of GDP", "source": "World Bank Open Data",
         "points": [{"year": 2023, "value": 3.339}],
         "latest": {"year": 2023, "value": 3.339}},
    ],
    "sources": [{"name": "WHO Global Health Observatory", "url": "https://www.who.int/data/gho",
                 "license": "CC-BY-NC-SA-3.0-IGO", "note": ""}],
}


# ---- no cache: degrade, never invent ----------------------------------------
def test_load_without_cache_is_unavailable(cache):
    assert not cache.exists()
    data = worlddata.load()
    assert data["available"] is False
    assert data["who"] == [] and data["world_bank"] == []


def test_summary_without_cache_is_unavailable(cache):
    assert worlddata.summary()["available"] is False


def test_load_with_corrupt_cache_is_unavailable(cache):
    cache.write_text("{not json")
    assert worlddata.load()["available"] is False


def test_load_without_cache_does_not_touch_the_network(cache, monkeypatch):
    """A request must never reach out to a publisher; only refresh() may."""
    def explode(*_a, **_k):
        raise AssertionError("load() must not perform network I/O")

    monkeypatch.setattr(worlddata.urllib.request, "urlopen", explode)
    assert worlddata.load()["available"] is False


# ---- cached data -------------------------------------------------------------
def test_summary_reports_each_indicator_with_its_year_and_source(cache):
    cache.write_text(json.dumps(SAMPLE))
    data = worlddata.summary()
    assert data["available"] is True
    assert data["country"] == "IND"
    assert data["fetched_at"].startswith("2026-01-02")

    by_label = {i["label"]: i for i in data["indicators"]}
    life = by_label["Life expectancy at birth"]
    # A statistic without its reference year is misleading, so it is carried.
    assert life["latest_year"] == 2023
    assert life["latest_value"] == 71.0
    assert life["source"] == "WHO GHO"
    assert life["code"] == "WHOSIS_000001"
    assert [p["year"] for p in life["points"]] == [2022, 2023]

    spend = by_label["Current health expenditure"]
    assert spend["source"] == "World Bank Open Data"
    assert spend["latest_value"] == 3.339


def test_summary_keeps_the_publisher_licences(cache):
    cache.write_text(json.dumps(SAMPLE))
    sources = worlddata.summary()["sources"]
    assert sources and sources[0]["license"] == "CC-BY-NC-SA-3.0-IGO"


# ---- indicator definitions ---------------------------------------------------
def test_who_indicators_are_distinct_and_have_units():
    codes = [i["code"] for i in worlddata.WHO_INDICATORS]
    assert len(codes) == len(set(codes)), "duplicate WHO indicator code"
    for spec in worlddata.WHO_INDICATORS:
        assert spec["label"] and spec["unit"], spec


def test_world_bank_indicators_are_distinct_and_have_units():
    codes = [i["code"] for i in worlddata.WB_INDICATORS]
    assert len(codes) == len(set(codes)), "duplicate World Bank indicator code"
    for spec in worlddata.WB_INDICATORS:
        assert spec["label"] and spec["unit"], spec


# ---- API ---------------------------------------------------------------------
def test_world_health_endpoint_returns_cached_data(client, cache):
    cache.write_text(json.dumps(SAMPLE))
    body = client.get("/api/world-health?country=IND").json()
    assert body["available"] is True
    assert body["country"] == "IND"
    assert len(body["indicators"]) == 2


def test_world_health_endpoint_rejects_a_bad_country_code(client):
    assert client.get("/api/world-health?country=India").status_code == 400
    assert client.get("/api/world-health?country=IN").status_code == 400
    assert client.get("/api/world-health?country=123").status_code == 400


def test_world_health_endpoint_degrades_without_a_cache(client, cache):
    body = client.get("/api/world-health?country=IND").json()
    assert body["available"] is False
    assert body["indicators"] == []
