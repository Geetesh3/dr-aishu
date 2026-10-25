"""Real-world health statistics from government and international sources.

Insights should show numbers that can be checked, so this module reads only
from official publishers and records the source, indicator code and retrieval
time alongside every value:

* WHO Global Health Observatory (ghoapi.azureedge.net) — the World Health
  Organization's own indicator API.
* World Bank Open Data (api.worldbank.org) — health spending and population.
* US CDC / NCHS via the NLM Clinical Tables API — used for the condition
  library rather than here.

Values are cached under `data/external/world/` and served from cache. Nothing
on a request path calls out to the network: an upstream outage must not slow
down or break a page, and a host with no egress must still serve the site. When
no cache exists the module reports `available: false` and the page says so
instead of inventing numbers.

Run:
    python scripts/refresh_world_data.py            # fill the cache
    python scripts/refresh_world_data.py --country IND
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "external" / "world"
CACHE = CACHE_DIR / "world_health.json"

WHO = "https://ghoapi.azureedge.net/api"
WB = "https://api.worldbank.org/v2"
UA = "Mozilla/5.0 (compatible; DrAishuDataBot/1.0; educational use)"

# WHO GHO indicators. `sex` picks the Dim1 value to report, or None for totals.
WHO_INDICATORS: list[dict[str, Any]] = [
    {"code": "WHOSIS_000001", "label": "Life expectancy at birth", "unit": "years", "sex": "SEX_BTSX"},
    {"code": "WHOSIS_000002", "label": "Healthy life expectancy (HALE) at birth", "unit": "years", "sex": "SEX_BTSX"},
    {"code": "MDG_0000000007", "label": "Under-five mortality rate", "unit": "per 1,000 live births", "sex": None},
    {"code": "MDG_0000000026", "label": "Maternal mortality ratio", "unit": "per 100,000 live births", "sex": None},
    {"code": "MDG_0000000020", "label": "Tuberculosis incidence", "unit": "per 100,000 per year", "sex": None},
    {"code": "NCDMORT3070", "label": "Probability of dying (30-70) from cardiovascular disease, cancer, diabetes or chronic respiratory disease", "unit": "%", "sex": "SEX_BTSX"},
    {"code": "NCD_GLUC_04", "label": "Raised fasting blood glucose (>=7.0 mmol/L)", "unit": "%", "sex": "SEX_BTSX"},
    {"code": "NCD_BMI_30C", "label": "Obesity among adults (BMI >= 30)", "unit": "%", "sex": "SEX_BTSX"},
    {"code": "WHS4_100", "label": "DTP3 immunisation coverage among 1-year-olds", "unit": "%", "sex": None},
    {"code": "SA_0000001688", "label": "Alcohol consumption per capita (15+)", "unit": "litres per year", "sex": "SEX_BTSX"},
]

# World Bank indicators. Codes are stable and documented in the WB API.
WB_INDICATORS: list[dict[str, str]] = [
    {"code": "SP.POP.TOTL", "label": "Population", "unit": "people"},
    {"code": "SH.XPD.CHEX.GD.ZS", "label": "Current health expenditure", "unit": "% of GDP"},
    {"code": "SH.XPD.CHEX.PC.CD", "label": "Current health expenditure per capita", "unit": "current US$"},
    {"code": "SH.MED.PHYS.ZS", "label": "Physicians", "unit": "per 1,000 people"},
    {"code": "SH.MED.BEDS.ZS", "label": "Hospital beds", "unit": "per 1,000 people"},
    {"code": "SP.DYN.LE00.IN", "label": "Life expectancy at birth (World Bank)", "unit": "years"},
]

DEFAULT_COUNTRY = "IND"
# Years pulled per indicator. Enough for a recent trend without a huge payload.
YEAR_RANGE = "2015:2025"


def _get_json(url: str, timeout: int = 45) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _who_series(country: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in WHO_INDICATORS:
        query = urllib.parse.urlencode({"$filter": f"SpatialDim eq '{country}'"})
        url = f"{WHO}/{spec['code']}?{query}"
        try:
            rows = _get_json(url).get("value", [])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            out.append({**spec, "source": "WHO GHO", "error": str(exc), "points": []})
            continue
        points = []
        for row in rows:
            value = row.get("NumericValue")
            year = row.get("TimeDim")
            if value is None or year is None:
                continue
            if spec["sex"] and row.get("Dim1") not in (spec["sex"], None, ""):
                continue
            points.append({"year": int(year), "value": round(float(value), 3)})
        # One value per year: the most recent published figure wins.
        dedup = {}
        for p in points:
            dedup[p["year"]] = p
        series = [dedup[y] for y in sorted(dedup)]
        out.append({**spec, "source": "WHO GHO", "url": url, "points": series,
                    "latest": series[-1] if series else None})
    return out


def _wb_series(country: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in WB_INDICATORS:
        url = (f"{WB}/country/{country}/indicator/{spec['code']}"
               f"?format=json&per_page=200&date={YEAR_RANGE}")
        try:
            payload = _get_json(url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            out.append({**spec, "source": "World Bank", "error": str(exc), "points": []})
            continue
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        points = sorted(
            ({"year": int(r["date"]), "value": round(float(r["value"]), 3)}
             for r in rows if r.get("value") is not None),
            key=lambda p: p["year"],
        )
        out.append({**spec, "source": "World Bank Open Data", "url": url, "points": points,
                    "latest": points[-1] if points else None})
    return out


def refresh(country: str = DEFAULT_COUNTRY) -> dict[str, Any]:
    """Fetch every indicator and write the cache. Network calls happen here only."""
    payload = {
        "country": country,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "who": _who_series(country),
        "world_bank": _wb_series(country),
        "sources": [
            {"name": "WHO Global Health Observatory", "url": "https://www.who.int/data/gho",
             "license": "CC-BY-NC-SA-3.0-IGO",
             "note": "World Health Organization indicator API."},
            {"name": "World Bank Open Data", "url": "https://data.worldbank.org",
             "license": "CC-BY-4.0",
             "note": "Health spending, workforce and population indicators."},
        ],
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def load(country: str = DEFAULT_COUNTRY) -> dict[str, Any]:
    """Read the cached statistics. Returns an unavailable stub when absent."""
    if not CACHE.exists():
        return {"available": False, "country": country, "who": [], "world_bank": [], "sources": []}
    try:
        payload = json.loads(CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"available": False, "country": country, "who": [], "world_bank": [], "sources": []}
    payload["available"] = True
    return payload


def summary(country: str = DEFAULT_COUNTRY) -> dict[str, Any]:
    """Headline figures plus full series, for the Insights page.

    The unavailable shape carries the same keys as the available one, so a caller
    (and the page's JavaScript) can read `indicators` and `sources` without
    branching on whether the cache exists.
    """
    payload = load(country)
    if not payload.get("available"):
        return {"available": False, "country": payload.get("country", country),
                "fetched_at": None, "indicators": [], "sources": []}
    indicators = [
        {
            "label": i["label"], "unit": i["unit"], "source": i["source"],
            "code": i["code"],
            "latest_year": (i.get("latest") or {}).get("year"),
            "latest_value": (i.get("latest") or {}).get("value"),
            "points": i.get("points", []),
        }
        for i in payload.get("who", []) + payload.get("world_bank", [])
    ]
    return {
        "available": True,
        "country": payload["country"],
        "fetched_at": payload.get("fetched_at"),
        "indicators": indicators,
        "sources": payload.get("sources", []),
    }
