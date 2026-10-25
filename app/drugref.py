"""Read access to the RxNorm-derived medicine reference (`data/medicines_reference.db`).

This answers "is this a real medicine, and what molecule is in it?" using RxNorm,
published by the US National Library of Medicine. It carries names and molecule
links only. It deliberately holds no dosing, indication or contraindication
data, because RxNorm does not publish those and they must stay hand-reviewed in
`app/data/medicines.json`.

Like the condition library, the database is built from a downloaded release and
is not committed, so every function degrades to "nothing found" when it is
absent. Callers treat a miss as "unknown", never as "safe".
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "medicines_reference.db"

_local = threading.local()

KIND_LABEL = {
    "ingredient": "Ingredient",
    "brand": "Brand name",
    "product": "Product",
    "synonym": "Synonym",
}


def _conn() -> sqlite3.Connection | None:
    if not DB_PATH.exists():
        return None
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def available() -> bool:
    return DB_PATH.exists()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _fts_query(query: str) -> str:
    tokens = re.findall(r"\w+", query or "")
    return " ".join(f'"{t}"*' for t in tokens)


def search(query: str, limit: int = 20, kind: str | None = None) -> dict[str, Any]:
    """Search medicine names. `kind` narrows to ingredient, brand or product."""
    conn = _conn()
    if conn is None:
        return {"total": 0, "results": [], "available": False}
    fts = _fts_query(query)
    if not fts:
        return {"total": 0, "results": [], "available": True}

    where = ["drug_fts MATCH ?"]
    params: list[Any] = [fts]
    if kind:
        where.append("kind = ?")
        params.append(kind)
    clause = " AND ".join(where)

    total = conn.execute(
        f"SELECT COUNT(*) FROM drug_fts WHERE {clause}", params).fetchone()[0]
    norm = _norm(query)
    # An exact name first, then names that start with the query, then the rest of
    # what the index matched.
    rows = conn.execute(
        f"SELECT name, norm_name, rxcui, tty, kind FROM drug_fts WHERE {clause} "
        f"ORDER BY CASE WHEN norm_name = ? THEN 0 WHEN norm_name LIKE ? THEN 1 ELSE 2 END, "
        f"length(name), name LIMIT ?",
        [*params, norm, f"{norm}%", max(1, min(limit, 100))]).fetchall()
    return {
        "total": total,
        "available": True,
        "results": [{"name": r["name"], "rxcui": r["rxcui"], "tty": r["tty"],
                     "kind": r["kind"], "kind_label": KIND_LABEL.get(r["kind"], r["kind"])}
                    for r in rows],
    }


def ingredients_for(name: str, limit: int = 10) -> list[dict[str, Any]]:
    """Which molecules a medicine name contains.

    Brand and product names are resolved to their ingredient through RxNorm's
    own relations, not by parsing the name, so "Glucophage" resolves to
    metformin because RxNorm says so. If no relation exists the product name is
    read back token by token as a fallback.
    """
    conn = _conn()
    if conn is None or not name.strip():
        return []
    norm = _norm(name)
    row = conn.execute(
        "SELECT rxcui, name FROM drug_names WHERE norm_name = ? "
        "ORDER BY CASE kind WHEN 'ingredient' THEN 0 WHEN 'brand' THEN 1 ELSE 2 END LIMIT 1",
        (norm,)).fetchone()
    if row is None:
        fts = _fts_query(name)
        if not fts:
            return []
        # Fall back to the best index hit, so a slightly different spelling of a
        # product name still resolves.
        row = conn.execute(
            "SELECT rxcui, name FROM drug_fts WHERE drug_fts MATCH ? "
            "ORDER BY length(name) LIMIT 1", (fts,)).fetchone()
        if row is None:
            return []
    rxcui = row["rxcui"]

    linked = conn.execute(
        "SELECT DISTINCT d.rxcui, d.name FROM ingredient_links l JOIN drugs d "
        "ON d.rxcui = l.ingredient_rxcui WHERE l.rxcui = ? ORDER BY d.name LIMIT ?",
        (rxcui, limit)).fetchall()
    if linked:
        return [{"rxcui": r["rxcui"], "name": r["name"]} for r in linked]

    direct = conn.execute(
        "SELECT rxcui, name FROM drugs WHERE rxcui = ? AND is_ingredient = 1", (rxcui,)).fetchone()
    if direct is not None:
        return [{"rxcui": direct["rxcui"], "name": direct["name"]}]

    # No relation on record: read the ingredient out of RxNorm's canonical name
    # for this entry. A synonym row may be the brand-only string ("Glucophage XR
    # 500 MG Tablet"), so the product row for the same RXCUI is preferred -- it
    # is the one that spells out the molecule.
    canonical = conn.execute(
        "SELECT name FROM drug_names WHERE rxcui = ? "
        "ORDER BY CASE kind WHEN 'product' THEN 0 WHEN 'ingredient' THEN 1 ELSE 2 END, "
        "length(name) LIMIT 1", (rxcui,)).fetchone()
    for candidate in {canonical["name"] if canonical else None, row["name"], name}:
        if candidate:
            found = _ingredients_from_product_name(candidate, conn, limit)
            if found:
                return found
    return []


def _ingredients_from_product_name(name: str, conn: sqlite3.Connection,
                                   limit: int) -> list[dict[str, Any]]:
    """Resolve an ingredient from a product name such as
    "24 HR metformin hydrochloride 500 MG Extended Release Oral Tablet [Glucophage]".

    RxNorm product names contain the ingredient, but not always at the start --
    "24 HR metformin hydrochloride 500 MG ..." leads with the release form. So
    every ingredient name is matched against the product name on word
    boundaries, and the longest matches win: "metformin hydrochloride" is
    preferred over "metformin", which is preferred over a stray "acid".
    """
    index = _ingredient_index(conn)
    if not index:
        return []
    product = _norm(name)
    padded = f" {product} "
    hits: list[tuple[int, str, str]] = []
    for ing_norm, (rxcui, display) in index.items():
        if f" {ing_norm} " in padded:
            hits.append((len(ing_norm), rxcui, display))
    hits.sort(reverse=True)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, rxcui, display in hits:
        if rxcui in seen:
            continue
        # Skip a shorter ingredient already covered by a longer match, so
        # "metformin" is not reported alongside "metformin hydrochloride".
        if any(display.lower() in kept.lower() and display.lower() != kept.lower()
               for kept in (o["name"] for o in out)):
            continue
        seen.add(rxcui)
        out.append({"rxcui": rxcui, "name": display})
        if len(out) >= limit:
            break
    return out


def _ingredient_index(conn: sqlite3.Connection) -> dict[str, tuple[str, str]]:
    """Normalised ingredient name -> (rxcui, display name), cached per process.

    The ingredient table is a few thousand rows, so holding it in memory avoids
    re-querying it for every product name that needs resolving.
    """
    cached = getattr(_local, "ingredient_index", None)
    if cached is None:
        cached = {r["norm_name"]: (r["rxcui"], r["name"])
                  for r in conn.execute("SELECT rxcui, name, norm_name FROM drugs")}
        _local.ingredient_index = cached
    return cached


def stats() -> dict[str, Any]:
    conn = _conn()
    if conn is None:
        return {"available": False, "ingredients": 0, "names": 0, "sources": []}
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.Error:
        meta = {}
    totals = json.loads(meta.get("totals", "{}")) if meta.get("totals") else {}
    return {
        "available": True,
        "ingredients": totals.get("ingredients", 0),
        "names": totals.get("names", 0),
        "brands": totals.get("brands", 0),
        "products": totals.get("products", 0),
        "source": meta.get("source"),
        "source_url": meta.get("source_url"),
        "license": meta.get("license"),
        "attribution": meta.get("attribution"),
        "built_at": meta.get("built_at"),
    }
