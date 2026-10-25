"""Read access to the condition library (`data/conditions.db`).

The library is built from open ontologies by
`scripts/fetch_condition_sources.py` + `scripts/build_condition_library.py`. It
is not committed: it is ~80 MB and reproducible from those scripts, and
`data/` is gitignored because it holds real user cases.

Because of that, every function here degrades instead of raising. A host that
has not built the library still serves the site: search returns nothing, the
stats report `available: false`, and callers fall back to the hand-curated
knowledge base. A missing optional dataset must never take the app down.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "conditions.db"

# One connection per thread: sqlite3 connections are not shareable across
# threads, and FastAPI runs sync endpoints in a worker pool.
_local = threading.local()

_COLUMNS = ("c.id, c.name, c.source, c.license, c.category, c.definition, "
            "c.is_rare, c.has_details, c.source_url")


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


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 prefix query.

    Each token is quoted, so FTS5 operators a user types (`AND`, `*`, `-`,
    quotes) are treated as literal text rather than syntax. The trailing `*`
    makes the last token a prefix match, so "diab" finds "diabetes".
    """
    tokens = re.findall(r"\w+", query or "")
    return " ".join(f'"{t}"*' for t in tokens)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "source": row["source"],
        "license": row["license"],
        "category": row["category"],
        "definition": row["definition"],
        "is_rare": bool(row["is_rare"]),
        "has_details": bool(row["has_details"]),
        "source_url": row["source_url"],
    }


def search(
    query: str = "",
    limit: int = 25,
    offset: int = 0,
    source: str | None = None,
    rare_only: bool = False,
    detailed_only: bool = False,
) -> dict[str, Any]:
    """Search conditions by name, synonym or definition.

    With no query, lists conditions ordered by name. Filters narrow the result
    set in SQL rather than in Python, because the table is far larger than one
    page.
    """
    conn = _conn()
    if conn is None:
        return {"total": 0, "limit": limit, "offset": offset, "results": [], "available": False}

    where: list[str] = []
    params: list[Any] = []
    if source:
        where.append("c.source = ?")
        params.append(source)
    if rare_only:
        where.append("c.is_rare = 1")
    if detailed_only:
        where.append("c.has_details = 1")
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    if query.strip():
        fts = _fts_query(query)
        if not fts:
            return {"total": 0, "limit": limit, "offset": offset, "results": [], "available": True}
        # A named condition should outrank a passing mention in a definition, so
        # rows whose name matches exactly sort first, then name prefixes, then
        # everything else the index matched.
        norm = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
        base = (f"FROM conditions_fts f JOIN conditions c ON c.rowid = f.rowid"
                f" WHERE conditions_fts MATCH ?")
        count_sql = f"SELECT COUNT(*) {base} {'AND ' + ' AND '.join(where) if where else ''}"
        total = conn.execute(count_sql, [fts, *params]).fetchone()[0]
        sql = (f"SELECT {_COLUMNS}, "
               f"CASE WHEN c.norm_name = ? THEN 0 "
               f"     WHEN c.norm_name LIKE ? THEN 1 ELSE 2 END AS rank "
               f"{base} {'AND ' + ' AND '.join(where) if where else ''} "
               f"ORDER BY rank, c.name LIMIT ? OFFSET ?")
        rows = conn.execute(sql, [norm, f"{norm}%", fts, *params, limit, offset]).fetchall()
    else:
        count_sql = f"SELECT COUNT(*) FROM conditions c{clause}"
        total = conn.execute(count_sql, params).fetchone()[0]
        sql = f"SELECT {_COLUMNS} FROM conditions c{clause} ORDER BY c.name LIMIT ? OFFSET ?"
        rows = conn.execute(sql, [*params, limit, offset]).fetchall()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "available": True,
        "results": [_row_to_dict(r) for r in rows],
    }


def get(condition_id: str) -> dict[str, Any] | None:
    """Fetch one condition with its synonyms and cross-references."""
    conn = _conn()
    if conn is None:
        return None
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM conditions c WHERE c.id = ?", (condition_id,)
    ).fetchone()
    if row is None:
        return None
    record = _row_to_dict(row)
    record["synonyms"] = [
        r["term"] for r in conn.execute(
            "SELECT term FROM condition_synonyms WHERE condition_id = ? ORDER BY term",
            (condition_id,))
    ]
    record["xrefs"] = [
        {"system": r["system"], "value": r["value"]} for r in conn.execute(
            "SELECT system, value FROM condition_xrefs WHERE condition_id = ? ORDER BY system",
            (condition_id,))
    ]
    return record


def lookup(name: str) -> dict[str, Any] | None:
    """Find a condition by exact normalised name.

    Exact only: "type 1 diabetes mellitus" and "type 2 diabetes mellitus" differ
    by one character and are different diseases, so this never approximates.
    """
    conn = _conn()
    if conn is None or not name:
        return None
    norm = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM conditions c WHERE c.norm_name = ? "
        f"ORDER BY CASE c.source WHEN 'kb' THEN 0 ELSE 1 END LIMIT 1", (norm,)
    ).fetchone()
    return _row_to_dict(row) if row else None


def stats() -> dict[str, Any]:
    """Library totals, per-source counts and the licences they carry."""
    conn = _conn()
    if conn is None:
        return {"available": False, "conditions": 0, "sources": []}
    totals = {}
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        totals = json.loads(meta.get("totals", "{}"))
        built_at = meta.get("built_at")
    except sqlite3.Error:
        built_at = None
    by_source = [
        {"source": r["source"], "license": r["license"], "count": r["n"]}
        for r in conn.execute(
            "SELECT source, license, COUNT(*) AS n FROM conditions GROUP BY source, license "
            "ORDER BY n DESC")
    ]
    return {
        "available": True,
        "conditions": totals.get("conditions", 0),
        "synonyms": totals.get("synonyms", 0),
        "cross_references": totals.get("xrefs", 0),
        "assessable": totals.get("with_details", 0),
        "rare": totals.get("rare", 0),
        "built_at": built_at,
        "sources": by_source,
    }
