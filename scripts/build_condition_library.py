"""Build the searchable condition library from the fetched open ontologies.

Produces `data/conditions.db`, a SQLite database with an FTS5 index. The point
of the library is breadth with provenance: every row carries the ontology it
came from, that ontology's licence and a stable identifier (DOID, MONDO, ORPHA,
ICD-10-CM, MeSH), so a name on the site can always be checked against its
source. Nothing is invented and nothing is inferred across sources.

The hand-curated knowledge base (`app/data/diseases.json`) is imported as the
source `kb` and kept apart from the ontologies: it is the only source that has
symptoms, medicines and routines, so the UI can mark which conditions the
assistant can actually reason about rather than just name.

Cross-source matching is deliberately exact, on the normalised preferred name.
Fuzzy name matching would merge distinct diseases -- "diabetes mellitus type 1"
and "type 2" differ by one character and are different conditions -- so records
are left unmerged when their names disagree.

Run:
    python scripts/fetch_condition_sources.py
    python scripts/build_condition_library.py
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "external" / "conditions"
KB_DISEASES = ROOT / "app" / "data" / "diseases.json"
MANIFEST = SRC / "condition_sources.json"

SCHEMA = """
CREATE TABLE conditions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    norm_name   TEXT NOT NULL,
    category    TEXT,
    definition  TEXT,
    source      TEXT NOT NULL,
    source_url  TEXT,
    license     TEXT NOT NULL,
    is_rare     INTEGER NOT NULL DEFAULT 0,
    has_details INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE condition_synonyms (
    condition_id TEXT NOT NULL,
    term         TEXT NOT NULL,
    norm_term    TEXT NOT NULL,
    kind         TEXT
);
CREATE TABLE condition_xrefs (
    condition_id TEXT NOT NULL,
    system       TEXT NOT NULL,
    value        TEXT NOT NULL
);
CREATE VIRTUAL TABLE conditions_fts USING fts5(
    name, synonyms, definition, content=''
);
CREATE INDEX idx_cond_norm ON conditions(norm_name);
CREATE INDEX idx_cond_source ON conditions(source);
CREATE INDEX idx_syn_norm ON condition_synonyms(norm_term);
CREATE INDEX idx_xref ON condition_xrefs(system, value);
"""


def normalise(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace.

    Used for exact-match lookup and dedup only, never for fuzzy merging.
    """
    text = html.unescape(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_obo_quotes(value: str) -> str:
    value = value.strip()
    if value.startswith('"'):
        end = value.rfind('"')
        if end > 0:
            return value[1:end]
    return value


def parse_obo(path: Path) -> Iterator[dict]:
    """Yield `[Term]` stanzas from an OBO file.

    Obsolete terms are skipped: they are superseded identifiers, and listing
    them would put withdrawn names in front of users.
    """
    current: dict | None = None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("["):
                if current and current.get("id"):
                    yield current
                current = {} if line == "[Term]" else None
                continue
            if current is None or not line:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("def: "):
                current["definition"] = _strip_obo_quotes(line[5:])
            elif line.startswith("synonym: "):
                value = _strip_obo_quotes(line[9:])
                if value:
                    current.setdefault("synonyms", []).append(value)
            elif line.startswith("subset: "):
                current.setdefault("subsets", set()).add(line[8:].split(" {")[0].strip())
            elif line.startswith("xref: "):
                current.setdefault("xrefs", []).append(line[6:].strip())
    if current and current.get("id"):
        yield current


_RARE_SUBSETS = {"rare", "gard_rare", "nord_rare", "orphanet_rare", "ordo_disorder",
                 "doid_rare", "inferred_rare"}


def iter_mondo() -> Iterator[dict]:
    for term in parse_obo(SRC / "mondo.obo"):
        tid = term.get("id", "")
        name = term.get("name", "")
        if not tid.startswith("MONDO:") or not name or name.startswith("obsolete"):
            continue
        # MONDO re-exports non-disease branches (BFO, UBERON, GO) under their own
        # ids, which the prefix check above already drops.
        yield {
            "id": tid,
            "name": name,
            "definition": term.get("definition", ""),
            "synonyms": term.get("synonyms", []),
            "xrefs": term.get("xrefs", []),
            "source": "mondo",
            "is_rare": int(bool(term.get("subsets", set()) & _RARE_SUBSETS)),
        }


def iter_doid() -> Iterator[dict]:
    for term in parse_obo(SRC / "doid.obo"):
        tid = term.get("id", "")
        name = term.get("name", "")
        if not tid.startswith("DOID:") or not name or name.startswith("obsolete"):
            continue
        yield {
            "id": tid,
            "name": name,
            "definition": term.get("definition", ""),
            "synonyms": term.get("synonyms", []),
            "xrefs": term.get("xrefs", []),
            "source": "doid",
            "is_rare": 0,
        }


def iter_orphanet() -> Iterator[dict]:
    """Stream the Orphanet rare-disease nomenclature.

    The file is 54 MB of XML; iterparse with clearing keeps memory flat.
    """
    for _, elem in ET.iterparse(SRC / "orphanet_product1.xml", events=("end",)):
        if elem.tag != "Disorder":
            continue
        orpha = elem.findtext("OrphaCode")
        name = elem.findtext("Name")
        if not orpha or not name:
            elem.clear()
            continue
        synonyms = [s.text for s in elem.iter("Synonym") if s.text]
        xrefs = []
        for ref in elem.iter("ExternalReference"):
            src = ref.findtext("Source")
            val = ref.findtext("Reference")
            if src and val:
                xrefs.append(f"{src}:{val}")
        yield {
            "id": f"ORPHA:{orpha}",
            "name": name,
            "definition": "",
            "synonyms": synonyms,
            "xrefs": xrefs,
            "source": "orphanet",
            "is_rare": 1,
        }
        elem.clear()


def iter_icd10cm() -> Iterator[dict]:
    data = json.loads((SRC / "icd10cm.json").read_text())
    seen: set[str] = set()
    for rows in data.values():
        for row in rows:
            if not isinstance(row, list) or len(row) < 2:
                continue
            code, name = row[0], row[1]
            if not code or code in seen:
                continue
            seen.add(code)
            yield {
                "id": f"ICD10CM:{code}",
                "name": name,
                "definition": "",
                "synonyms": [],
                "xrefs": [f"ICD10CM:{code}"],
                "source": "icd10cm",
                "is_rare": 0,
            }


# MeSH tree branches holding diseases and symptoms:
# C = Diseases, F03 = Mental Disorders, SP = Psychiatry and Psychology.
_MESH_DISEASE_TREES = ("C", "F03", "SP")


def iter_mesh() -> Iterator[dict]:
    """Stream MeSH descriptors, keeping only the disease and symptom branches."""
    for _, elem in ET.iterparse(SRC / "mesh_desc2025.xml", events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue
        ui = elem.findtext("DescriptorUI")
        name = elem.findtext("DescriptorName/String")
        if not ui or not name:
            elem.clear()
            continue
        trees = [t.text or "" for t in elem.iter("TreeNumber")]
        if not any(t.startswith(_MESH_DISEASE_TREES) for t in trees):
            elem.clear()
            continue
        synonyms = [s.text for s in elem.iter("Term/String") if s.text]
        scope = elem.findtext("ScopeNote")
        if scope:
            scope = re.sub(r"<[^>]+>", "", scope)
        yield {
            "id": f"MESH:{ui}",
            "name": name,
            "definition": scope or "",
            "synonyms": synonyms,
            "xrefs": [f"MESH:{ui}"] + [f"MESHTREE:{t}" for t in trees],
            "source": "mesh",
            "is_rare": 0,
        }
        elem.clear()


def iter_kb() -> Iterator[dict]:
    """The hand-curated knowledge base: the only source with symptoms and medicines."""
    raw = json.loads(KB_DISEASES.read_text())["diseases"]
    for disease in raw:
        yield {
            "id": f"KB:{disease['id']}",
            "name": disease["name"],
            "definition": disease.get("description", ""),
            "synonyms": [s.replace("_", " ") for s in disease.get("key_symptoms", [])],
            "xrefs": [f"ICD10:{disease['icd10']}"] if disease.get("icd10") else [],
            "source": "kb",
            "is_rare": 0,
            "category": disease.get("category"),
            "has_details": 1,
        }


SOURCE_ITERATORS = {
    "mondo": iter_mondo,
    "doid": iter_doid,
    "orphanet": iter_orphanet,
    "icd10cm": iter_icd10cm,
    "mesh": iter_mesh,
    "kb": iter_kb,
}

# Which files must exist for a source to be imported.
SOURCE_FILES = {
    "mondo": "mondo.obo",
    "doid": "doid.obo",
    "orphanet": "orphanet_product1.xml",
    "icd10cm": "icd10cm.json",
    "mesh": "mesh_desc2025.xml",
}


def _licenses() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text()).get("sources", {})


def _flush(conn, conds, syns, xrefs) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO conditions "
        "(id,name,norm_name,category,definition,source,source_url,license,is_rare,has_details) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)", conds)
    conn.executemany(
        "INSERT INTO condition_synonyms (condition_id,term,norm_term,kind) VALUES (?,?,?,?)", syns)
    conn.executemany(
        "INSERT INTO condition_xrefs (condition_id,system,value) VALUES (?,?,?)", xrefs)


def build(out: Path) -> dict:
    licenses = _licenses()
    if out.exists():
        out.unlink()
    conn = sqlite3.connect(out)
    conn.executescript(SCHEMA)

    stats: dict[str, int] = {}
    for source, iterator in SOURCE_ITERATORS.items():
        needed = SOURCE_FILES.get(source)
        if needed and not (SRC / needed).exists():
            print(f"  {source}: skipped (missing {needed})")
            continue
        meta = licenses.get(source, {})
        lic = "hand-curated" if source == "kb" else meta.get("license", "unknown")
        url = "" if source == "kb" else meta.get("url", "")

        count = 0
        batch_cond, batch_syn, batch_xref = [], [], []
        for rec in iterator():
            norm = normalise(rec["name"])
            if not norm:
                continue
            batch_cond.append((
                rec["id"], rec["name"], norm, rec.get("category"), rec.get("definition", ""),
                rec["source"], url, lic, rec.get("is_rare", 0), rec.get("has_details", 0),
            ))
            for term in dict.fromkeys(rec.get("synonyms", [])):
                term = term.strip()
                if term and len(term) < 200:
                    batch_syn.append((rec["id"], term, normalise(term), "synonym"))
            for xref in dict.fromkeys(rec.get("xrefs", [])):
                if ":" in xref:
                    system, _, value = xref.partition(":")
                    batch_xref.append((rec["id"], system.strip(), value.strip()))
            count += 1
            if len(batch_cond) >= 5000:
                _flush(conn, batch_cond, batch_syn, batch_xref)
                batch_cond, batch_syn, batch_xref = [], [], []
        if batch_cond:
            _flush(conn, batch_cond, batch_syn, batch_xref)
        stats[source] = count
        print(f"  {source}: {count:,} conditions")
    conn.commit()

    # A condition the assistant can actually assess is one whose name matches a
    # knowledge-base condition exactly. Marking by exact name keeps the badge
    # honest: a similarly-spelled ontology entry is not given the KB's symptoms.
    conn.execute("""
        UPDATE conditions SET has_details = 1
        WHERE source != 'kb' AND norm_name IN
              (SELECT norm_name FROM conditions WHERE source = 'kb')
    """)
    conn.commit()

    # Index from the same rows the API reads, so search and listing never diverge.
    conn.execute("""
        INSERT INTO conditions_fts (rowid, name, synonyms, definition)
        SELECT c.rowid, c.name,
               COALESCE((SELECT group_concat(s.term, ' ')
                         FROM condition_synonyms s WHERE s.condition_id = c.id), ''),
               COALESCE(c.definition, '')
        FROM conditions c
    """)
    conn.execute("INSERT INTO conditions_fts(conditions_fts) VALUES('optimize')")
    conn.commit()

    totals = {
        "conditions": conn.execute("SELECT COUNT(*) FROM conditions").fetchone()[0],
        "synonyms": conn.execute("SELECT COUNT(*) FROM condition_synonyms").fetchone()[0],
        "xrefs": conn.execute("SELECT COUNT(*) FROM condition_xrefs").fetchone()[0],
        "with_details": conn.execute("SELECT COUNT(*) FROM conditions WHERE has_details=1").fetchone()[0],
        "rare": conn.execute("SELECT COUNT(*) FROM conditions WHERE is_rare=1").fetchone()[0],
        "by_source": stats,
    }
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.executemany("INSERT INTO meta VALUES (?,?)", [
        ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("totals", json.dumps(totals)),
        ("licenses", json.dumps({k: v.get("license") for k, v in licenses.items()})),
    ])
    conn.commit()
    conn.close()
    return totals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "conditions.db"))
    args = ap.parse_args(argv)

    if not SRC.exists():
        print("No ontology files found. Run scripts/fetch_condition_sources.py first.",
              file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building {out} ...")
    totals = build(out)
    print(f"\n  conditions  : {totals['conditions']:,}")
    print(f"  synonyms    : {totals['synonyms']:,}")
    print(f"  xrefs       : {totals['xrefs']:,}")
    print(f"  with details: {totals['with_details']:,}")
    print(f"  rare        : {totals['rare']:,}")
    print(f"  size        : {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
