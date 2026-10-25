"""Build the medicine reference database from RxNorm.

RxNorm is published by the US National Library of Medicine. Its Terms of Service
state that the RxNorm normalized names and codes (RXCUI) created by NLM are in
the public domain, while the *full* dataset also bundles proprietary
terminologies that carry per-source restrictions. This script therefore reads
only rows with `SAB=RXNORM` -- the NLM-created names and codes -- and nothing
from the bundled proprietary sources.

That gives Dr. Aishu an authoritative check on whether a medicine name is real
and what molecule it contains, including Indian brand names that RxNorm records
as brand names for the same ingredient. It deliberately carries no dosing,
indication or contraindication data: RxNorm does not publish those, and the
clinical fields in `app/data/medicines.json` remain hand-reviewed.

Run:
    python scripts/fetch_medicine_sources.py     # downloads the release
    python scripts/build_medicine_reference.py   # builds data/medicines_reference.db
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARCHIVE = ROOT / "data" / "external" / "medicines" / "rxnorm_full_prescribe.zip"
DB_PATH = ROOT / "data" / "medicines_reference.db"

# RxNorm term types worth indexing for a name lookup:
#   IN/PIN  ingredient and precise ingredient (the molecule)
#   BN      brand name
#   MIN     multiple ingredients
#   SCD/SBD clinical drug and branded drug ("metformin 500 MG Oral Tablet")
#   SY      synonym
TERM_TYPES = {"IN", "PIN", "BN", "MIN", "SCD", "SBD", "SY", "DF", "DFG"}
INGREDIENT_TYPES = {"IN", "PIN", "MIN"}

LICENSE = "Public domain (NLM-created RxNorm names and codes)"
ATTRIBUTION = ("RxNorm, U.S. National Library of Medicine. Only NLM-created "
               "RxNorm names and codes (SAB=RXNORM) are used.")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode = WAL;
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE drugs (
            rxcui       TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            norm_name   TEXT NOT NULL,
            tty         TEXT NOT NULL,
            is_ingredient INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE drug_names (
            rxcui     TEXT NOT NULL,
            name      TEXT NOT NULL,
            norm_name TEXT NOT NULL,
            tty       TEXT NOT NULL,
            kind      TEXT NOT NULL          -- 'ingredient' | 'brand' | 'synonym' | 'product'
        );
        CREATE TABLE ingredient_links (
            rxcui            TEXT NOT NULL,   -- brand or product RXCUI
            ingredient_rxcui TEXT NOT NULL,   -- molecule RXCUI
            PRIMARY KEY (rxcui, ingredient_rxcui)
        );
        CREATE INDEX idx_drugs_norm ON drugs(norm_name);
        CREATE INDEX idx_names_norm ON drug_names(norm_name);
        CREATE INDEX idx_names_rxcui ON drug_names(rxcui);
        CREATE INDEX idx_links_rxcui ON ingredient_links(rxcui);
    """)


def _finish_fts(conn: sqlite3.Connection) -> None:
    """Add the full-text index over every name after the bulk insert.

    Building FTS last is much faster than maintaining it during the insert.
    """
    conn.executescript("""
        CREATE VIRTUAL TABLE drug_fts USING fts5(
            name, norm_name, rxcui UNINDEXED, tty UNINDEXED, kind UNINDEXED,
            tokenize = 'unicode61'
        );
        INSERT INTO drug_fts (name, norm_name, rxcui, tty, kind)
            SELECT name, norm_name, rxcui, tty, kind FROM drug_names;
    """)


def build(archive: Path = ARCHIVE, db_path: Path = DB_PATH) -> dict:
    if not archive.exists():
        raise SystemExit(
            f"RxNorm archive not found at {archive}.\n"
            f"Run: python scripts/fetch_medicine_sources.py")

    if db_path.exists():
        db_path.unlink()
    for suffix in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + suffix)
        if side.exists():
            side.unlink()

    started = time.time()
    conn = sqlite3.connect(db_path)
    _schema(conn)

    ingredients: dict[str, dict] = {}
    names: list[tuple] = []
    links: set[tuple[str, str]] = set()
    skipped = 0
    rel_rows = 0

    with zipfile.ZipFile(archive) as z:
        member = next((n for n in z.namelist() if n.endswith("RXNCONSO.RRF")), None)
        if member is None:
            raise SystemExit("RXNCONSO.RRF not present in the archive")
        with z.open(member) as fh:
            for raw in fh:
                parts = raw.decode("utf-8", "replace").split("|")
                if len(parts) < 17:
                    skipped += 1
                    continue
                rxcui, sab, tty, name = parts[0], parts[11], parts[12], parts[14]
                # Only NLM-created RxNorm rows: the bundled proprietary
                # terminologies carry separate reuse restrictions.
                if sab != "RXNORM" or tty not in TERM_TYPES or not name.strip():
                    continue
                n = norm(name)
                if not n:
                    continue
                is_ing = tty in INGREDIENT_TYPES
                kind = "ingredient" if is_ing else ("brand" if tty == "BN" else
                        ("product" if tty in ("SCD", "SBD", "DF", "DFG") else "synonym"))
                names.append((rxcui, name.strip(), n, tty, kind))
                if is_ing:
                    # An ingredient's canonical name is the plain ingredient name,
                    # not a precise or multi-ingredient variant.
                    if rxcui not in ingredients or tty == "IN":
                        ingredients[rxcui] = {"name": name.strip(), "norm": n, "tty": tty}

        # Brand-to-ingredient links come from the relation table. RxNorm states
        # these both ways (`has_ingredient` on the brand, `ingredient_of` on the
        # product); reading both directions covers brands and clinical products
        # without needing to know which RXCUI is which.
        rel_member = next((n for n in z.namelist() if n.endswith("RXNREL.RRF")), None)
        if rel_member is None:
            raise SystemExit("RXNREL.RRF not present in the archive")
        with z.open(rel_member) as fh:
            for raw in fh:
                parts = raw.decode("utf-8", "replace").split("|")
                if len(parts) < 11 or parts[10] != "RXNORM":
                    continue
                rxcui1, rela, rxcui2 = parts[0], parts[7], parts[4]
                if rela == "has_ingredient" or rela == "ingredient_of":
                    owner, molecule = (rxcui1, rxcui2) if rela == "has_ingredient" else (rxcui2, rxcui1)
                elif rela == "has_tradename" or rela == "tradename_of":
                    # has_tradename: brand -> ingredient; tradename_of: ingredient -> brand
                    owner, molecule = ((rxcui1, rxcui2) if rela == "has_tradename"
                                       else (rxcui2, rxcui1))
                else:
                    continue
                rel_rows += 1
                if molecule in ingredients:
                    links.add((owner, molecule))

    conn.executemany(
        "INSERT INTO drugs (rxcui, name, norm_name, tty, is_ingredient) VALUES (?,?,?,?,?)",
        [(r, i["name"], i["norm"], i["tty"], 1) for r, i in ingredients.items()],
    )
    # A brand or product name is kept as a name row, not as its own molecule.
    conn.executemany(
        "INSERT INTO drug_names (rxcui, name, norm_name, tty, kind) VALUES (?,?,?,?,?)",
        names,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO ingredient_links (rxcui, ingredient_rxcui) VALUES (?,?)",
        sorted(links),
    )
    _finish_fts(conn)

    conn.executemany("INSERT INTO meta (key, value) VALUES (?,?)", [
        ("source", "RxNorm (US National Library of Medicine)"),
        ("source_url", "https://www.nlm.nih.gov/research/umls/rxnorm/"),
        ("license", LICENSE),
        ("attribution", ATTRIBUTION),
        ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("totals", json.dumps({
            "ingredients": len(ingredients),
            "names": len(names),
            "brands": sum(1 for n in names if n[4] == "brand"),
            "products": sum(1 for n in names if n[4] == "product"),
            "ingredient_links": len(links),
            "rows_skipped": skipped,
        })),
    ])
    conn.commit()
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    size = db_path.stat().st_size
    digest = hashlib.sha256(db_path.read_bytes()).hexdigest()
    totals = {"ingredients": len(ingredients), "names": len(names),
              "brands": sum(1 for n in names if n[4] == "brand"),
              "products": sum(1 for n in names if n[4] == "product"),
              "ingredient_links": len(links), "rows_skipped": skipped}
    return {"db": str(db_path), "size_bytes": size, "sha256": digest,
            "totals": totals, "seconds": round(time.time() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--out", type=Path, default=DB_PATH)
    args = ap.parse_args(argv)

    result = build(args.archive, args.out)
    print(f"Wrote {result['db']} ({result['size_bytes'] / 1e6:.1f} MB) "
          f"in {result['seconds']}s")
    print(f"  sha256 {result['sha256']}")
    for k, v in result["totals"].items():
        print(f"  {k:<16} {v:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
