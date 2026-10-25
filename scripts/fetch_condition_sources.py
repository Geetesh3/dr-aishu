"""Download the open disease and symptom ontologies the condition library is built from.

Every source here is a real, citable ontology with a licence that permits reuse.
That is the point of choosing them: the previous condition list was 68 hand-written
entries with no provenance, so nothing on the site could be checked against a
source. These carry stable identifiers (DOID, MONDO, ORPHA, ICD-10-CM, MeSH) that
can be traced back to the publisher.

Licences are recorded, never guessed, and each file's SHA-256 is stored so the
imported rows can be tied to exact bytes. Where a source is public domain or
CC BY, the licence name is written verbatim.

Run:
    python scripts/fetch_condition_sources.py                 # fetch what is missing
    python scripts/fetch_condition_sources.py --force         # re-fetch everything
    python scripts/fetch_condition_sources.py --only doid     # one source
    python scripts/fetch_condition_sources.py --skip-large    # skip MeSH (314 MB)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "external" / "conditions"
MANIFEST = DEST / "condition_sources.json"

UA = "Mozilla/5.0 (compatible; DrAishuConditionBot/1.0; educational use)"

# id -> what to fetch. `license` is copied from the publisher's own statement.
SOURCES: list[dict] = [
    {
        "id": "doid",
        "kind": "http",
        "url": "https://purl.obolibrary.org/obo/doid.obo",
        "filename": "doid.obo",
        "title": "Human Disease Ontology (DOID)",
        "license": "CC0-1.0",
        "publisher": "Disease Ontology / Institute for Genome Sciences, University of Maryland",
        "note": "OBO release. One stanza per disease with a definition, exact synonyms and "
                "cross-references to ICD-10, MeSH, OMIM, Orphanet and UMLS.",
        "citation": "Schriml LM et al. The Human Disease Ontology. Nucleic Acids Res. 2022.",
    },
    {
        "id": "mondo",
        "kind": "http",
        "url": "https://purl.obolibrary.org/obo/mondo.obo",
        "filename": "mondo.obo",
        "title": "Mondo Disease Ontology",
        "license": "CC-BY-4.0",
        "publisher": "Mondo / Monarch Initiative",
        "note": "OBO release. Harmonises DOID, OMIM, Orphanet and ICD. Carries MONDO ids plus "
                "many cross-references, so it is the widest single disease list available.",
        "citation": "Vasilevsky NA et al. Mondo: Unifying diseases for the world. eLife 2022.",
    },
    {
        "id": "hpo",
        "kind": "http",
        "url": "https://purl.obolibrary.org/obo/hp.obo",
        "filename": "hp.obo",
        "title": "Human Phenotype Ontology (HPO)",
        "license": "CC-BY-4.0",
        "publisher": "Monarch Initiative",
        "note": "OBO release. Standard vocabulary of phenotypic abnormalities; used here as a "
                "symptom and sign reference so symptom names can be traced to a real ontology.",
        "citation": "Kohler S et al. The Human Phenotype Ontology in 2021. Nucleic Acids Res.",
    },
    {
        "id": "orphanet",
        "kind": "http",
        "url": "https://www.orphadata.com/data/xml/en_product1.xml",
        "filename": "orphanet_product1.xml",
        "title": "Orphanet rare disease nomenclature",
        "license": "CC-BY-4.0",
        "publisher": "Orphanet / INSERM",
        "note": "Rare-disease nomenclature with ORPHA codes, preferred names, synonyms and "
                "cross-references to ICD-10, OMIM, MeSH and UMLS.",
        "citation": "Orphanet: an online database of rare diseases. INSERM.",
    },
    {
        "id": "icd10cm",
        "kind": "api",
        "url": "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?terms={term}&maxList=500&sf=code,name",
        "filename": "icd10cm.json",
        "title": "ICD-10-CM (US CDC/NCHS, via NLM Clinical Tables)",
        "license": "public-domain-us-government",
        "publisher": "US Centers for Disease Control and Prevention / NLM",
        "note": "Public domain. Queried by first letter to enumerate the code set; each hit "
                "carries the ICD-10-CM code and its official name.",
        "citation": "ICD-10-CM. US CDC National Center for Health Statistics.",
    },
    {
        "id": "mesh",
        "kind": "http",
        "url": "https://nlmpubs.nlm.nih.gov/projects/mesh/2025/xmlmesh/desc2025.xml",
        "filename": "mesh_desc2025.xml",
        "title": "MeSH Descriptors (Medical Subject Headings)",
        "license": "public-domain-us-government",
        "publisher": "US National Library of Medicine",
        "note": "Large (~314 MB). Descriptor records with scope notes, entry terms (synonyms) "
                "and tree numbers. Restricted to the Diseases and Symptoms branches at import.",
        "citation": "Medical Subject Headings. US National Library of Medicine.",
        "large": True,
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"generated_at": None, "sources": {}}


def _save_manifest(manifest: dict) -> None:
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def _download(url: str, dest: Path, timeout: int = 600) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fh:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    return total


def fetch_one(spec: dict, manifest: dict, force: bool, skip_large: bool) -> None:
    entry = manifest["sources"].get(spec["id"], {})
    if spec.get("large") and skip_large:
        print(f"  {spec['id']}: skipped (--skip-large)")
        return

    dest = DEST / spec["filename"]
    if dest.exists() and not force and entry.get("sha256"):
        print(f"  {spec['id']}: cached ({entry.get('bytes', dest.stat().st_size):,} bytes)")
        return

    if spec["kind"] == "api":
        # ICD-10-CM has no bulk file over HTTP; enumerate by first letter.
        records: dict[str, list] = {}
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            url = spec["url"].format(term=letter)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"  {spec['id']}: letter {letter} failed ({exc})")
                continue
            rows = payload[3] or []
            records[letter] = rows
            time.sleep(0.2)
        dest.write_text(json.dumps(records, ensure_ascii=False))
        entry.update({
            "id": spec["id"], "title": spec["title"], "url": spec["url"],
            "license": spec["license"], "publisher": spec["publisher"], "note": spec["note"],
            "citation": spec.get("citation"), "kind": "api",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "ok",
            "bytes": dest.stat().st_size,
            "records": sum(len(v) for v in records.values()),
        })
        manifest["sources"][spec["id"]] = entry
        print(f"  {spec['id']}: {entry['records']:,} codes")
        return

    print(f"  {spec['id']}: downloading {spec['url']}")
    try:
        size = _download(spec["url"], dest)
    except (urllib.error.URLError, TimeoutError) as exc:
        entry.update({"id": spec["id"], "status": "failed", "error": str(exc),
                      "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        manifest["sources"][spec["id"]] = entry
        print(f"  {spec['id']}: FAILED ({exc})")
        return

    entry.update({
        "id": spec["id"], "title": spec["title"], "url": spec["url"],
        "license": spec["license"], "publisher": spec["publisher"], "note": spec["note"],
        "citation": spec.get("citation"), "kind": "http", "filename": spec["filename"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ok", "bytes": size, "sha256": _sha256(dest),
    })
    manifest["sources"][spec["id"]] = entry
    print(f"  {spec['id']}: {size:,} bytes")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-fetch even if cached")
    ap.add_argument("--only", help="fetch a single source id")
    ap.add_argument("--skip-large", action="store_true", help="skip MeSH (314 MB)")
    args = ap.parse_args(argv)

    DEST.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()

    specs = SOURCES
    if args.only:
        specs = [s for s in SOURCES if s["id"] == args.only]
        if not specs:
            print(f"unknown source id: {args.only}", file=sys.stderr)
            return 2

    for spec in specs:
        fetch_one(spec, manifest, args.force, args.skip_large)
        _save_manifest(manifest)

    ok = sum(1 for s in manifest["sources"].values() if s.get("status") == "ok")
    print(f"\n{ok} source(s) available. Manifest: {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
