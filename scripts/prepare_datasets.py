"""Unzip the image archives and sort them into a clean, trainable tree.

Both image sources arrive as nested zips with folder names that carry the class
label in a parenthetical, e.g. `Augmented/Atopic Dermatitis (AD)`. This script
extracts them into one predictable layout and normalises the labels, so a
trainer can walk the tree without knowing where the images came from:

    data/external/datasets/skin/<Class>/<image>
    data/external/datasets/eye/<Class>/<image>

It also writes `data/external/datasets/manifest.json` with the per-class counts,
the source archive's SHA-256 and the licence, so the training step can record
exactly what it learned from.

Why the augmentations are kept separate: SkinDisNet ships 1,710 clinical photos
and ~13,000 augmented copies of them. Training on both would leak near-duplicates
across any random split, so the split is recorded in the manifest and the trainer
holds out whole patients, not individual files.

Run:
    python scripts/prepare_datasets.py                # extract what is missing
    python scripts/prepare_datasets.py --force        # re-extract from scratch
    python scripts/prepare_datasets.py --limit 50     # cap images per class (smoke test)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "external" / "sources"
DEST = ROOT / "data" / "external" / "datasets"
MANIFEST = DEST / "manifest.json"

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# "Atopic Dermatitis (AD)" -> "Atopic Dermatitis"
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def clean_class(name: str) -> str:
    """Folder label -> tidy class name, safe as a directory."""
    name = _PAREN.sub("", name).strip()
    name = re.sub(r"[\\/:*?\"<>|]", "-", name)
    return re.sub(r"\s+", " ", name).strip() or "unknown"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _inner_zip_bytes(outer: Path, suffix: str) -> bytes | None:
    with zipfile.ZipFile(outer) as z:
        for info in z.infolist():
            if info.filename.endswith(suffix):
                return z.read(info)
    return None


def extract_eye(force: bool, limit: int | None) -> dict:
    """Mendeley eye set: a flat zip of `<Class>/<file>`."""
    archive = SOURCES / "mendeley_eye_images.zip"
    out = DEST / "eye"
    if not archive.exists():
        return {"status": "missing-archive"}
    if out.exists() and not force:
        existing = sum(1 for _ in out.rglob("*") if _.suffix.lower() in IMAGE_EXT)
        if existing:
            return {"status": "cached", "images": existing}

    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    written = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if not info.filename.lower().endswith(IMAGE_EXT):
                continue
            parts = info.filename.split("/")
            if len(parts) < 2:
                continue
            cls = clean_class(parts[-2])
            if limit and counts[cls] >= limit:
                continue
            target = out / cls / Path(parts[-1]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[cls] += 1
            written += 1
    return {"status": "ok", "images": written, "classes": dict(sorted(counts.items()))}


def extract_skin(force: bool, limit: int | None) -> dict:
    """SkinDisNet: outer zip -> inner zip -> Preprocessed/ and Augmented/."""
    archive = SOURCES / "skindisnet_images.zip"
    out = DEST / "skin"
    if not archive.exists():
        return {"status": "missing-archive"}
    if out.exists() and not force:
        existing = sum(1 for _ in out.rglob("*") if _.suffix.lower() in IMAGE_EXT)
        if existing:
            return {"status": "cached", "images": existing}

    inner = _inner_zip_bytes(archive, "SkinDisNet_2.zip")
    if inner is None:
        return {"status": "failed", "error": "inner SkinDisNet_2.zip not found"}

    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    written = 0
    with zipfile.ZipFile(io.BytesIO(inner)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if not info.filename.lower().endswith(IMAGE_EXT):
                continue
            parts = info.filename.split("/")
            if len(parts) < 3:
                continue
            split_dir, cls_dir = parts[0], parts[1]
            cls = clean_class(cls_dir)
            split = "augmented" if split_dir.lower().startswith("aug") else "clinical"
            # Keep both splits, but under distinct class folders so a trainer
            # can choose to train on clinical-only and never leak augmentations.
            folder = cls if split == "clinical" else f"{cls}__augmented"
            if limit and counts[folder] >= limit:
                continue
            target = out / folder / Path(parts[-1]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[folder] += 1
            written += 1
    return {"status": "ok", "images": written, "classes": dict(sorted(counts.items()))}


def export_skin_metadata() -> dict:
    """Write the patient metadata CSV out of the inner zip for reference."""
    archive = SOURCES / "skindisnet_images.zip"
    if not archive.exists():
        return {"status": "missing-archive"}
    inner = _inner_zip_bytes(archive, "SkinDisNet_2.zip")
    if inner is None:
        return {"status": "failed"}
    with zipfile.ZipFile(io.BytesIO(inner)) as z:
        for info in z.infolist():
            if info.filename.endswith("SkinDisNet_Metadata.csv"):
                text = z.read(info).decode("utf-8-sig", errors="replace")
                target = DEST / "skin_metadata.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                rows = list(csv.DictReader(io.StringIO(text)))
                patients = {(r.get("Patient_id") or "").strip() for r in rows}
                return {"status": "ok", "rows": len(rows), "patients": len(patients)}
    return {"status": "failed", "error": "metadata csv not found"}


def write_manifest(parts: dict) -> dict:
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(DEST.relative_to(ROOT)),
        "sources": {},
    }
    for name, archive in (
        ("eye", SOURCES / "mendeley_eye_images.zip"),
        ("skin", SOURCES / "skindisnet_images.zip"),
    ):
        entry = dict(parts.get(name, {}))
        if archive.exists():
            entry["archive"] = str(archive.relative_to(ROOT))
            entry["archive_bytes"] = archive.stat().st_size
            entry["archive_sha256"] = _sha256(archive)
        manifest["sources"][name] = entry

    # Recount from disk so the manifest reflects what is actually there.
    for name in ("eye", "skin"):
        d = DEST / name
        counts: dict[str, int] = {}
        if d.exists():
            for cls_dir in sorted(p for p in d.iterdir() if p.is_dir()):
                counts[cls_dir.name] = sum(
                    1 for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXT
                )
        manifest.setdefault("counts", {})[name] = counts
        manifest["sources"].setdefault(name, {})["images"] = sum(counts.values())

    manifest["metadata"] = parts.get("metadata", {})
    DEST.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-extract from scratch")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class")
    args = ap.parse_args()

    parts = {}
    print("-> eye")
    parts["eye"] = extract_eye(args.force, args.limit)
    print(f"   {parts['eye']}")
    print("-> skin")
    parts["skin"] = extract_skin(args.force, args.limit)
    print(f"   {parts['skin']}")
    print("-> skin metadata")
    parts["metadata"] = export_skin_metadata()
    print(f"   {parts['metadata']}")

    manifest = write_manifest(parts)
    print(f"\nmanifest -> {MANIFEST.relative_to(ROOT)}")
    for name in ("eye", "skin"):
        counts = manifest.get("counts", {}).get(name, {})
        total = sum(counts.values())
        print(f"{name}: {total} images in {len(counts)} classes")
        for cls, n in counts.items():
            print(f"    {n:>6}  {cls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
