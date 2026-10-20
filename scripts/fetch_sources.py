"""Download every external source Dr. Aishu's dataset is built from.

Each source is fetched once into `data/external/sources/` and recorded in
`sources.json` with its URL, licence, byte size, SHA-256 and fetch time. The
Excel builder reads that manifest, so every row in the workbook can be traced
back to the exact bytes it came from.

Licences are recorded but not guessed: where a source publishes no licence the
manifest says `unknown` rather than assuming one. That matters because two of
these sources (the GitHub symptom repo and the Mendeley eye-image set) carry no
licence at all, and MedicineNet's robots.txt explicitly disallows the AI
crawlers, so its text is reference-only and never training data.

Run:
    python scripts/fetch_sources.py              # fetch what is missing
    python scripts/fetch_sources.py --force      # re-fetch everything
    python scripts/fetch_sources.py --only hf    # one source
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
DEST = ROOT / "data" / "external" / "sources"
MANIFEST = DEST / "sources.json"

UA = "Mozilla/5.0 (compatible; DrAishuDatasetBot/1.0; educational use)"

# id -> how to fetch it. `kind` drives the download strategy: `http` is a plain
# file, `api` is a JSON endpoint we keep as-is, `note` records a source we
# cannot download and why.
SOURCES: list[dict] = [
    {
        "id": "hf_diseases_symptoms",
        "kind": "http",
        "url": "https://huggingface.co/datasets/QuyenAnhDE/Diseases_Symptoms/resolve/main/Diseases_Symptoms.csv",
        "filename": "hf_diseases_symptoms.csv",
        "title": "Diseases, Symptoms and Treatments (HuggingFace QuyenAnhDE)",
        "license": "unknown",
        "note": "400 diseases with symptoms AND treatments. No licence declared.",
    },
    {
        "id": "github_disease_symptom",
        "kind": "github_repo",
        "repo": "hemasriram111/Disease-Symptom-Dataset",
        "url": "https://github.com/hemasriram111/Disease-Symptom-Dataset",
        "title": "Disease-Symptom-Dataset (GitHub)",
        "license": "none-declared",
        "note": "4920-row binary symptom matrix plus descriptions, precautions and severities. Repo declares no licence.",
        "files": [
            "dataset.csv",
            "cleaned_dataset.csv",
            "symptom_Description.csv",
            "symptom_precaution.csv",
            "Symptom-severity.csv",
        ],
    },
    {
        "id": "nature_rules_ml",
        "kind": "http",
        "url": "https://www.nature.com/articles/s41598-024-69029-8",
        "filename": "nature_s41598-024-69029-8.html",
        "title": "Identifying diseases symptoms and general rules using supervised and unsupervised ML (Sci Rep 2024)",
        "license": "cc-by-4.0",
        "note": "Open-access article; its companion data lives in the GitHub and Kaggle sources.",
    },
    {
        "id": "mendeley_eye_images",
        "kind": "mendeley_zip",
        "dataset_id": "n9zp473wfw",
        "version": 1,
        "url": "https://data.mendeley.com/public-api/zip/n9zp473wfw/download/1",
        "filename": "mendeley_eye_images.zip",
        "title": "Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid)",
        "license": "unknown",
        "doi": "10.17632/n9zp473wfw.2",
        "note": "2298 eye images across four classes. No licence declared on the record.",
    },
    {
        "id": "skindisnet_images",
        "kind": "mendeley_zip",
        "dataset_id": "yj3md44hxg",
        "version": 2,
        "url": "https://data.mendeley.com/public-api/zip/yj3md44hxg/download/2",
        "filename": "skindisnet_images.zip",
        "title": "SkinDisNet: Multi-class clinical skin disease images and metadata",
        "license": "cc-by-nc-4.0",
        "doi": "10.1016/j.dib.2025.112239",
        "note": "1710 clinical skin images in 6 classes, plus patient metadata. Non-commercial.",
    },
    {
        "id": "sciencedirect_skindisnet_paper",
        "kind": "note",
        "url": "https://www.sciencedirect.com/science/article/pii/S2352340925009606",
        "title": "SkinDisNet data-in-brief paper (ScienceDirect)",
        "license": "cc-by-nc-4.0",
        "note": "Publisher blocks automated access (HTTP 403). Metadata resolved via Crossref: DOI 10.1016/j.dib.2025.112239. The dataset itself is fetched as skindisnet_images.",
    },
    {
        "id": "kaggle_symptom_notebook",
        "kind": "note",
        "url": "https://www.kaggle.com/code/geeteshkumarsahoo/medical-symptoms-text-and-audio-classification",
        "title": "Medical Symptoms Text and Audio Classification (Kaggle notebook)",
        "license": "unknown",
        "note": "Kaggle requires an authenticated session to download notebook/dataset files; no credential is configured, so the notebook is recorded as a reference only.",
    },
    {
        "id": "medicinenet_index",
        "kind": "medicinenet_index",
        "url": "https://www.medicinenet.com/diseases_and_conditions/article.htm",
        "title": "MedicineNet Diseases & Conditions index",
        "license": "copyright-webmd",
        "note": (
            "Article text is copyrighted. robots.txt disallows GPTBot/CCBot/ClaudeBot, "
            "so this text is reference-only and is never used for training."
        ),
    },
    {
        "id": "mendeley_disease_symptoms_2023",
        "kind": "http",
        "url": (
            "https://data.mendeley.com/public-files/datasets/2cxccsxydc/files/"
            "a16f2141-6e3c-466f-9ce9-28d3d8d91070/file_downloaded"
        ),
        "filename": "mendeley_disease_symptoms_2023.csv",
        "title": "Disease and symptoms dataset 2023 (Mendeley Data)",
        "license": "unknown",
        "doi": "10.17632/2cxccsxydc.1",
        "note": (
            "773 diseases x 377 binary symptom columns, 246,945 rows. The Mendeley "
            "record declares no licence. This is the raw file behind the "
            "mendeley_2023 training set; see scripts/import_symptom_dataset.py."
        ),
    },
    {
        "id": "deeplesion_metadata",
        "kind": "box_shared",
        "url": "https://nihcc.app.box.com/v/DeepLesion/folder/50715173939",
        "shared_name": "pealvgmu6kw3you44a8bc1blrluzf0hv",
        "folder": "deeplesion",
        "title": "DeepLesion - NIH Clinical Center lesion annotations and documentation",
        "license": "nih-research-use",
        "note": (
            "32,735 lesion annotations over 32,120 CT slices from 4,427 patients. "
            "Only the annotation CSV and documentation are fetched: the image "
            "archives (Images_png, ~238 GB; Key_slices.zip, ~4.9 GB) exceed this "
            "environment's disk and are deliberately not downloaded. The readme "
            "states no redistribution licence, so treat as NIH research use. "
            "Lesion boxes are radiology ground truth, not symptom text."
        ),
        "files": [
            (305569842654, "DL_info.csv"),
            (306056134060, "readme.pdf"),
            (307829610290, "changelog.txt"),
            (316932195591, "batch_download_zips.py"),
            (306055882594, "FAQ.pdf"),
        ],
    },
    {
        "id": "pmc_ids",
        "kind": "http",
        "url": "https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz",
        "filename": "PMC-ids.csv.gz",
        "title": "PMC identifier mapping (PMCID / PMID / DOI / ISSN)",
        "license": "public-domain-nlm",
        "note": (
            "The only dataset file still served from the PMC FTP root; the legacy "
            "article bundles moved to the PMC Cloud Service in 2026. Maps every "
            "PMC record across PMID, DOI and journal - an identifier table, not "
            "article text, so it carries no copyright."
        ),
    },
    {
        "id": "pmc_ml_disease_diagnosis_review",
        "kind": "http",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8950225/",
        "filename": "pmc_ml_disease_diagnosis_review.html",
        "title": "Machine-Learning-Based Disease Diagnosis: A Comprehensive Review (Healthcare 2022)",
        "license": "cc-by-4.0",
        "doi": "10.3390/healthcare10030541",
        "note": (
            "Open-access CC BY review. Its Table 18 is a catalogue of public "
            "disease datasets (UCI heart, CKD, Pima diabetes, ...); extracted to "
            "the Dataset_Catalogue sheet by scripts/extract_pmc_catalogue.py."
        ),
    },
    {
        "id": "google_health_ai",
        "kind": "note",
        "url": "https://research.google/research-areas/health-ai/",
        "title": "Google Research - Health AI",
        "license": "copyright-google",
        "note": (
            "A research-area landing page: it links to publications and blog "
            "posts but publishes no downloadable dataset and states no licence. "
            "Recorded as a reference only."
        ),
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: int = 600) -> int:
    """Stream a URL to disk, returning the byte count. Raises on HTTP errors."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size


def fetch_http(src: dict, force: bool) -> dict:
    dest = DEST / src["filename"]
    if dest.exists() and not force:
        return {"status": "cached", "bytes": dest.stat().st_size}
    try:
        n = _download(src["url"], dest)
        return {"status": "ok", "bytes": n}
    except urllib.error.HTTPError as exc:
        return {"status": "failed", "error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        return {"status": "failed", "error": str(exc)}


def fetch_github_repo(src: dict, force: bool) -> dict:
    out = DEST / "github"
    out.mkdir(parents=True, exist_ok=True)
    raw = f"https://raw.githubusercontent.com/{src['repo']}/HEAD/"
    files, failed = [], []
    for name in src["files"]:
        dest = out / name
        if dest.exists() and not force:
            files.append(name)
            continue
        try:
            _download(raw + name, dest, timeout=180)
            files.append(name)
        except Exception as exc:  # noqa: BLE001
            failed.append({"file": name, "error": str(exc)})
    result = {"status": "ok" if files else "failed", "files": files}
    if failed:
        result["failed"] = failed
    return result


def fetch_mendeley_zip(src: dict, force: bool) -> dict:
    dest = DEST / src["filename"]
    if dest.exists() and not force:
        return {"status": "cached", "bytes": dest.stat().st_size}
    try:
        n = _download(src["url"], dest, timeout=1800)
        return {"status": "ok", "bytes": n}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)}


def fetch_medicinenet_index(src: dict, force: bool) -> dict:
    """Fetch the 25 A-Z index pages. Article crawling is a separate script."""
    out = DEST / "medicinenet" / "alpha"
    out.mkdir(parents=True, exist_ok=True)
    got, failed = 0, []
    for letter in "abcdefghijklmnopqrstuvwxy":
        dest = out / f"alpha_{letter}.htm"
        if dest.exists() and not force:
            got += 1
            continue
        url = f"https://www.medicinenet.com/diseases_and_conditions/alpha_{letter}.htm"
        try:
            _download(url, dest, timeout=60)
            got += 1
        except Exception as exc:  # noqa: BLE001
            failed.append({"letter": letter, "error": str(exc)})
        time.sleep(1)  # polite gap between index pages
    return {"status": "ok" if got else "failed", "pages": got, "failed": failed}


def fetch_box_shared(src: dict, force: bool) -> dict:
    """Download individual files from a public Box shared folder.

    Box has no anonymous REST API for shared folders: the web app posts to
    `box_download_shared_file` with the folder's `shared_name` token (the
    opaque string in the page's `Box.postStreamData`). We read that token from
    the source config, which is why it is pinned above rather than scraped.
    """
    out = DEST / src.get("folder", src["id"])
    out.mkdir(parents=True, exist_ok=True)
    shared = src["shared_name"]
    endpoint = "https://nihcc.app.box.com/index.php?rm=box_download_shared_file"
    files, failed = [], []
    for file_id, name in src["files"]:
        dest = out / name
        if dest.exists() and not force:
            files.append(name)
            continue
        url = f"{endpoint}&shared_name={shared}&file_id=f_{file_id}"
        try:
            _download(url, dest, timeout=600)
            files.append(name)
        except Exception as exc:  # noqa: BLE001
            failed.append({"file": name, "error": str(exc)})
        time.sleep(1)  # polite gap between Box downloads
    result = {"status": "ok" if files else "failed", "files": files}
    if failed:
        result["failed"] = failed
    return result


HANDLERS = {
    "http": fetch_http,
    "github_repo": fetch_github_repo,
    "mendeley_zip": fetch_mendeley_zip,
    "medicinenet_index": fetch_medicinenet_index,
    "box_shared": fetch_box_shared,
}


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"generated_at": None, "sources": {}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-fetch even if present")
    ap.add_argument("--only", action="append", default=None, help="source id(s) to fetch")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    entries = manifest.setdefault("sources", {})

    selected = [s for s in SOURCES if not args.only or s["id"] in args.only]
    if not selected:
        print(f"No source matched {args.only}", file=sys.stderr)
        return 2

    failures = 0
    for src in selected:
        sid = src["id"]
        print(f"-> {sid}")
        handler = HANDLERS.get(src["kind"])
        if handler is None:
            result = {"status": "reference-only", "note": src.get("note")}
        else:
            result = handler(src, args.force)

        entry = {
            "id": sid,
            "title": src["title"],
            "url": src["url"],
            "license": src["license"],
            "note": src.get("note"),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **result,
        }
        if src.get("doi"):
            entry["doi"] = src["doi"]

        # Hash whatever landed on disk so the Excel build can prove provenance.
        if result.get("status") in ("ok", "cached"):
            fname = src.get("filename")
            if fname and (DEST / fname).exists():
                p = DEST / fname
                entry["sha256"] = _sha256(p)
                entry["bytes"] = p.stat().st_size
            elif src["kind"] == "github_repo":
                entry["sha256"] = {
                    f: _sha256(DEST / "github" / f) for f in result.get("files", [])
                }
            elif src["kind"] == "box_shared":
                folder = DEST / src.get("folder", sid)
                entry["sha256"] = {
                    f: _sha256(folder / f)
                    for f in result.get("files", [])
                    if (folder / f).exists()
                }
                entry["bytes"] = sum(
                    (folder / f).stat().st_size for f in result.get("files", [])
                    if (folder / f).exists()
                )

        entries[sid] = entry
        flag = result.get("status")
        extra = result.get("bytes") or result.get("pages") or ""
        print(f"   {flag} {extra}")
        if flag == "failed":
            failures += 1

    manifest["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmanifest -> {MANIFEST.relative_to(ROOT)}")
    print(f"{len(selected) - failures}/{len(selected)} sources ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
