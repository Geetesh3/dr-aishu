"""Download the RxNorm release used by the medicine reference database.

RxNorm is published by the US National Library of Medicine. The full monthly
release is ~73 MB zipped; the archive is kept in `data/external/medicines/`
(gitignored) and only the built SQLite database is read by the app.

Run:
    python scripts/fetch_medicine_sources.py
    python scripts/fetch_medicine_sources.py --url <release url>
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
OUT_DIR = ROOT / "data" / "external" / "medicines"
ARCHIVE = OUT_DIR / "rxnorm_full_prescribe.zip"
MANIFEST = OUT_DIR / "medicine_sources.json"

# NLM publishes the monthly release under a dated filename. Pinning a release
# keeps builds reproducible; update the URL deliberately rather than silently
# picking up whatever is current.
DEFAULT_URL = "https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_01062025.zip"
UA = "Mozilla/5.0 (compatible; DrAishuDataBot/1.0; educational use)"


def fetch(url: str = DEFAULT_URL, out: Path = ARCHIVE) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    digest = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=180) as resp, open(out, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            fh.write(chunk)
    record = {
        "name": "RxNorm Full Monthly Prescribable Release",
        "publisher": "U.S. National Library of Medicine",
        "url": url,
        "license": "Public domain (NLM-created names and codes); full release bundles restricted sources",
        "homepage": "https://www.nlm.nih.gov/research/umls/rxnorm/",
        "file": out.name,
        "size_bytes": out.stat().st_size,
        "sha256": digest.hexdigest(),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - started, 1),
        "note": "Only SAB=RXNORM rows are imported; bundled proprietary terminologies are not.",
    }
    existing = []
    if MANIFEST.exists():
        try:
            existing = json.loads(MANIFEST.read_text()).get("sources", [])
        except (json.JSONDecodeError, OSError):
            existing = []
    existing = [s for s in existing if s.get("file") != record["file"]] + [record]
    MANIFEST.write_text(json.dumps({"sources": existing}, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", type=Path, default=ARCHIVE)
    args = ap.parse_args(argv)

    if args.out.exists():
        print(f"Already downloaded: {args.out} "
              f"({args.out.stat().st_size / 1e6:.1f} MB). Delete it to refetch.")
        return 0
    try:
        rec = fetch(args.url, args.out)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    print(f"Downloaded {rec['file']} ({rec['size_bytes'] / 1e6:.1f} MB) "
          f"in {rec['seconds']}s")
    print(f"  sha256 {rec['sha256']}")
    print(f"  manifest {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
