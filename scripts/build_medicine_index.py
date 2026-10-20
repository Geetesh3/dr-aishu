"""Build the full medicine product index from the Indian pharmacy catalogue.

The earlier brand layer kept a handful of brands per generic because it exists to
decorate a suggested medicine. This builder keeps *every* product, so the app can
answer "does this exist, what is it called, who makes it and what does it cost"
across the whole catalogue rather than a sample.

Scope and honesty
-----------------
The source is a product catalogue of 253,974 real, marketed Indian products. It
is not 2.5 million clinical medicines, and no such dataset exists: the largest
open clinical sources are openFDA at roughly 262,000 labels and RxNorm at roughly
100,000 clinical drugs. Counting every product, strength, pack and manufacturer
across every market is an estimate, not data. So this file records exactly what
it holds and the API reports that number, rather than inflating it.

What the index is used for
--------------------------
Search and brand lookup only. It carries no indication, dose or contraindication
data, so it can never decide which medicine is suggested. Clinical guidance stays
in the curated ``medicines.json``.

Output format
-------------
Grouped by molecule and stored as positional arrays, which cuts the file to a
fraction of the size of one object per product:

    {"molecules": {"paracetamol": [[name, manufacturer, pack, price, form, disc], ...]}}

Usage:
    python scripts/build_medicine_index.py path/to/indian_medicine_data.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "app" / "data" / "medicine_index.json"
DATASET_URL = (
    "https://raw.githubusercontent.com/junioralive/Indian-Medicine-Dataset/"
    "main/DATA/indian_medicine_data.csv"
)

# Salt and ester suffixes stripped so "chlorpheniramine maleate" resolves to
# "chlorpheniramine" and products group under one molecule.
SALT_SUFFIXES = (
    "hydrochloride", "hydrobromide", "maleate", "oxalate", "sodium", "potassium",
    "calcium", "magnesium", "sulfate", "sulphate", "succinate", "tartrate",
    "fumarate", "citrate", "phosphate", "acetate", "besylate", "mesylate",
    "nitrate", "chloride", "bromide", "dihydrate", "monohydrate", "trihydrate",
    "anhydrous", "proxetil", "pivoxil", "medoxomil", "decanoate", "propionate",
    "dipropionate", "valerate", "furoate", "hemihydrate", "bitartrate",
)

FORM_KEYWORDS = (
    ("injection", ("injection", "infusion", "vial", "ampoule", "syringe", "prefilled")),
    ("inhaler", ("inhaler", "rotacap", "respule", "spray", "inhalation")),
    ("eye", ("eye drop", "eye ointment", "ophthalmic", "eye/ear")),
    ("ear", ("ear drop", "otic")),
    ("nasal", ("nasal", "nose drop")),
    ("topical", ("cream", "ointment", "gel", "lotion", "patch", "dusting powder", "shampoo", "soap")),
    ("sublingual", ("sublingual",)),
    ("oral", (
        "tablet", "capsule", "syrup", "suspension", "sachet", "granule",
        "drops", "solution", "elixir", "dispersible", "mouth", "oral",
    )),
)


def base_molecule(raw: str) -> str:
    """Reduce a composition string to a bare molecule name.

    "Amoxycillin  (500mg) " -> "amoxycillin"
    "Chlorpheniramine Maleate" -> "chlorpheniramine"
    """
    text = (raw or "").strip().lower()
    if not text:
        return ""
    text = text.split(",")[0]
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|%|w/v|w/w)\b", "", text)
    text = re.sub(r"[^a-z\s\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    while len(words) > 1 and words[-1] in SALT_SUFFIXES:
        words.pop()
    return " ".join(words)


def form_of(pack: str, name: str) -> str:
    text = f"{pack or ''} {name or ''}".lower()
    for form, keys in FORM_KEYWORDS:
        if any(k in text for k in keys):
            return form
    return "unknown"


def is_combination(comp1: str, comp2: str) -> bool:
    """True when the product holds more than one active molecule.

    Combinations are indexed under their first molecule but flagged, so search
    can show them without presenting a fixed-dose combination as if it were the
    plain single-ingredient product.
    """
    second = (comp2 or "").strip().lower()
    if second and second not in ("na", "n/a", "-", "0"):
        return True
    first = (comp1 or "").strip().lower()
    return "," in first or "+" in first or " and " in first


def _price(raw: str) -> float | None:
    try:
        value = float(str(raw).strip())
        return round(value, 2) if value > 0 else None
    except (TypeError, ValueError):
        return None


def build(csv_path: Path) -> dict:
    molecules: dict[str, list] = defaultdict(list)
    total = matched = 0
    no_composition = 0

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            total += 1
            comp1 = (row.get("short_composition1") or "").strip()
            comp2 = (row.get("short_composition2") or "").strip()
            molecule = base_molecule(comp1)
            if not molecule:
                no_composition += 1
                continue
            name = (row.get("name") or "").strip()
            if not name:
                continue
            matched += 1
            combo = is_combination(comp1, comp2)
            molecules[molecule].append([
                name,
                (row.get("manufacturer_name") or "").strip()[:60],
                (row.get("pack_size_label") or "").strip()[:70],
                _price(row.get("price(₹)")),
                form_of(row.get("pack_size_label", ""), name),
                1 if str(row.get("Is_discontinued", "")).strip().lower() == "true" else 0,
                1 if combo else 0,
            ])

    return {
        "source": DATASET_URL,
        "note": (
            "Full product index: every product in the source catalogue, grouped by "
            "molecule. Search and brand lookup only -- no indication, dose or "
            "contraindication data, so it never influences which medicine is "
            "suggested. Clinical guidance lives in medicines.json."
        ),
        "count": matched,
        "rows_read": total,
        "rows_without_composition": no_composition,
        "molecule_count": len(molecules),
        "columns": ["name", "manufacturer", "pack", "price_inr", "form", "discontinued", "combination"],
        "molecules": molecules,
    }


def _sort_key(row: list) -> tuple:
    """Best-first within a molecule: single-ingredient, live, then cheapest.

    Pre-sorted here so the search endpoint can take the first N rows of a
    molecule and stop, instead of collecting thousands and sorting per request.
    """
    price = row[3]
    return (row[6], row[5], price if price is not None else 1e9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", type=Path, help="path to indian_medicine_data.csv")
    ap.add_argument("--download", action="store_true", help="fetch the catalogue first")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    csv_path = args.csv
    if args.download or csv_path is None:
        csv_path = Path("/tmp/indian_medicine_data.csv")
        if not csv_path.exists() or args.download:
            print(f"downloading {DATASET_URL}")
            urllib.request.urlretrieve(DATASET_URL, csv_path)

    data = build(csv_path)
    # Sort each molecule best-first so search can slice instead of sort.
    for rows in data["molecules"].values():
        rows.sort(key=_sort_key)
    # Sort molecules alphabetically so identical queries rank identically.
    data["molecules"] = {k: data["molecules"][k] for k in sorted(data["molecules"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))

    size_mb = args.out.stat().st_size / 1_048_576
    print(f"rows read          : {data['rows_read']}")
    print(f"products indexed   : {data['count']}")
    print(f"distinct molecules : {data['molecule_count']}")
    print(f"without a molecule : {data['rows_without_composition']}")
    print(f"written            : {args.out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
