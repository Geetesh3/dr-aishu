"""Import the Indian medicine brand catalogue into a compact brand index.

Source: https://github.com/junioralive/Indian-Medicine-Dataset
File:   DATA/indian_medicine_data.csv (~254k rows)

That dataset is a *pharmacy product catalogue*: brand name, price, manufacturer,
pack size and generic composition. It carries **no indication, symptom or
disease column at all**.

That shapes the merge. Dr. Aishu's clinical layer (uses, dose, contraindications,
OTC status, which conditions a medicine treats) stays exactly as it is, curated
in ``app/data/medicines.json``. The catalogue is merged as a *brand layer* on top
of it, answering "what is this generic actually called in an Indian pharmacy, and
what does it cost?". We never derive an indication from the catalogue, because
inventing one would let the app recommend, say, an anti-diabetic for a headache.

Usage:
    python scripts/import_indian_medicines.py path/to/indian_medicine_data.csv
    python scripts/import_indian_medicines.py --download
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

MEDICINES_PATH = ROOT / "app" / "data" / "medicines.json"
OUT_PATH = ROOT / "app" / "data" / "indian_brands.json"

DATASET_URL = (
    "https://raw.githubusercontent.com/junioralive/Indian-Medicine-Dataset/"
    "main/DATA/indian_medicine_data.csv"
)

# The catalogue spells several generics the Indian way, or appends a salt, while
# the curated KB uses plain INN. Without these aliases real matches are silently
# lost. Note the KB has genuine omeprazole/ranitidine entries, so those must NOT
# be folded into the generic "antacid" bucket -- that would mislabel a PPI.
COMPOSITION_ALIASES = {
    "amoxycillin": "amoxicillin",
    "clavulanic acid": "clavulanate",
    "acetaminophen": "paracetamol",
    "ferrous ascorbate": "ferrous sulfate",
    "ferrous fumarate": "ferrous sulfate",
    "levocetirizine": "cetirizine",
    "lactobacillus": "probiotic",
    "saccharomyces boulardii": "probiotic",
    "zinc sulphate": "zinc sulfate/gluconate",
    "zinc gluconate": "zinc sulfate/gluconate",
    "methylcobalamin": "cyanocobalamin",
    "insulin glargine": "insulin",
    "insulin aspart": "insulin",
    "human insulin": "insulin",
    "levosalbutamol": "salbutamol",
    "albuterol": "salbutamol",
    "dexpanthenol": "emollient",
    "white petrolatum": "emollient",
    "petrolatum": "emollient",
    "liquid paraffin": "emollient",
    "glycerin": "emollient",
    "ispaghula": "ispaghula husk",
    "plantago ovata": "ispaghula husk",
    "ors": "oral rehydration salts",
    "vitamin c": "ascorbic acid",
    "cranberry": "cranberry extract",
    "artemether": "artemether-lumefantrine",
    "hydroxypropyl methylcellulose": "hypromellose",
    "carboxymethylcellulose": "hypromellose",
    "penicillin v": "phenoxymethylpenicillin",
    "fluticasone propionate": "fluticasone",
    "nitroglycerin": "glyceryl trinitrate",
    "glucosamine": "glucosamine sulfate",
    "vaccinium macrocarpon": "cranberry extract",
}

# Salt / ester suffixes the catalogue appends to the molecule name.
_SALT_SUFFIXES = (
    "hydrochloride", "hydrobromide", "maleate", "oxalate", "sodium", "potassium",
    "calcium", "magnesium", "sulfate", "sulphate", "succinate", "tartrate",
    "fumarate", "citrate", "phosphate", "acetate", "besylate", "mesylate",
    "nitrate", "chloride", "bromide", "dihydrate", "monohydrate", "trihydrate",
    "anhydrous", "proxetil", "pivoxil", "medoxomil", "trihydrate",
)

# Rows per generic to keep. The catalogue lists hundreds of near-identical
# brands per molecule; a handful of the best-known live ones is what a user needs.
MAX_BRANDS_PER_GENERIC = 25

# Ranking purely by price surfaces obscure labels ("Babymol", "Dagmol") that no
# pharmacy counter recognises. These major manufacturers are matched first, so the
# list reads like what a patient would actually see on the shelf.
MAJOR_MANUFACTURERS = (
    "sun pharmaceutical", "cipla", "intas", "torrent", "alkem", "abbott",
    "zydus", "cadila", "lupin", "mankind", "dr. reddy", "dr reddy", "glenmark",
    "micro labs", "usv", "ajanta", "teva", "emcure", "hetero", "natco",
    "wockhardt", "ipca", "jb chemicals", "eris", "indoco", "blue cross",
    "alkem laboratories", "himalaya", "dabur", "gsk", "glaxo", "pfizer",
    "sanofi", "novartis", "ranbaxy", "aurobindo", "granules", "alembic",
)


def _manufacturer_rank(name: str) -> int:
    low = (name or "").lower()
    return 0 if any(m in low for m in MAJOR_MANUFACTURERS) else 1


def _is_combination(comp1: str, comp2: str) -> bool:
    """True when the product contains more than one active molecule.

    These must not be listed as equivalents of a single-ingredient generic.
    "Coldwok Tablet" is paracetamol *plus* phenylpropanolamine; presenting it as
    plain paracetamol would mislead a user into taking a decongestant that is
    contraindicated in hypertension, and the curated entry has no data about the
    second ingredient.
    """
    second = (comp2 or "").strip().lower()
    if second and second not in ("na", "n/a", "-", "0"):
        return True
    first = (comp1 or "").strip().lower()
    if "," in first or "+" in first or " and " in first:
        return True
    return False


# Dosage forms are matched to the route the curated entry actually describes.
# A blanket "no injections" rule would be wrong in both directions: it would drop
# the insulin vial (the only form insulin comes in) and it would keep the
# paracetamol vial, which the app's oral "500-1000 mg" advice does not cover.
ROUTE_BY_GENERIC = {
    "clotrimazole": "topical",
    "emollient": "topical",
    "hydrocortisone": "topical",
    "salbutamol": "inhaler",
    "budesonide": "inhaler",
    "fluticasone": "nasal",
    "hypromellose": "eye",
    "xylometazoline": "nasal",
    "glyceryl trinitrate": "sublingual",
    "insulin": "injection",
}

_FORM_KEYWORDS = {
    "injection": ("injection", "infusion", "vial", "ampoule", "syringe"),
    "inhaler": ("inhaler", "rotacap", "respule", "spray"),
    "eye": ("eye drop", "eye ointment", "ophthalmic"),
    "ear": ("ear drop", "otic"),
    "nasal": ("nasal", "nose drop"),
    "topical": ("cream", "ointment", "gel", "lotion", "patch", "dusting powder"),
    "sublingual": ("sublingual",),
    "oral": (
        "tablet", "capsule", "syrup", "suspension", "sachet", "granule",
        "drops", "solution", "elixir", "dispersible", "mouth",
    ),
}


def _brand_stem(name: str) -> str:
    """The marketing name of a product, without strength or dosage form.

    "Dolo 650 Tablet" -> "dolo"; "Crocin Advance Tablet" -> "crocin".
    """
    text = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    words = [w for w in text.split() if not w.isdigit()]
    return words[0] if words else ""


# Brands Indian pharmacies and patients actually recognise by name. Ordering by
# price alone buries these under hundreds of equally-cheap labels, and a
# frequency heuristic does worse -- it surfaces store-brand SKU spam such as
# "StayHappi Azithromycin" and "DavaIndia ...", which are catalogue artefacts
# rather than names anyone asks for at a counter.
#
# Every entry here was verified to exist in the source catalogue, so this only
# reorders real rows; it never invents a product. Matching is by stem, and any
# name not listed simply falls through to the price ordering.
RECOGNISED_BRANDS = frozenset({
    "dolo", "crocin", "calpol", "pacimol", "metacin", "combiflam", "zerodol",
    "hifenac", "nise", "sumo", "azee", "zithromax", "azithral", "montek",
    "allegra", "cetzine", "okacet", "avil", "volini", "meftal", "cyclopam",
    "pan", "rantac", "zinetac", "omez", "cremaffin", "duphalac", "betnovate",
    "candid", "tenovate", "augmentin", "monocef", "taxim", "mahacef", "cifran",
    "ciplox", "metrogyl", "flagyl", "norflox", "benadryl", "ascoril",
    "grilinctus", "cheston", "alex", "sinarest", "otrivin", "nasivion",
    "refresh", "glycomet", "gluconorm", "thyronorm", "eltroxin", "ecosprin",
    "storvas", "atorva", "rosuvas", "telma", "amlong", "amlokind", "zyloric",
    "febux", "calcimax", "neurobion", "folvite", "orofer",
})


def _recognised(name: str) -> int:
    return 0 if _brand_stem(name) in RECOGNISED_BRANDS else 1


def _form_of(pack: str, brand: str) -> str:
    text = f"{pack or ''} {brand or ''}".lower()
    for form, keys in _FORM_KEYWORDS.items():
        if any(k in text for k in keys):
            return form
    return "unknown"


def _form_rank(form: str, route: str) -> int:
    """Lower is better. Exact route match wins; parenteral never outranks oral."""
    if form == route:
        return 0
    if route == "oral" and form in ("injection", "inhaler", "eye", "ear", "sublingual"):
        return 3
    if form == "oral":
        return 1
    return 2


def _base_composition(raw: str) -> str:
    """Reduce a composition string to its molecule name.

    "Amoxycillin  (500mg) " -> "amoxycillin"
    "Ambroxol (30mg/5ml) , Levosalbutamol (1mg/5ml)" -> "ambroxol"
    "Chlorpheniramine Maleate" -> "chlorpheniramine"
    """
    text = (raw or "").strip().lower()
    if not text:
        return ""
    text = text.split(",")[0]
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|%)\b", "", text)
    text = re.sub(r"[^a-z\s\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Drop a trailing salt/ester so "chlorpheniramine maleate" resolves.
    words = text.split()
    while len(words) > 1 and words[-1] in _SALT_SUFFIXES:
        words.pop()
    return " ".join(words)


def _parse_price(raw: str) -> float | None:
    try:
        value = float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _normalise_brand(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def build_index(csv_path: Path) -> tuple[dict, dict, dict]:
    medicines = json.loads(MEDICINES_PATH.read_text(encoding="utf-8"))["medicines"]
    generic_to_id = {
        (m.get("generic") or "").strip().lower(): m["id"]
        for m in medicines
        if m.get("generic")
    }

    # Pass 1: nothing to precompute; ordering uses the verified recognition list
    # plus manufacturer prominence, both of which are static.
    per_generic: dict[str, list[dict]] = defaultdict(list)
    seen_brand_keys: set[tuple[str, str]] = set()
    stats = {
        "rows": 0, "matched": 0, "discontinued": 0, "no_composition": 0,
        "unmatched": 0, "combination": 0,
    }
    unmatched_molecules: dict[str, int] = defaultdict(int)

    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            stats["rows"] += 1
            comp1 = row.get("short_composition1", "")
            comp2 = row.get("short_composition2", "")
            if _is_combination(comp1, comp2):
                stats["combination"] += 1
                continue

            molecule = _base_composition(comp1)
            if not molecule:
                stats["no_composition"] += 1
                continue

            molecule = COMPOSITION_ALIASES.get(molecule, molecule)
            med_id = generic_to_id.get(molecule)
            if not med_id:
                stats["unmatched"] += 1
                unmatched_molecules[molecule] += 1
                continue

            stats["matched"] += 1
            discontinued = str(row.get("Is_discontinued", "")).strip().upper() == "TRUE"
            if discontinued:
                stats["discontinued"] += 1

            brand = (row.get("name") or "").strip()
            if not brand:
                continue
            dedupe_key = (med_id, _normalise_brand(brand))
            if dedupe_key in seen_brand_keys:
                continue
            seen_brand_keys.add(dedupe_key)

            per_generic[med_id].append({
                "brand": brand,
                "manufacturer": (row.get("manufacturer_name") or "").strip(),
                "pack": (row.get("pack_size_label") or "").strip(),
                "price_inr": _parse_price(row.get("price(\u20b9)") or row.get("price")),
                "discontinued": discontinued,
                "composition": (row.get("short_composition1") or "").strip(),
            })

    brands: dict[str, list[dict]] = {}
    for med_id, entries in per_generic.items():
        route = ROUTE_BY_GENERIC.get(med_id, "oral")
        for entry in entries:
            entry["form"] = _form_of(entry["pack"], entry["brand"])
        # Live, correctly-routed products first; then widely-recognised brands;
        # then major manufacturers; then price, then alphabetically so repeated
        # runs produce an identical file.
        entries.sort(key=lambda e: (
            e["discontinued"],
            _form_rank(e["form"], route),
            _recognised(e["brand"]),
            _manufacturer_rank(e["manufacturer"]),
            e["price_inr"] if e["price_inr"] is not None else float("inf"),
            e["brand"].lower(),
        ))
        brands[med_id] = entries[:MAX_BRANDS_PER_GENERIC]

    return {
        "source": DATASET_URL,
        "note": (
            "Brand, manufacturer, pack size and price only. This catalogue has no "
            "indication data; clinical guidance comes from medicines.json."
        ),
        "generics_covered": len(brands),
        "brands_total": sum(len(v) for v in brands.values()),
        "brands": brands,
    }, stats, unmatched_molecules


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", help="path to indian_medicine_data.csv")
    ap.add_argument("--download", action="store_true", help="fetch the CSV from GitHub")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    if args.download or not csv_path or not csv_path.exists():
        csv_path = Path("/tmp/indian_medicine_data.csv")
        print(f"downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, csv_path)

    payload, stats, unmatched = build_index(csv_path)
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"rows read            : {stats['rows']}")
    print(f"matched to a generic : {stats['matched']}")
    print(f"  discontinued       : {stats['discontinued']}")
    print(f"no composition       : {stats['no_composition']}")
    print(f"unmatched molecule   : {stats['unmatched']}")
    print(f"combinations skipped : {stats['combination']}")
    print(f"generics covered     : {payload['generics_covered']}")
    print(f"brands written       : {payload['brands_total']} -> {out}")
    print(f"size                 : {out.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
