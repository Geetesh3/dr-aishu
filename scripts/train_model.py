"""Train Dr. Aishu's classifier from seed data and any consented real cases.

Run:  python scripts/train_model.py [--backend tfidf|transformer] [--min-confidence 0]

The script merges four sources:
  1. synthetic seed examples generated from the knowledge base
  2. an imported external dataset, if one has been brought in with
     scripts/import_symptom_dataset.py
  3. consented real cases stored by the running app
  4. user/clinician corrections, which are up-weighted
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import learning, store  # noqa: E402

SEED_FILE = ROOT / "data" / "training" / "seed.jsonl"
# Any *.jsonl under data/training that is not the seed file is treated as an
# imported external dataset. Each is weighted below a real case but above
# nothing, so it broadens coverage without outweighing observed data.
TRAIN_DIR = ROOT / "data" / "training"
IMPORT_WEIGHT = 0.8
# Narrative examples generated from the consolidated workbook by
# scripts/import_workbook_training.py. Unlike the raw symptom-list imports these
# are phrased as patient descriptions, and they were measured to *raise*
# held-out accuracy on a fifth of the seed set across five splits
# (0.8256-0.8314 -> 0.8333-0.8663), so they are included by default.
WORKBOOK_FILE = TRAIN_DIR / "workbook_2024.jsonl"


def load_seed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_imports() -> list[dict]:
    """Load every imported dataset under data/training except the seed file.

    The workbook-generated file is skipped here: it is handled separately so it
    can be included by default while the raw symptom-list imports stay opt-in.
    """
    rows: list[dict] = []
    for path in sorted(TRAIN_DIR.glob("*.jsonl")):
        if path in (SEED_FILE, WORKBOOK_FILE):
            continue
        for row in load_seed(path):
            row.setdefault("origin", path.stem)
            row.setdefault("weight", IMPORT_WEIGHT)
            rows.append(row)
    return rows


def load_workbook() -> list[dict]:
    """Narrative examples built from the consolidated workbook."""
    rows = load_seed(WORKBOOK_FILE)
    for row in rows:
        row.setdefault("origin", "workbook")
        row.setdefault("weight", IMPORT_WEIGHT)
    return rows


def collect(min_confidence: float, use_seed: bool, use_imports: bool = False) -> tuple[list[dict], dict]:
    """Gather labelled examples from every source.

    `use_imports` is off by default and should stay off unless re-measured.
    Mixing the imported dataset into training was tested against a held-out
    fifth of the seed set and *lowered* top-1 accuracy (0.601 -> 0.523; even
    restricting to rows the knowledge base already agrees with left it at
    0.572). The imported rows are symptom lists rather than narrative, and they
    pull the TF-IDF vocabulary away from the phrasing real users type. The
    dataset is therefore kept for coverage analysis -- see
    scripts/import_symptom_dataset.py -- and only mixed in on request.
    """
    seed = load_seed(SEED_FILE) if use_seed else []
    imports = load_imports() if (use_seed and use_imports) else []
    workbook = load_workbook() if use_seed else []
    real = store.training_examples(min_confidence=min_confidence)
    corrections = [r for r in real if r["origin"] == "feedback"]
    # Searches recorded without free-text consent still contribute here: events
    # carry symptom ids and labels, so the classifier learns from every search
    # without ever reading what the user typed.
    events = store.event_training_examples(min_confidence=min_confidence)
    # Consented image uploads contribute the text read from them (lab reports,
    # prescriptions), which the classifier can learn from like any other text.
    uploads = store.upload_examples(min_confidence=min_confidence)
    examples = seed + workbook + imports + real + events + uploads

    # Validate labels against the knowledge base so a typo cannot create a ghost class.
    valid = set(__import__("app.kb", fromlist=["KB"]).KB.diseases.keys())
    bad = [e for e in examples if e["label"] not in valid]
    examples = [e for e in examples if e["label"] in valid]

    summary = {
        "seed": len(seed),
        "workbook": len(workbook),
        "imported": len(imports),
        "real_cases": len(real) - len(corrections),
        "corrections": len(corrections),
        "search_events": len(events),
        "uploads": len(uploads),
        "invalid_labels_dropped": len(bad),
        "total": len(examples),
    }
    return examples, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Dr. Aishu's diagnosis classifier.")
    ap.add_argument("--backend", choices=["tfidf", "transformer", "auto"], default="auto")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="skip stored cases whose top-diagnosis confidence is below this")
    ap.add_argument("--no-seed", action="store_true", help="train only on real stored cases")
    ap.add_argument("--use-imports", action="store_true",
                    help="also mix in imported external datasets (measured to lower accuracy; "
                         "see collect())")
    ap.add_argument("--epochs", type=int, default=4, help="transformer only")
    ap.add_argument("--base-model", default="distilbert-base-uncased", help="transformer only")
    args = ap.parse_args()

    examples, summary = collect(
        args.min_confidence, use_seed=not args.no_seed, use_imports=args.use_imports,
    )
    print("data sources:", json.dumps(summary, indent=2))

    if not examples:
        print("No training data available. Run scripts/seed_training_data.py first.")
        return 1

    backend = args.backend
    if backend == "auto":
        backend = "tfidf"

    try:
        if backend == "transformer":
            report = learning.train_transformer(examples, base_model=args.base_model, epochs=args.epochs)
        else:
            report = learning.train_tfidf(examples)
    except ValueError as exc:
        print(f"training failed: {exc}")
        return 1

    print(json.dumps({
        "backend": report.backend,
        "examples": report.n_examples,
        "labels": report.n_labels,
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "cv_accuracy": report.cv_accuracy,
        "cv_std": report.cv_std,
        "seconds": report.seconds,
        "notes": report.notes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
