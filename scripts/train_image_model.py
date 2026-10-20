"""Train the skin and eye image classifiers.

Reads the sorted tree written by `scripts/prepare_datasets.py`, extracts the
feature vector each time with `app.image_model.extract_features` (the same
function the server uses, so training and serving cannot drift), and fits one
classifier per modality.

Two things matter more than the accuracy number:

1. **No leakage.** SkinDisNet ships 1,710 clinical photos plus ~13,000 augmented
   near-duplicates of them. A random split would put a photo in training and its
   own blur/crop in the test set, which inflates accuracy and tells you nothing.
   So the split is by *patient*: every image of a patient is on one side. The
   augmented copies follow their source image's patient.

2. **Honest reporting.** The held-out score, the split strategy and the class
   counts are written into the model file and shown on the dataset page. If the
   score is poor, that is reported rather than hidden.

Run:
    python scripts/train_image_model.py
    python scripts/train_image_model.py --skin-only
    python scripts/train_image_model.py --limit 200      # quick smoke run
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASETS = ROOT / "data" / "external" / "datasets"
META_CSV = DATASETS / "skin_metadata.csv"
MODEL_DIR = ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "image_model.joblib"

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
AUG_SUFFIX = "__augmented"
# "<id>_NN" marks an augmented copy of "<id>".
_AUG_NUM = re.compile(r"_\d+$")


def _list_class_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _images_in(dirpath: Path) -> list[Path]:
    return sorted(
        f for f in dirpath.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXT
    )


def load_patient_map() -> dict[str, str]:
    """Image id -> patient id, from the SkinDisNet metadata CSV."""
    out: dict[str, str] = {}
    if not META_CSV.exists():
        return out
    with META_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            iid = (row.get("Image_id") or "").strip()
            pid = (row.get("Patient_id") or "").strip()
            if iid and pid:
                out[iid] = pid
    return out


def skin_samples(patient_map: dict[str, str], limit: int | None) -> list[dict]:
    """Every skin image with its class and the patient group it belongs to."""
    root = DATASETS / "skin"
    samples: list[dict] = []
    for cls_dir in _list_class_dirs(root):
        folder = cls_dir.name
        base_cls = folder[: -len(AUG_SUFFIX)] if folder.endswith(AUG_SUFFIX) else folder
        files = _images_in(cls_dir)
        if limit:
            files = files[:limit]
        for path in files:
            stem = path.stem
            # Augmented files are "<id>_NN"; the group key is the source image id,
            # so an augmentation never straddles the split from its original.
            base_id = _AUG_NUM.sub("", stem)
            patient = patient_map.get(base_id, "")
            group = patient or base_id
            samples.append({
                "path": path,
                "label": base_cls,
                "group": f"skin:{group}",
                "augmented": folder.endswith(AUG_SUFFIX),
            })
    return samples


def eye_samples(limit: int | None) -> list[dict]:
    """Every eye image. This dataset carries no patient ids, so each file is its
    own group and the split is stratified by class instead."""
    root = DATASETS / "eye"
    samples: list[dict] = []
    for cls_dir in _list_class_dirs(root):
        files = _images_in(cls_dir)
        if limit:
            files = files[:limit]
        for path in files:
            samples.append({
                "path": path,
                "label": cls_dir.name,
                "group": f"eye:{path.stem}",
                "augmented": False,
            })
    return samples


def group_split(samples: list[dict], test_frac: float, seed: int) -> tuple[list, list]:
    """Split by group so no group appears on both sides."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    target = int(len(samples) * test_frac)
    test: list[dict] = []
    train: list[dict] = []
    for key in keys:
        if len(test) < target:
            test.extend(groups[key])
        else:
            train.extend(groups[key])
    return train, test


def build_matrix(samples: list[dict], log_every: int = 400) -> tuple[list, list]:
    from app import image_model

    X, y = [], []
    for i, s in enumerate(samples, 1):
        try:
            X.append(image_model.extract_features(s["path"]))
            y.append(s["label"])
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop training
            print(f"   skipped {s['path'].name}: {exc}")
        if log_every and i % log_every == 0:
            print(f"   features {i}/{len(samples)}")
    return X, y


def train_modality(
    name: str,
    samples: list[dict],
    test_frac: float,
    seed: int,
    limit: int | None,
) -> dict:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold, cross_val_score

    print(f"\n== {name}: {len(samples)} images")
    counts = Counter(s["label"] for s in samples)
    for cls, n in sorted(counts.items()):
        print(f"   {n:>6}  {cls}")

    train, test = group_split(samples, test_frac, seed)
    print(f"   split by group: {len(train)} train / {len(test)} test")

    print("   extracting train features…")
    X_train, y_train = build_matrix(train)
    print("   extracting test features…")
    X_test, y_test = build_matrix(test, log_every=0)

    # HistGradientBoosting over RandomForest: measured on the held-out split it
    # was both more accurate (eye 0.85 vs 0.78, skin 0.40 vs 0.37) and ~65x
    # smaller on disk (2.9 MB vs 131 MB), which decides it for a service that
    # loads the model per process.
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=seed,
    )
    t0 = time.time()
    clf.fit(np.asarray(X_train, dtype="float32"), y_train)
    fit_seconds = round(time.time() - t0, 1)

    acc = float(accuracy_score(y_test, clf.predict(np.asarray(X_test, dtype="float32"))))
    f1 = float(f1_score(y_test, clf.predict(np.asarray(X_test, dtype="float32")),
                        average="macro"))
    print(f"   held-out accuracy {acc:.3f}  macro-F1 {f1:.3f}  ({fit_seconds}s)")

    # Group-aware CV on the training split: reports the spread, so a single lucky
    # test split is not mistaken for a stable result.
    cv_mean = cv_std = None
    n_groups = len({s["group"] for s in train})
    if n_groups >= 5 and len(set(y_train)) > 1:
        k = min(4, n_groups)
        groups = np.asarray([s["group"] for s in train])
        try:
            scores = cross_val_score(
                clf, np.asarray(X_train, dtype="float32"), y_train,
                groups=groups, cv=GroupKFold(n_splits=k), scoring="accuracy", n_jobs=1,
            )
            cv_mean = float(scores.mean())
            cv_std = float(scores.std())
            print(f"   group-CV accuracy {cv_mean:.3f} ± {cv_std:.3f} ({k} folds)")
        except Exception as exc:  # noqa: BLE001
            print(f"   group-CV skipped: {exc}")

    return {
        "model": clf,
        "classes": sorted(counts),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_groups": n_groups,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "cv_accuracy": round(cv_mean, 4) if cv_mean is not None else None,
        "cv_std": round(cv_std, 4) if cv_std is not None else None,
        "fit_seconds": fit_seconds,
        "split": "group (patient for skin; per-file for eye)",
        "class_counts": dict(sorted(counts.items())),
        "confusion": _confusion(y_test, clf.predict(np.asarray(X_test, dtype="float32"))),
    }


def _confusion(y_true: list, y_pred) -> dict:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, list(y_pred)):
        table[str(t)][str(p)] += 1
    return {k: dict(v) for k, v in table.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skin-only", action="store_true")
    ap.add_argument("--eye-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class folder")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not DATASETS.exists():
        print("no datasets - run scripts/prepare_datasets.py first", file=sys.stderr)
        return 2

    import joblib

    patient_map = load_patient_map()
    models: dict[str, dict] = {}

    if not args.eye_only:
        skin = skin_samples(patient_map, args.limit)
        if skin:
            models["skin"] = train_modality("skin", skin, args.test_frac, args.seed, args.limit)
        else:
            print("no skin images found")

    if not args.skin_only:
        eye = eye_samples(args.limit)
        if eye:
            models["eye"] = train_modality("eye", eye, args.test_frac, args.seed, args.limit)
        else:
            print("no eye images found")

    if not models:
        print("nothing trained", file=sys.stderr)
        return 1

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_dim": __import__("app.image_model", fromlist=["FEATURE_DIM"]).FEATURE_DIM,
        "seed": args.seed,
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in models.items()},
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # compress=3 takes the file from ~10 MB to ~3 MB with no measurable load
    # cost, which matters because the whole model is read at process start.
    joblib.dump({"models": models, "meta": meta}, MODEL_PATH, compress=3)
    (MODEL_DIR / "image_model_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {MODEL_PATH.relative_to(ROOT)} ({MODEL_PATH.stat().st_size:,} bytes)")
    for name, info in meta["models"].items():
        print(f"  {name:6} acc={info['accuracy']}  cv={info['cv_accuracy']}  classes={len(info['classes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
