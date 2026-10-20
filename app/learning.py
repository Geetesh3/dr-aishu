"""Dr. Aishu's learning module.

Three interchangeable backends, chosen automatically by what is installed:

* `transformer` - a HuggingFace sequence classifier (free, open source). Best
  accuracy. Enable with `pip install transformers torch`.
* `tfidf` - scikit-learn TF-IDF + linear classifier. Always available, trains in
  under a second on modest data, and is the default.
* `none` - no model trained yet; the app falls back to pure knowledge-base rules.

Every backend exposes the same interface, so the analyzer does not care which one
is active. Models are saved to `data/models/` and hot-reloaded after training.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
TFIDF_PATH = MODEL_DIR / "tfidf.joblib"
TRANSFORMER_DIR = MODEL_DIR / "transformer"
META_PATH = MODEL_DIR / "meta.json"

_lock = threading.Lock()
_state: dict[str, Any] = {"backend": None, "model": None, "meta": {}}


@dataclass
class TrainReport:
    backend: str
    n_examples: int
    n_labels: int
    accuracy: float | None = None
    macro_f1: float | None = None
    cv_accuracy: float | None = None
    cv_std: float | None = None
    seconds: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Availability probing
# ---------------------------------------------------------------------------
def _transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def available_backends() -> list[str]:
    out = []
    if _sklearn_available():
        out.append("tfidf")
    if _transformers_available():
        out.append("transformer")
    return out


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------
def build_document(text: str, symptom_ids: list[str] | None = None) -> str:
    """Combine free text with canonical symptom tokens.

    Symptom ids are appended as `sym_<id>` tokens so a case that arrived as an
    image with little OCR text still carries signal.
    """
    parts = [text or ""]
    for sid in symptom_ids or []:
        parts.append(f"sym_{sid}")
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# TF-IDF backend
# ---------------------------------------------------------------------------
def train_tfidf(
    examples: list[dict[str, Any]],
    min_examples_per_label: int = 2,
    test_size: float = 0.2,
) -> TrainReport:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    import joblib

    started = time.time()
    docs = [build_document(e.get("text", ""), e.get("symptoms")) for e in examples]
    labels = [e["label"] for e in examples]
    weights = [float(e.get("weight", 1.0)) for e in examples]

    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    keep = {lab for lab, n in counts.items() if n >= min_examples_per_label}
    filtered = [(d, l, w) for d, l, w in zip(docs, labels, weights) if l in keep]
    notes: list[str] = []
    dropped = len(docs) - len(filtered)
    if dropped:
        notes.append(f"dropped {dropped} example(s) from labels with fewer than "
                     f"{min_examples_per_label} samples")

    if len(filtered) < 4 or len(keep) < 2:
        raise ValueError(
            f"not enough training data: {len(filtered)} usable examples across {len(keep)} labels"
        )

    docs, labels, weights = zip(*filtered)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            C=4.0,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=None,
        )),
    ])

    accuracy = macro_f1 = cv_accuracy = cv_std = None
    # Stratify only when every label has enough members for a split.
    can_split = all(counts[l] >= 3 for l in keep) and len(keep) > 1
    if can_split:
        try:
            x_tr, x_te, y_tr, y_te, w_tr, _ = train_test_split(
                docs, labels, weights, test_size=test_size, random_state=42, stratify=labels,
            )
            pipeline.fit(x_tr, y_tr, clf__sample_weight=w_tr)
            pred = pipeline.predict(x_te)
            accuracy = float(accuracy_score(y_te, pred))
            macro_f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))

            # A single hold-out split is a noisy estimate. Cross-validation gives
            # the honest expected accuracy and its spread across folds.
            folds = min(5, min(counts[l] for l in keep))
            if folds >= 2:
                from sklearn.model_selection import cross_val_score

                scores = cross_val_score(
                    pipeline, docs, labels, cv=folds,
                    scoring="accuracy", n_jobs=None,
                    params={"clf__sample_weight": weights},
                )
                cv_accuracy = float(scores.mean())
                cv_std = float(scores.std())
        except (ValueError, TypeError) as exc:
            notes.append(f"held-out evaluation unavailable ({exc}); trained on all data")
            pipeline.fit(docs, labels, clf__sample_weight=weights)
    else:
        notes.append("dataset too small for a held-out split; trained on all data")
        pipeline.fit(docs, labels, clf__sample_weight=weights)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, TFIDF_PATH)

    report = TrainReport(
        backend="tfidf",
        n_examples=len(docs),
        n_labels=len(keep),
        accuracy=accuracy,
        macro_f1=macro_f1,
        cv_accuracy=cv_accuracy,
        cv_std=cv_std,
        seconds=round(time.time() - started, 2),
        detail={"labels": sorted(keep)},
        notes=notes,
    )
    _write_meta(report)
    _activate("tfidf", pipeline, report)
    return report


# ---------------------------------------------------------------------------
# Transformer backend
# ---------------------------------------------------------------------------
def train_transformer(
    examples: list[dict[str, Any]],
    base_model: str = "distilbert-base-uncased",
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    test_size: float = 0.2,
) -> TrainReport:
    """Fine-tune a HuggingFace classifier. Intended for a Colab GPU runtime."""
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    started = time.time()
    docs = [build_document(e.get("text", ""), e.get("symptoms")) for e in examples]
    labels = [e["label"] for e in examples]
    label_list = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    if len(label_list) < 2 or len(docs) < 8:
        raise ValueError(f"not enough data for transformer training: {len(docs)} examples, "
                         f"{len(label_list)} labels")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    class _DS(Dataset):
        def __init__(self, texts, ys):
            self.enc = tokenizer(list(texts), truncation=True, padding=True, max_length=256)
            self.ys = ys

        def __len__(self):
            return len(self.ys)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.ys[i])
            return item

    split = max(1, int(len(docs) * (1 - test_size)))
    train_ds = _DS(docs[:split], [label2id[l] for l in labels[:split]])
    eval_ds = _DS(docs[split:], [label2id[l] for l in labels[split:]]) if split < len(docs) else None

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(label_list), id2label=id2label, label2id=label2id,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = TRANSFORMER_DIR
    args = TrainingArguments(
        output_dir=str(out_dir / "_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()

    metrics = trainer.evaluate() if eval_ds is not None else {}
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    report = TrainReport(
        backend="transformer",
        n_examples=len(docs),
        n_labels=len(label_list),
        accuracy=metrics.get("eval_accuracy"),
        macro_f1=None,
        seconds=round(time.time() - started, 2),
        detail={"base_model": base_model, "labels": label_list,
                "device": "cuda" if torch.cuda.is_available() else "cpu"},
        notes=[] if eval_ds is not None else ["no held-out set; trained on all data"],
    )
    _write_meta(report)
    _activate("transformer", {"dir": str(out_dir), "labels": label_list}, report)
    return report


# ---------------------------------------------------------------------------
# Loading / prediction
# ---------------------------------------------------------------------------
def _write_meta(report: TrainReport) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps({
        "backend": report.backend,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_examples": report.n_examples,
        "n_labels": report.n_labels,
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "cv_accuracy": report.cv_accuracy,
        "cv_std": report.cv_std,
        "seconds": report.seconds,
        "notes": report.notes,
        "detail": report.detail,
    }, indent=2))


def _activate(backend: str, model: Any, report: TrainReport | None = None) -> None:
    with _lock:
        _state["backend"] = backend
        _state["model"] = model
        _state["meta"] = {
            "backend": backend,
            "n_examples": report.n_examples if report else None,
            "n_labels": report.n_labels if report else None,
            "accuracy": report.accuracy if report else None,
        }


def load_model() -> str | None:
    """Load a previously trained model from disk. Returns the active backend."""
    if TFIDF_PATH.exists():
        try:
            import joblib
            model = joblib.load(TFIDF_PATH)
            meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
            _activate("tfidf", model)
            _state["meta"] = meta
            return "tfidf"
        except Exception:
            pass

    if TRANSFORMER_DIR.exists() and _transformers_available():
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
            model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_DIR))
            model.eval()
            meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
            _activate("transformer", {"model": model, "tokenizer": tokenizer,
                                      "labels": list(model.config.id2label.values())})
            _state["meta"] = meta
            return "transformer"
        except Exception:
            pass

    return None


def is_ready() -> bool:
    return _state["backend"] is not None


def status() -> dict[str, Any]:
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
        except Exception:
            meta = {}
    return {
        "backend": _state["backend"],
        "ready": is_ready(),
        "available_backends": available_backends(),
        "meta": meta,
        "model_files": {
            "tfidf": TFIDF_PATH.exists(),
            "transformer": TRANSFORMER_DIR.exists(),
        },
    }


def predict(
    text: str,
    symptom_ids: list[str] | None = None,
    top_k: int = 6,
    min_probability: float = 0.08,
) -> list[dict[str, Any]]:
    """Return ranked {label, probability} predictions, or [] when no model is ready.

    `min_probability` discards near-uniform output. With 42 classes the uniform
    prior is about 0.024, so a floor of 0.08 keeps genuine matches (typically
    0.10 and above) while rejecting the flat distribution produced for text that
    carries no medical signal.
    """
    if not is_ready():
        return []

    doc = build_document(text, symptom_ids)
    if not doc.strip():
        return []

    backend = _state["backend"]
    try:
        if backend == "tfidf":
            model = _state["model"]
            probs = model.predict_proba([doc])[0]
            classes = list(model.classes_)
        elif backend == "transformer":
            import torch
            bundle = _state["model"]
            tok, mdl = bundle["tokenizer"], bundle["model"]
            enc = tok(doc, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            classes = [mdl.config.id2label[i] for i in range(len(probs))]
        else:
            return []
    except Exception:
        return []

    pairs = sorted(zip(classes, probs), key=lambda p: -p[1])[:top_k]
    return [
        {"label": label, "probability": round(float(p), 4)}
        for label, p in pairs
        if float(p) >= min_probability
    ]
