import json
from collections import Counter

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

from predict import MODEL_PATH, normalize_text
from prepare import ROOT, prepare_splits

SEED = 42
C_VALUES = (1.0, 4.0, 16.0)
THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def evaluate(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "n": len(labels),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def train():
    splits, removed = prepare_splits()
    texts = {split: [row["text"] for row in rows] for split, rows in splits.items()}
    labels = {split: [row["label"] for row in rows] for split, rows in splits.items()}

    best_rank = (-1.0, -1.0)
    for c in C_VALUES:
        pipeline = make_pipeline(
            HashingVectorizer(
                analyzer="char", ngram_range=(2, 5), preprocessor=normalize_text,
                n_features=2**17, alternate_sign=False, norm=None,
            ),
            TfidfTransformer(sublinear_tf=True),
            LogisticRegression(C=c, solver="liblinear", max_iter=1000, random_state=SEED),
        )
        pipeline.fit(texts["train"], labels["train"])
        scores = pipeline.predict_proba(texts["val"])[:, 1]
        for threshold in THRESHOLDS:
            metrics = evaluate(labels["val"], scores >= threshold)
            rank = (metrics["f1"], metrics["precision"])
            if rank > best_rank:
                best_rank = rank
                best_pipeline, best_c, best_threshold = pipeline, c, threshold
                validation = metrics

    probabilities = best_pipeline.predict_proba(texts["test"])[:, 1]
    predictions = probabilities >= best_threshold
    report = {
        "seed": SEED,
        "C": best_c,
        "threshold": best_threshold,
        "split_counts": {split: dict(sorted(Counter(y).items())) for split, y in labels.items()},
        "removed": removed,
        "validation": validation,
        "test": evaluate(labels["test"], predictions),
    }
    joblib.dump((best_pipeline, best_threshold), MODEL_PATH, compress=3)
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    # Print a few available mistakes for inspection using the source IDs.
    for expected in (0, 1):
        errors = np.flatnonzero((np.array(labels["test"]) == expected) & (predictions != expected))
        ordered = sorted(errors, key=lambda i: abs(probabilities[i] - best_threshold), reverse=True)
        for i in ordered[:8]:
            print(splits["test"][i]["kazsandra_id"], expected, int(predictions[i]),
                  f"{probabilities[i]:.4f}")


if __name__ == "__main__":
    train()
