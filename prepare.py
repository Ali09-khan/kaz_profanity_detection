import csv
import io
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import HashingVectorizer

from predict import normalize_text

ROOT = Path(__file__).resolve().parent
SOURCE = (
    "https://raw.githubusercontent.com/IS2AI/KazSAnDRA/"
    "d769cf7f7bd62b59f88915508d4bc062310b54d6/dataset/"
)
ARCHIVES = ("06_sc_train_ib", "09_sc_valid", "10_sc_test")
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
SPELLING_FOLD = str.maketrans("әəғқңөұүһій", "аагкноуухии")


def load_reviews(review_ids):
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    reviews = {}
    for name in ARCHIVES:
        path = cache / f"{name}.zip"
        if not path.exists():
            with urlopen(SOURCE + path.name, timeout=60) as response:
                contents = response.read()
            path.write_bytes(contents)
        with ZipFile(path) as archive, archive.open(f"{name}.csv") as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")):
                if row["custom_id"] in review_ids:
                    reviews[row["custom_id"]] = row["text"]
    if review_ids.difference(reviews):
        raise ValueError("The KazSAnDRA archives are missing expected review IDs.")
    return reviews


def duplicate_key(text):
    return "".join(char for char in normalize_text(text) if char.isalnum())


def near_duplicate_groups(keys):
    folded = [re.sub(r"(.)\1+", r"\1", key.translate(SPELLING_FOLD)) for key in keys]
    vectorizer = HashingVectorizer(
        analyzer="char", ngram_range=(2, 3), binary=True, alternate_sign=False,
        n_features=2**18, dtype=np.float32,
    )
    features = vectorizer.transform(folded)
    graph = (features @ features.T) >= 0.84

    # Short strings can lose several n-grams with a single mistyped letter.
    # Link identical folded strings and short strings sharing a one-deletion form.
    first_seen = {}
    edges = []
    for index, key in enumerate(folded):
        signatures = {key}
        if 5 <= len(key) <= 24:
            signatures.update(key[:n] + key[n + 1:] for n in range(len(key)))
        for signature in sorted(signatures):
            previous = first_seen.setdefault(signature, index)
            if previous != index:
                edges.append((previous, index))

    if edges:
        sources, targets = zip(*edges)
        graph = graph + coo_matrix(
            (np.ones(len(edges), dtype=bool), (sources, targets)), shape=graph.shape
        )
    return connected_components(graph, directed=False)[1]


def prepare_splits():
    with (ROOT / "labels.csv").open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    labeled = [row for row in manifest if row["label"]]
    reviews = load_reviews({row["kazsandra_id"] for row in labeled})
    by_key = defaultdict(list)
    removed = Counter(row["review"] for row in manifest if not row["label"])
    for row in labeled:
        row["text"] = reviews[row["kazsandra_id"]]
        row["label"] = int(row["label"])
        key = duplicate_key(row["text"])
        if key:
            by_key[key].append(row)
        else:
            removed["empty_after_normalizing"] += 1

    unique = {}
    for key, items in by_key.items():
        if len({item["label"] for item in items}) > 1:
            removed["conflicting_duplicates"] += len(items)
        else:
            unique[key] = max(items, key=lambda item: SPLIT_PRIORITY[item["source_split"]])
            removed["exact_duplicates"] += len(items) - 1

    groups = defaultdict(list)
    for row, group in zip(unique.values(), near_duplicate_groups(list(unique)), strict=True):
        groups[group].append(row)

    splits = {name: [] for name in SPLIT_PRIORITY}
    for items in groups.values():
        split = max((item["source_split"] for item in items), key=SPLIT_PRIORITY.get)
        for row in items:
            if row["source_split"] == split:
                splits[split].append(row)
            else:
                removed["near_duplicate_in_holdout"] += 1

    return splits, dict(removed)


if __name__ == "__main__":
    splits, removed = prepare_splits()
    for split, rows in splits.items():
        print(split, dict(sorted(Counter(row["label"] for row in rows).items())))
    print("Removed", removed)
