"""Dataset access for the learning-curve protocol: full train pool + a fixed test set.

Reuses the library's `_SPECS` and stratified/k-shot samplers so splits stay identical to the
CLI's. The learning-curve protocol is: draw ONE fixed test set (keyed on `test_seed` only),
keep the *entire* train pool, and resample k examples/class from it across many training seeds.
Test is never subsampled and never depends on train size — the comparability property the
low-N study needs (see docs/low-n-plan.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hypothesis_vectorizer.train.data import (
    _NEWSGROUP_GLOSS,
    _SPECS,
    per_class_indices,
    stratified_indices,
)


@dataclass
class RawData:
    name: str
    task: str
    class_names: list[str]
    class_descriptions: list[str]
    train_texts: list[str]
    y_train: np.ndarray  # full train pool
    test_texts: list[str]
    y_test: np.ndarray  # fixed test set
    test_seed: int

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def load_raw(dataset: str, *, test_size: int = 2000, test_seed: int = 7) -> RawData:
    from datasets import load_dataset

    spec = _SPECS[dataset]
    ds = load_dataset(spec["hf"], revision=spec.get("revision"))
    train, test = ds["train"], ds[spec["test_split"]]
    tf, lf = spec["text_field"], spec["label_field"]
    train_texts, y_train = list(train[tf]), np.asarray(train[lf], dtype=np.int64)
    test_texts, y_test = list(test[tf]), np.asarray(test[lf], dtype=np.int64)

    classes = spec["classes"]
    if classes is None:
        classes = [name for _, name in sorted(set(zip(train["label"], train["label_text"])))]
    descriptions = spec["descriptions"]
    if descriptions is None:
        descriptions = [f"{c}: {_NEWSGROUP_GLOSS.get(c, c)}" for c in classes]

    te = stratified_indices(y_test, min(test_size, len(y_test)), np.random.default_rng(test_seed))
    return RawData(
        name=dataset,
        task=spec["task"],
        class_names=classes,
        class_descriptions=descriptions,
        train_texts=train_texts,
        y_train=y_train,
        test_texts=[test_texts[i] for i in te],
        y_test=y_test[te],
        test_seed=test_seed,
    )


def kshot_indices(y_train: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Exactly k train indices per class, resampled per seed (the low-N training draw)."""
    return per_class_indices(y_train, k, np.random.default_rng(1000 + seed))


def subsample_indices(y_train: np.ndarray, k: int | str, seed: int) -> np.ndarray:
    """k examples/class, or all of the train pool when k == 'all'."""
    if k == "all":
        idx = np.arange(len(y_train))
        np.random.default_rng(1000 + seed).shuffle(idx)
        return idx
    return kshot_indices(y_train, int(k), seed)


AVAILABLE = list(_SPECS.keys())
