"""Metrics for the paper: accuracy, macro/weighted-F1, per-class P/R/F1, calibration (ECE).

All metrics take integer label arrays plus an optional (n, n_classes) probability matrix.
`compute_metrics` returns a flat dict suitable for a results CSV row; `per_class_table`
returns the rows for a per-class table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE: bin by predicted confidence, |accuracy - confidence| weighted by bin size."""
    if proba is None:
        return float("nan")
    proba = np.asarray(proba, dtype=float)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    *,
    n_classes: int | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if proba is not None:
        labels = list(range(n_classes)) if n_classes else None
        try:
            out["log_loss"] = float(log_loss(y_true, proba, labels=labels))
        except Exception:
            out["log_loss"] = float("nan")
        out["ece"] = expected_calibration_error(y_true, proba)
    return out


def per_class_table(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> list[dict[str, Any]]:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {"class": class_names[i], "precision": float(p[i]), "recall": float(r[i]),
         "f1": float(f[i]), "support": int(s[i])}
        for i in labels
    ]


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and (1-alpha) percentile bootstrap CI over a set of per-seed scores."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)
