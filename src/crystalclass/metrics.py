"""Classification metrics: accuracy, confusion matrix, per-class scores."""

from __future__ import annotations

import numpy as np

from crystalclass.sim import LABELS


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = len(LABELS)) -> np.ndarray:
    """Return the confusion matrix ``C[i, j]`` = count of true ``i`` predicted ``j``."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall fraction of correct predictions."""
    y_true = np.asarray(y_true)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.asarray(y_pred) == y_true))


def per_class_scores(cm: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, and F1 from a confusion matrix.

    Args:
        cm: Confusion matrix with true classes on rows.

    Returns:
        Mapping from class name to a dict with ``precision``, ``recall``,
        ``f1``, and ``support``.
    """
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(LABELS):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
        }
    return out


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean of the per-class F1 scores."""
    scores = per_class_scores(cm)
    return float(np.mean([s["f1"] for s in scores.values()]))


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Bundle accuracy, macro-F1, per-class scores, and the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(cm),
        "per_class": per_class_scores(cm),
        "confusion_matrix": cm.tolist(),
        "labels": list(LABELS),
    }
