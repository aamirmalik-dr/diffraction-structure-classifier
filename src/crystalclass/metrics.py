"""Classification metrics: accuracy, confusion matrix, per-class scores, and the
uncertainty needed to tell a real gap from a lucky one.

Accuracy on a finite test set is an estimate, not a fact. Two methods scored on a
few hundred patterns can differ by several points for no reason at all, so this
module also provides:

* :func:`wilson_interval`, a 95% confidence interval on a single accuracy;
* :func:`mcnemar`, the paired test for "is method A really better than method B",
  which is the right test here because every method is scored on the *same*
  patterns and their errors are correlated.

Both are used by the compare benchmark, and every headline claim in RESULTS.md is
qualified by them.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

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


def wilson_interval(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside ``[0, 1]``
    and behaves at small ``n`` and extreme proportions (hcp recall sits at 1.00,
    where the normal interval degenerates to zero width).

    Args:
        n_correct: Number of successes.
        n_total: Number of trials.
        z: Normal quantile; 1.96 gives a 95% interval.

    Returns:
        ``(low, high)``. Returns ``(0.0, 1.0)`` for ``n_total == 0``.
    """
    if n_total == 0:
        return 0.0, 1.0
    p = n_correct / n_total
    denom = 1.0 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return float(centre - half), float(centre + half)


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact paired McNemar test: is ``pred_a`` better than ``pred_b``?

    Both methods are scored on the same patterns, so their errors are correlated
    and independent confidence intervals are the wrong tool: they overlap long
    after the paired difference has become clear. McNemar conditions on the
    discordant pairs (cases where exactly one method is right) and asks whether
    they split evenly, which is exactly the question "is A better than B".

    Args:
        y_true: True class indices.
        pred_a: Predictions from method A.
        pred_b: Predictions from method B.

    Returns:
        Dict with ``n_a_only`` (A right, B wrong), ``n_b_only`` (B right, A
        wrong), ``p_value`` (two-sided exact binomial), and ``significant``
        (p < 0.05).
    """
    correct_a = np.asarray(pred_a) == np.asarray(y_true)
    correct_b = np.asarray(pred_b) == np.asarray(y_true)
    n_a_only = int(np.sum(correct_a & ~correct_b))
    n_b_only = int(np.sum(~correct_a & correct_b))
    discordant = n_a_only + n_b_only
    p = float(stats.binomtest(n_a_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "p_value": p,
        "significant": bool(p < 0.05),
    }


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Bundle accuracy (with a 95% CI), macro-F1, per-class scores, and the matrix."""
    cm = confusion_matrix(y_true, y_pred)
    n = int(np.asarray(y_true).size)
    n_correct = int(np.sum(np.asarray(y_pred) == np.asarray(y_true)))
    lo, hi = wilson_interval(n_correct, n)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci95": [lo, hi],
        "n_test": n,
        "macro_f1": macro_f1(cm),
        "per_class": per_class_scores(cm),
        "confusion_matrix": cm.tolist(),
        "labels": list(LABELS),
    }
