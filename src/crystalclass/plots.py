"""Plotting helpers for the hero figures and benchmark visualisations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crystalclass.sim import LABELS, Pattern

_METHOD_COLORS = {
    "classical_rf": "#4c72b0",
    "radial_cnn": "#dd8452",
    "pattern_cnn": "#c44e52",
    "classical_svm": "#8172b3",
}


def _color(name: str, i: int) -> str:
    palette = ["#4c72b0", "#dd8452", "#c44e52", "#55a868", "#8172b3", "#937860"]
    return _METHOD_COLORS.get(name, palette[i % len(palette)])


def plot_confusion_matrix(cm: np.ndarray, path: str | Path, title: str = "") -> None:
    """Save a normalised confusion-matrix heatmap with count annotations."""
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    norm = cm / np.where(row == 0, 1, row)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if title:
        ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = int(cm[i, j])
            if val:
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="white" if norm[i, j] > 0.5 else "#333333",
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pattern_gallery(patterns: list[Pattern], path: str | Path, title: str = "") -> None:
    """Save a labelled grid of diffraction patterns."""
    n = len(patterns)
    cols = min(n, 6)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.35 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, patterns):
        vmax = np.percentile(p.image, 99.5)
        ax.imshow(p.image, cmap="magma", vmax=vmax)
        za = "".join(str(z) for z in p.meta.get("zone_axis", ()))
        ax.set_title(f"{p.label}  [{za}]\ndose {int(p.meta.get('dose', 0))}", fontsize=8)
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sweep(sweep: dict, methods: list[str], path: str | Path, metric: str = "accuracy") -> None:
    """Save an accuracy-versus-parameter curve, one line per method."""
    rows = sweep["rows"]
    values = [r["value"] for r in rows]
    axis = sweep["axis"]
    labels = {
        "dose": "dose (counts at brightest spot)",
        "keep_fraction": "fraction of reflections kept",
        "orientation_spread": "orientation spread (degrees)",
    }
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, name in enumerate(methods):
        ys = [r[name][metric] for r in rows]
        ax.plot(values, ys, "-o", color=_color(name, i), label=name, linewidth=2, markersize=5)
    if axis == "dose":
        ax.set_xscale("log")
    ax.axhline(1.0 / len(LABELS), color="#999999", ls="--", lw=1, label="chance")
    ax.set_xlabel(labels.get(axis, axis))
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_per_class_bars(summary: dict, path: str | Path, title: str = "") -> None:
    """Save a grouped bar chart of per-class recall for one method summary."""
    per = summary["per_class"]
    recalls = [per[name]["recall"] for name in LABELS]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(range(len(LABELS)), recalls, color="#55a868")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
