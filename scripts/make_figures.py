"""Regenerate every figure in the repository from the committed benchmark JSON.

Run after the benchmarks have been executed:

    crystalclass samples
    crystalclass train --model classical
    crystalclass train --model radial
    crystalclass train --model pattern
    for c in configs/*.yaml; do crystalclass benchmark $c; done
    python scripts/make_figures.py

It reads ``results/*.json`` and writes the sweep curves, the confusion-matrix
hero, the per-class recall bars, and the ablation chart. The labelled pattern
gallery is regenerated directly from the simulator.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crystalclass.plots import (
    plot_confusion_matrix,
    plot_pattern_gallery,
    plot_per_class_bars,
    plot_sweep,
)
from crystalclass.sim import SimConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
METHODS = ["classical_rf", "radial_cnn", "pattern_cnn"]


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def gallery() -> None:
    specs = [
        ("fcc", 120.0, 0),
        ("bcc", 120.0, 1),
        ("diamond", 120.0, 2),
        ("rocksalt", 120.0, 3),
        ("hcp", 120.0, 4),
        ("sc", 120.0, 5),
    ]
    patterns = [simulate(SimConfig(structure=s, dose=d), np.random.default_rng(seed)) for s, d, seed in specs]
    plot_pattern_gallery(patterns, FIGURES / "pattern_gallery.png", title="Simulated zone-axis patterns")


def sweeps() -> None:
    for name, out in [
        ("dose_sweep.json", "dose_sweep.png"),
        ("reflection_sweep.json", "reflection_sweep.png"),
        ("orientation_sweep.json", "orientation_sweep.png"),
    ]:
        if (RESULTS / name).exists():
            plot_sweep(_load(name), METHODS, FIGURES / out, metric="accuracy")


def confusion_and_per_class() -> None:
    if not (RESULTS / "compare.json").exists():
        return
    comp = _load("compare.json")
    pat = comp["methods"]["pattern_cnn"]
    cm = np.array(pat["confusion_matrix"])
    acc = pat["accuracy"]
    plot_confusion_matrix(cm, FIGURES / "confusion_matrix.png", title=f"pattern CNN, accuracy {acc:.2f}")
    plot_per_class_bars(pat, FIGURES / "per_class_recall.png", title="pattern CNN per-class recall")


def ablation() -> None:
    if not (RESULTS / "ablation.json").exists():
        return
    data = _load("ablation.json")["variants"]
    variants = list(data)
    ref = [data[v]["test_accuracy"] for v in variants]
    low = [data[v]["low_dose_accuracy"] for v in variants]
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(x - 0.2, ref, width=0.4, label="reference dose", color="#4c72b0")
    ax.bar(x + 0.2, low, width=0.4, label="dose 20", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel("pattern CNN accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Domain-randomisation ablation")
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation.png", dpi=130)
    plt.close(fig)


def hero() -> None:
    """Compose the confusion matrix and the gallery into one hero image."""
    conf = FIGURES / "confusion_matrix.png"
    gal = FIGURES / "pattern_gallery.png"
    if not (conf.exists() and gal.exists()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, img, title in zip(axes, [gal, conf], ["", ""]):
        ax.imshow(plt.imread(img))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "hero.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    gallery()
    sweeps()
    confusion_and_per_class()
    ablation()
    hero()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
