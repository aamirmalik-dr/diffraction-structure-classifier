"""Build and execute the tutorial notebook.

The notebook walks from a single simulated pattern through noise, orientation,
and missing reflections, then trains the classical baseline and the two CNNs on
a small pool, and ends on the confusion matrix and an accuracy-versus-noise
curve. It trains inline (small pool) so it runs end to end from a clean clone in
a couple of minutes, independent of the committed model weights.

    python scripts/build_notebook.py

Requires nbformat and nbclient (the ``dev`` extra).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "tutorial.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# Classifying crystal structure from electron diffraction\n\n"
        "This notebook goes from one simulated diffraction pattern to a scored "
        "classifier, with a picture at every step. Everything is synthetic and "
        "the labels are exact. It trains small models inline, so it runs end to "
        "end in a couple of minutes on a CPU.",
    ),
    (
        "code",
        "%matplotlib inline\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from crystalclass.sim import SimConfig, simulate, LABELS\n"
        "print('structure types:', LABELS)",
    ),
    (
        "markdown",
        "## 1. One pattern, and the physics behind it\n\n"
        "A zone-axis pattern is the set of reflections in the plane "
        "perpendicular to the beam. Which reflections survive is set by the "
        "structure factor, so the systematic absences that separate the lattices "
        "are built in. Here is a clean FCC pattern and its radial profile.",
    ),
    (
        "code",
        "p = simulate(SimConfig(structure='fcc', dose=800, orientation_spread=0.0), np.random.default_rng(0))\n"
        "fig, ax = plt.subplots(1, 2, figsize=(10, 4))\n"
        "ax[0].imshow(p.image, cmap='magma', vmax=np.percentile(p.image, 99.5))\n"
        "ax[0].set_title(f'{p.label}  zone {p.meta[\"zone_axis\"]}'); ax[0].axis('off')\n"
        "ax[1].plot(p.profile, color='#4c72b0'); ax[1].set_xlabel('radius (px)'); ax[1].set_title('radial profile')\n"
        "plt.tight_layout(); plt.show()",
    ),
    (
        "markdown",
        "## 2. The six structure types\n\n"
        "Simple cubic, BCC, FCC, diamond, rock salt, and HCP. Diamond and rock "
        "salt share the FCC ring skeleton and are the hard confusions; HCP is the "
        "easy hexagonal one.",
    ),
    (
        "code",
        "fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))\n"
        "for ax, name in zip(axes, LABELS):\n"
        "    q = simulate(SimConfig(structure=name, dose=400, orientation_spread=0.0), np.random.default_rng(3))\n"
        "    ax.imshow(q.image, cmap='magma', vmax=np.percentile(q.image, 99.5)); ax.axis('off'); ax.set_title(name)\n"
        "plt.tight_layout(); plt.show()",
    ),
    (
        "markdown",
        "## 3. What makes it hard: dose, orientation, missing reflections\n\n"
        "The same FCC crystal under falling dose, then a tilt off the zone axis, "
        "then with most reflections dropped. These are the three axes the "
        "benchmark sweeps.",
    ),
    (
        "code",
        "fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))\n"
        "settings = [dict(dose=800), dict(dose=15), dict(dose=200, orientation_spread=10.0), dict(dose=200, keep_fraction=0.35)]\n"
        "titles = ['dose 800', 'dose 15', 'tilt 10 deg', '35% of spots']\n"
        "for ax, s, t in zip(axes, settings, titles):\n"
        "    q = simulate(SimConfig(structure='fcc', **s), np.random.default_rng(1))\n"
        "    ax.imshow(q.image, cmap='magma', vmax=np.percentile(q.image, 99.5)); ax.axis('off'); ax.set_title(t)\n"
        "plt.tight_layout(); plt.show()",
    ),
    (
        "markdown",
        "## 4. Build seeded datasets\n\n" "A domain-randomised training pool and a class-balanced test set.",
    ),
    (
        "code",
        "from crystalclass.datasets import make_training_dataset, make_dataset\n"
        "from crystalclass.benchmark import _test_config\n"
        "train = make_training_dataset(1200, seed=0)\n"
        "test = make_dataset(len(LABELS) * 40, seed=99, base=_test_config({'dose': 40.0, 'orientation_spread': 3.0}))\n"
        "print('train', train.images.shape, 'test', test.images.shape)",
    ),
    (
        "markdown",
        "## 5. Classical baseline and the two CNNs\n\n"
        "The classical model reads scale- and rotation-invariant ring features "
        "into a fair-tuned random forest. The 1D CNN sees the radial profile; the "
        "2D CNN sees the whole pattern.",
    ),
    (
        "code",
        "from crystalclass.classical import train_classical\n"
        "from crystalclass.train import TrainSettings, train_model\n"
        "from crystalclass.net import predict_pattern, predict_radial\n"
        "from crystalclass.metrics import accuracy, summarize\n"
        "\n"
        "rf = train_classical(train.images, train.labels, kind='rf')\n"
        "radial, _ = train_model(TrainSettings(model='radial', pool_size=1200, epochs=12, seed=0), pool=train)\n"
        "pattern, _ = train_model(TrainSettings(model='pattern', pool_size=1200, epochs=12, seed=0), pool=train)\n"
        "\n"
        "acc = {\n"
        "    'classical_rf': accuracy(test.labels, rf.predict(test.images)),\n"
        "    'radial_cnn': accuracy(test.labels, predict_radial(radial, test.profiles)),\n"
        "    'pattern_cnn': accuracy(test.labels, predict_pattern(pattern, test.images)),\n"
        "}\n"
        "acc",
    ),
    (
        "markdown",
        "## 6. Confusion matrix\n\n" "Where the errors live, for the 2D CNN.",
    ),
    (
        "code",
        "from crystalclass.plots import plot_confusion_matrix\n"
        "import numpy as np\n"
        "rep = summarize(test.labels, predict_pattern(pattern, test.images))\n"
        "plot_confusion_matrix(np.array(rep['confusion_matrix']), 'tutorial_confusion.png', title='pattern CNN')\n"
        "plt.figure(figsize=(5.2, 4.6)); plt.imshow(plt.imread('tutorial_confusion.png')); plt.axis('off'); plt.show()",
    ),
    (
        "markdown",
        "## 7. Accuracy versus dose\n\n" "The headline axis: how each method degrades as photons vanish.",
    ),
    (
        "code",
        "doses = [8, 15, 30, 60, 120, 300]\n"
        "curves = {k: [] for k in acc}\n"
        "for d in doses:\n"
        "    ds = make_dataset(len(LABELS) * 30, seed=200 + d, base=_test_config({'dose': float(d)}))\n"
        "    curves['classical_rf'].append(accuracy(ds.labels, rf.predict(ds.images)))\n"
        "    curves['radial_cnn'].append(accuracy(ds.labels, predict_radial(radial, ds.profiles)))\n"
        "    curves['pattern_cnn'].append(accuracy(ds.labels, predict_pattern(pattern, ds.images)))\n"
        "\n"
        "plt.figure(figsize=(6, 4))\n"
        "for k, c in zip(curves, ['#4c72b0', '#dd8452', '#c44e52']):\n"
        "    plt.plot(doses, curves[k], '-o', color=c, label=k)\n"
        "plt.axhline(1/len(LABELS), ls='--', color='#999', label='chance')\n"
        "plt.xscale('log'); plt.xlabel('dose'); plt.ylabel('accuracy'); plt.ylim(0, 1.02); plt.legend(); plt.grid(alpha=0.3)\n"
        "plt.tight_layout(); plt.show()",
    ),
    (
        "markdown",
        "## 8. Takeaway\n\n"
        "The numbers here come from a small inline training run, so they are "
        "noisier than the committed benchmark in `RESULTS.md`, but the shape is "
        "the same: all methods are strong at high dose, they separate as photons "
        "and reflections vanish, and the hard confusions are diamond against FCC "
        "against rock salt. The full benchmark, with fixed seeds and larger "
        "pools, regenerates from the YAML configs in `configs/`.",
    ),
]


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    print("executing notebook...")
    client = NotebookClient(
        nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}
    )
    client.execute()
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
