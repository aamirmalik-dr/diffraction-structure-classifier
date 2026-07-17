"""crystalclass: crystal-structure classification from simulated electron diffraction.

Public entry points:

* :func:`crystalclass.sim.simulate` and :class:`crystalclass.sim.SimConfig` to
  generate a diffraction pattern with exact ground truth.
* :func:`crystalclass.datasets.make_dataset` /
  :func:`crystalclass.datasets.make_training_dataset` to build seeded datasets.
* :func:`crystalclass.classical.train_classical` for the fair-tuned baseline.
* :func:`crystalclass.train.train_model` for the 1D and 2D CNNs.
* :func:`crystalclass.benchmark.run_config` to run a YAML benchmark.
"""

from __future__ import annotations

from crystalclass.sim import LABELS, Pattern, SimConfig, simulate
from crystalclass.structures import STRUCTURE_NAMES, get_structure

__all__ = [
    "LABELS",
    "Pattern",
    "SimConfig",
    "simulate",
    "STRUCTURE_NAMES",
    "get_structure",
]

__version__ = "0.1.0"
