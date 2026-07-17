"""Saving and loading simulated patterns as compressed ``.npz`` files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crystalclass.sim import LABEL_INDEX, Pattern


def save_pattern(path: str | Path, pattern: Pattern) -> None:
    """Save a :class:`~crystalclass.sim.Pattern` to a compressed ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=pattern.image,
        profile=pattern.profile,
        spots_xy=pattern.spots_xy,
        spots_intensity=pattern.spots_intensity,
        spots_hkl=pattern.spots_hkl,
        label=pattern.label,
        meta=json.dumps(pattern.meta),
    )


def load_pattern(path: str | Path) -> Pattern:
    """Load a :class:`~crystalclass.sim.Pattern` written by :func:`save_pattern`."""
    data = np.load(path, allow_pickle=False)
    label = str(data["label"])
    return Pattern(
        image=data["image"],
        profile=data["profile"],
        label=label,
        label_index=LABEL_INDEX[label],
        spots_xy=data["spots_xy"],
        spots_intensity=data["spots_intensity"],
        spots_hkl=data["spots_hkl"],
        meta=json.loads(str(data["meta"])),
    )
