"""Build fixed, seeded datasets of simulated patterns.

A dataset is a stack of images, a stack of radial profiles, and the exact
integer labels. Class balance and seeding are explicit so every table in the
repository regenerates bit for bit.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from crystalclass.sim import LABELS, SimConfig, simulate


class Dataset:
    """A stack of patterns with images, radial profiles, and labels."""

    def __init__(self, images: np.ndarray, profiles: np.ndarray, labels: np.ndarray, meta: list[dict]):
        self.images = images.astype(np.float32)
        self.profiles = profiles.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.meta = meta

    def __len__(self) -> int:
        return len(self.labels)


def make_dataset(
    n: int,
    seed: int,
    base: SimConfig | None = None,
    class_balanced: bool = True,
) -> Dataset:
    """Generate ``n`` patterns with per-sample domain randomisation.

    Args:
        n: Number of patterns.
        seed: Master seed; each pattern draws from an independent child stream.
        base: Template config whose non-None fields are held fixed; the rest are
            randomised per pattern. Defaults to :class:`SimConfig` defaults.
        class_balanced: If True, assign labels round-robin so classes are
            evenly represented; otherwise draw the structure at random.

    Returns:
        A :class:`Dataset`.
    """
    base = base or SimConfig()
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n)

    images, profiles, labels, meta = [], [], [], []
    for i in range(n):
        cfg = base
        if class_balanced and base.structure is None:
            cfg = replace(base, structure=LABELS[i % len(LABELS)])
        pattern = simulate(cfg, np.random.default_rng(int(seeds[i])))
        images.append(pattern.image)
        profiles.append(pattern.profile)
        labels.append(pattern.label_index)
        meta.append(pattern.meta)
    return Dataset(
        images=np.stack(images),
        profiles=np.stack(profiles),
        labels=np.array(labels),
        meta=meta,
    )


def make_training_dataset(
    n: int,
    seed: int,
    size: int = 128,
    dose_log_range: tuple[float, float] = (8.0, 400.0),
    orient_max: float = 6.0,
    keep_min: float = 0.75,
    randomize_scale: bool = True,
    randomize_background: bool = True,
) -> Dataset:
    """Build a training pool with full per-sample domain randomisation.

    Each pattern draws an independent structure, dose (log-uniform),
    orientation spread, and missing-reflection fraction. The camera-length
    scale and the background are randomised unless the corresponding flag is
    cleared, which is how the training-side ablations remove a candidate cue.

    Args:
        n: Number of patterns.
        seed: Master seed.
        size: Image side length.
        dose_log_range: ``(low, high)`` dose in counts, sampled log-uniformly.
        orient_max: Maximum orientation spread in degrees.
        keep_min: Minimum kept-reflection fraction.
        randomize_scale: If False, the camera-length scale is held fixed.
        randomize_background: If False, the background is held fixed.

    Returns:
        A :class:`Dataset`.
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.log10(dose_log_range[0]), np.log10(dose_log_range[1])
    images, profiles, labels, meta = [], [], [], []
    for i in range(n):
        cfg = SimConfig(
            structure=LABELS[i % len(LABELS)],
            size=size,
            dose=float(10 ** rng.uniform(lo, hi)),
            orientation_spread=float(rng.uniform(0.0, orient_max)),
            keep_fraction=float(rng.uniform(keep_min, 1.0)),
            scale_frac=None if randomize_scale else 0.68,
            background_level=None if randomize_background else 0.03,
            diffuse_level=None if randomize_background else 0.06,
        )
        pattern = simulate(cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1))))
        images.append(pattern.image)
        profiles.append(pattern.profile)
        labels.append(pattern.label_index)
        meta.append(pattern.meta)
    return Dataset(
        images=np.stack(images),
        profiles=np.stack(profiles),
        labels=np.array(labels),
        meta=meta,
    )
