"""Tests for the uncertainty machinery and the integrity controls."""

import numpy as np

from crystalclass.datasets import make_dataset, make_scale_cue_dataset
from crystalclass.metrics import mcnemar, wilson_interval
from crystalclass.sim import LABELS


def test_wilson_interval_brackets_and_narrows():
    lo, hi = wilson_interval(70, 100)
    assert lo < 0.70 < hi
    wide = hi - lo
    lo2, hi2 = wilson_interval(7000, 10000)
    assert lo2 < 0.70 < hi2
    assert (hi2 - lo2) < wide / 5  # 100x the data -> ~10x tighter


def test_wilson_interval_handles_the_degenerate_ends():
    # hcp recall sits at exactly 1.00, where the normal approximation gives a
    # zero-width interval. Wilson must not.
    lo, hi = wilson_interval(50, 50)
    assert 0.0 < lo < 1.0
    assert hi <= 1.0
    lo, hi = wilson_interval(0, 50)
    assert lo >= 0.0
    assert 0.0 < hi < 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar_detects_a_real_difference_and_ignores_a_tie():
    y = np.zeros(200, dtype=int)
    # a is right everywhere; b is wrong on 30 of them -> strongly discordant.
    a = np.zeros(200, dtype=int)
    b = np.zeros(200, dtype=int)
    b[:30] = 1
    res = mcnemar(y, a, b)
    assert res["n_a_only"] == 30 and res["n_b_only"] == 0
    assert res["significant"] and res["p_value"] < 1e-6

    # Identical predictions: no discordant pairs, nothing to detect.
    same = mcnemar(y, a, a.copy())
    assert same["n_a_only"] == 0 and same["n_b_only"] == 0
    assert not same["significant"] and same["p_value"] == 1.0


def test_mcnemar_is_symmetric_in_p():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 300)
    a = rng.integers(0, 6, 300)
    b = rng.integers(0, 6, 300)
    assert np.isclose(mcnemar(y, a, b)["p_value"], mcnemar(y, b, a)["p_value"])


def test_scale_cue_dataset_decorrelates_the_lattice_parameter():
    """The control must sever the class -> lattice-parameter correlation.

    In the standard generator each class has one preset `a` jittered +-8%, so `a`
    is very nearly a class label. In the control every class must draw from the
    same range.
    """
    n = len(LABELS) * 40
    std = make_dataset(n, seed=0)
    dec = make_scale_cue_dataset(n, seed=0, a_range=(2.6, 6.1))

    def spread_of_class_means(ds):
        means = [
            np.mean([m["a"] for m, lab in zip(ds.meta, ds.labels) if lab == i]) for i in range(len(LABELS))
        ]
        return float(np.std(means))

    # Standard: class means of `a` are far apart (presets 2.87 ... 5.64).
    assert spread_of_class_means(std) > 1.0
    # Decorrelated: every class draws from one range, so the class means collapse.
    assert spread_of_class_means(dec) < 0.4
    # Labels stay balanced and exact.
    assert np.all(np.bincount(dec.labels, minlength=len(LABELS)) == n // len(LABELS))


def test_scale_cue_channel_is_photometric_not_geometric():
    """Changing `a` must not move spots in pixels; it may only change intensities.

    This is what makes the control clean: the ring geometry in pixels is
    scale-free by construction, so the intensity envelope ``exp(-B_dw s^2)`` is
    the channel through which absolute `a` reaches the classifier.
    """
    from crystalclass.sim import SimConfig, simulate

    kw = dict(structure="fcc", scale_frac=0.7, ring_span=4.0, orientation_spread=0.0, dose=4000)
    small = simulate(SimConfig(**kw, a=3.2), np.random.default_rng(0))
    large = simulate(SimConfig(**kw, a=5.8), np.random.default_rng(0))

    def rings(p, n_keep=12):
        """Pixel radii of the brightest spots, which are the visible ones."""
        r = np.hypot(p.spots_xy[:, 0] - 63.5, p.spots_xy[:, 1] - 63.5)
        order = np.argsort(-p.spots_intensity)[:n_keep]
        return np.sort(r[order]), p.spots_intensity[order]

    r_small, i_small = rings(small)
    r_large, i_large = rings(large)
    # Geometry: the visible rings land on identical pixel radii regardless of `a`.
    assert np.allclose(r_small, r_large, atol=1e-6)

    # Photometry: the fall-off from inner to outer ring is what `a` changes. The
    # larger cell has smaller |g| per ring, so its envelope decays more slowly.
    def falloff(p):
        r = np.hypot(p.spots_xy[:, 0] - 63.5, p.spots_xy[:, 1] - 63.5)
        outer = p.spots_intensity[r > 0.8 * r.max()]
        return float(outer.max()) if outer.size else 0.0

    assert falloff(large) > 3.0 * falloff(small), (
        "the outer rings of the larger cell must be markedly brighter; this is the cue "
        "that configs/scale_cue.yaml measures"
    )


def test_blank_training_control_uses_its_own_labels():
    """Regression guard for the leakage control's wiring.

    The 'trained on blanked patterns' control must pair images with the labels of
    the same draw. Pairing them with labels from a separately constructed dataset
    silently degrades it into a label-shuffle control, which also returns chance
    and so would look like a pass.
    """
    n = len(LABELS) * 12
    ds = make_dataset(n, seed=42, base=None)
    # Round-robin assignment is what makes the two constructors agree today; the
    # control must not depend on that coincidence.
    assert np.array_equal(ds.labels, np.array([i % len(LABELS) for i in range(n)]))
    shuffled = make_dataset(n, seed=42, class_balanced=False)
    assert not np.array_equal(shuffled.labels, ds.labels), (
        "class_balanced=False draws classes at random, so any control relying on "
        "two datasets sharing a label order would break silently"
    )
