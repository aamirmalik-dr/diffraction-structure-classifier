"""Tests that pin down the physical claims the README makes.

These exist because each one was asserted in prose before it was ever checked,
and one of them turned out to be false.
"""

import numpy as np

from crystalclass.sim import SimConfig, simulate
from crystalclass.structures import get_structure, zone_axis_reflections


def _rings(name, zone=(0, 0, 1)):
    """Normalised ring radii, multiplicities, and mean intensities down `zone`."""
    s = get_structure(name)
    r = zone_axis_reflections(s, zone, s.a, g_max=12.0 / s.a)
    g = r.g_len / r.g_len.min()
    inten = r.intensity / r.intensity.max()
    out = {}
    for gg, ii in zip(g, inten):
        out.setdefault(round(float(gg), 4), []).append(float(ii))
    keys = sorted(out)[:6]
    return keys, [len(out[k]) for k in keys], [float(np.mean(out[k])) for k in keys]


def test_rocksalt_and_sc_are_geometrically_identical_down_001():
    """Rock salt and simple cubic are NOT distinguishable by geometry down [001].

    Down [001] every reflection has l = 0, so rock salt's all-odd family is
    impossible and only all-even reflections survive: the strong/weak alternation
    that defines rock salt is invisible on this zone axis, and what is left is a
    square net identical in form to simple cubic's. The README once explained the
    2D model's rock-salt win by an angular difference; there is none.
    """
    r_sc, m_sc, i_sc = _rings("sc")
    r_rs, m_rs, i_rs = _rings("rocksalt")
    assert np.allclose(r_sc, r_rs, atol=1e-6), "ring radii must coincide"
    assert m_sc == m_rs, "ring multiplicities must coincide"
    # Only the intensity fall-off differs, and it does differ.
    assert not np.allclose(i_sc, i_rs, atol=0.02)


def test_rocksalt_has_no_odd_reflections_down_001():
    s = get_structure("rocksalt")
    r = zone_axis_reflections(s, (0, 0, 1), s.a, g_max=8.0 / s.a)
    assert r.hkl.size
    assert np.all(r.hkl % 2 == 0), "all surviving [001] reflections must be all-even"


def test_rocksalt_odd_reflections_appear_off_001():
    """Off [001] the all-odd family is allowed, so the alternation is visible."""
    s = get_structure("rocksalt")
    r = zone_axis_reflections(s, (0, 1, 1), s.a, g_max=8.0 / s.a)
    all_odd = np.all(r.hkl % 2 == 1, axis=1)
    assert all_odd.any(), "[011] must show all-odd reflections"
    assert r.intensity[all_odd].max() < 0.25 * r.intensity.max(), "all-odd must be weak"


def test_readout_noise_is_independent_of_dose():
    """A detector's readout floor is a camera property, not an exposure property.

    Guards against the earlier model, which scaled readout sigma with the dose and
    so had no readout floor at all where it matters most (low dose).
    """
    cfg = dict(structure="fcc", blank_spots=True, background_level=0.0, diffuse_level=0.0)
    # With no spots, no background and dose ~ 0, all that remains is readout.
    lo = simulate(SimConfig(**cfg, dose=1e-6, readout_counts=1.0), np.random.default_rng(0))
    hi = simulate(SimConfig(**cfg, dose=1e-6, readout_counts=4.0), np.random.default_rng(0))
    # Clipping at zero halves the Gaussian, so compare spreads, not exact sigmas.
    assert hi.image.std() > 3.0 * lo.image.std()

    # And the floor must not grow with dose: measure a far-corner region where the
    # pattern contributes nothing at two very different doses.
    a = simulate(SimConfig(structure="fcc", dose=10.0), np.random.default_rng(1))
    b = simulate(SimConfig(structure="fcc", dose=1000.0), np.random.default_rng(1))
    scaled_ratio = b.image[:8, :8].std() / (a.image[:8, :8].std() + 1e-9)
    assert scaled_ratio < 100.0  # would be ~100x if readout scaled with dose


def test_orientation_spread_is_the_half_normal_scale_not_the_std():
    """`orientation_spread` is the sigma of the normal BEFORE the absolute value."""
    spread = 8.0
    tilts = np.array(
        [
            simulate(SimConfig(structure="fcc", orientation_spread=spread), np.random.default_rng(i)).meta[
                "tilt_deg"
            ]
            for i in range(400)
        ]
    )
    assert np.all(tilts >= 0.0)
    # Half-normal: mean = sigma*sqrt(2/pi) ~ 0.798 sigma, std = sigma*sqrt(1-2/pi) ~ 0.603 sigma.
    assert np.isclose(tilts.mean(), spread * np.sqrt(2 / np.pi), rtol=0.15)
    assert np.isclose(tilts.std(), spread * np.sqrt(1 - 2 / np.pi), rtol=0.15)


def test_keep_fraction_dropout_is_intensity_independent():
    """Dropout is random spot loss, not a detection threshold.

    A detection threshold would raise the surviving set's mean intensity by
    removing the weakest spots first. This must not do that.
    """
    kept_means = []
    for seed in range(12):
        full = simulate(
            SimConfig(structure="sc", keep_fraction=1.0, orientation_spread=0.0),
            np.random.default_rng(seed),
        )
        part = simulate(
            SimConfig(structure="sc", keep_fraction=0.5, orientation_spread=0.0),
            np.random.default_rng(seed),
        )
        kept_means.append(part.spots_intensity.mean() - full.spots_intensity.mean())
    # No systematic enrichment in strong spots.
    assert abs(float(np.mean(kept_means))) < 0.05


def test_first_ring_radius_in_pixels_carries_no_class_information():
    """The camera-length construction makes pattern scale class-independent.

    r1_px = scale_frac * (size/2) / ring_span, independent of g1 and hence of the
    structure, which is why absolute pixel scale cannot leak the class.
    """
    from crystalclass.sim import LABELS

    radii = []
    for name in LABELS:
        p = simulate(
            SimConfig(structure=name, scale_frac=0.7, ring_span=4.0, orientation_spread=0.0, dose=800),
            np.random.default_rng(0),
        )
        r = np.hypot(p.spots_xy[:, 0] - 63.5, p.spots_xy[:, 1] - 63.5)
        radii.append(r.min())
    assert np.std(radii) < 0.5, "inner-ring pixel radius must be the same for every class"
