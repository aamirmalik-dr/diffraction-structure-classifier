import numpy as np

from crystalclass.sim import LABELS, SimConfig, simulate


def test_shapes_and_label():
    p = simulate(SimConfig(structure="fcc", dose=200), np.random.default_rng(0))
    assert p.image.shape == (128, 128)
    assert p.profile.shape == (64,)
    assert p.label == "fcc"
    assert p.label_index == LABELS.index("fcc")


def test_determinism():
    cfg = SimConfig(structure="bcc", dose=120)
    a = simulate(cfg, np.random.default_rng(3))
    b = simulate(cfg, np.random.default_rng(3))
    assert np.array_equal(a.image, b.image)


def test_size_constant_across_classes():
    sizes = {simulate(SimConfig(structure=n), np.random.default_rng(1)).image.shape for n in LABELS}
    assert sizes == {(128, 128)}


def test_blank_spots_removes_signal():
    seed = 5
    full = simulate(SimConfig(structure="fcc", dose=300), np.random.default_rng(seed))
    blank = simulate(SimConfig(structure="fcc", dose=300, blank_spots=True), np.random.default_rng(seed))
    # blanked image has far less spot contrast than the full one
    assert full.image.std() > blank.image.std()


def test_higher_dose_is_cleaner():
    low = simulate(SimConfig(structure="fcc", dose=8), np.random.default_rng(2))
    high = simulate(SimConfig(structure="fcc", dose=2000), np.random.default_rng(2))
    # relative noise (std of background region) falls with dose
    corner_low = low.image[:16, :16].std() / (low.image.max() + 1e-9)
    corner_high = high.image[:16, :16].std() / (high.image.max() + 1e-9)
    assert corner_high < corner_low


def test_missing_reflections_reduce_spots():
    full = simulate(SimConfig(structure="sc", keep_fraction=1.0), np.random.default_rng(7))
    sparse = simulate(SimConfig(structure="sc", keep_fraction=0.3), np.random.default_rng(7))
    assert sparse.meta["n_spots"] <= full.meta["n_spots"]
