import numpy as np

from crystalclass.structures import (
    STRUCTURE_NAMES,
    get_structure,
    reciprocal_matrix,
    structure_factors,
    zone_axis_reflections,
)


def _f2(name, hkl):
    s = get_structure(name)
    return structure_factors(s, np.array([hkl]), s.lattice_matrix())[0]


def test_all_structures_registered():
    assert set(STRUCTURE_NAMES) == {"sc", "bcc", "fcc", "diamond", "rocksalt", "hcp"}


def test_reciprocal_metric_cubic():
    s = get_structure("sc")
    b = reciprocal_matrix(s.lattice_matrix())
    g = b @ np.array([1, 1, 1])
    assert np.isclose(np.linalg.norm(g), np.sqrt(3) / s.a, rtol=1e-6)


def test_bcc_absences():
    assert _f2("bcc", (1, 1, 0)) > 1e-3  # h+k+l even allowed
    assert _f2("bcc", (1, 0, 0)) < 1e-3  # h+k+l odd forbidden
    assert _f2("bcc", (1, 1, 1)) < 1e-3


def test_fcc_absences():
    assert _f2("fcc", (1, 1, 1)) > 1e-3  # all odd allowed
    assert _f2("fcc", (2, 0, 0)) > 1e-3  # all even allowed
    assert _f2("fcc", (1, 1, 0)) < 1e-3  # mixed parity forbidden
    assert _f2("fcc", (2, 1, 0)) < 1e-3


def test_diamond_extra_absences():
    assert _f2("diamond", (1, 1, 1)) > 1e-3
    assert _f2("diamond", (2, 2, 0)) > 1e-3
    assert _f2("diamond", (2, 0, 0)) < 1e-3  # diamond-only extinction
    assert _f2("diamond", (2, 2, 2)) < 1e-3


def test_rocksalt_weak_and_strong():
    weak = _f2("rocksalt", (1, 1, 1))  # difference of scattering factors
    strong = _f2("rocksalt", (2, 0, 0))  # sum of scattering factors
    assert strong > weak > 1e-3


def test_zone_axis_excludes_direct_beam():
    s = get_structure("fcc")
    refl = zone_axis_reflections(s, (0, 0, 1), s.a, g_max=2.0)
    assert refl.hkl.shape[0] > 0
    assert not np.any(np.all(refl.hkl == 0, axis=1))
    # all reflections lie in the plane perpendicular to the beam
    assert np.allclose(refl.hkl[:, 2], 0)
