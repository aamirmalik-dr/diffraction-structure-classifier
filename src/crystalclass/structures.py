"""Crystal structures, reciprocal lattices, and kinematical structure factors.

This module is the physical ground truth of the project. Each structure type
is defined by a conventional lattice (cubic or hexagonal) and an atomic basis.
Everything else, systematic absences included, is derived from the basis by
computing the structure factor, so the extinction rules that distinguish the
Bravais lattices are never hard coded. They emerge from

    F(hkl) = sum_j  f_j(s) * exp(2 pi i (h x_j + k y_j + l z_j))

being zero (or weak) for particular (h, k, l). This keeps the labels exact:
a pattern's class is the structure it was generated from, and the allowed
reflections follow from first principles.

Conventions
-----------
Real-space lattice vectors are the columns of ``A`` in angstroms. The
crystallographic reciprocal basis (no 2 pi factor) is ``B = inv(A).T`` so that
``b_i . a_j = delta_ij`` and ``|g_hkl| = 1 / d_hkl``. The scattering variable
is ``s = |g| / 2 = sin(theta) / lambda``. The electron atomic scattering factor
uses a single-Gaussian screened form ``f(s) = Z * exp(-B_dw * s^2)``: monotone
in ``s``, increasing in atomic number ``Z``, and, crucially, its exact
functional shape does not affect which reflections are extinct, since absences
come from the complex phase sum alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Debye-Waller-like envelope for the atomic scattering factor, in angstrom^2.
# Sets how fast spot intensity falls with scattering angle. Physical, not tuned.
B_DW = 3.0


@dataclass(frozen=True)
class Structure:
    """A crystal structure type: conventional lattice plus atomic basis.

    Attributes:
        name: Short identifier, also the classification label.
        system: Crystal system, ``"cubic"`` or ``"hexagonal"``.
        a: Conventional lattice parameter in angstroms.
        c_over_a: Axial ratio for hexagonal cells; ignored for cubic.
        basis: Tuple of ``(x, y, z, Z)`` entries, fractional coordinates and
            atomic number for each atom in the conventional cell.
        zone_axes: Low-index beam directions ``[u, v, w]`` used for sampling
            zone-axis patterns, expressed in the conventional cell.
    """

    name: str
    system: str
    a: float
    c_over_a: float
    basis: tuple[tuple[float, float, float, int], ...]
    zone_axes: tuple[tuple[int, int, int], ...]

    def lattice_matrix(self, a: float | None = None) -> np.ndarray:
        """Return the real-space lattice matrix (columns are a1, a2, a3).

        Args:
            a: Override lattice parameter; defaults to ``self.a``.

        Returns:
            A 3x3 array in angstroms.
        """
        a = self.a if a is None else a
        if self.system == "cubic":
            return a * np.eye(3)
        if self.system == "hexagonal":
            c = a * self.c_over_a
            return np.array(
                [
                    [a, -0.5 * a, 0.0],
                    [0.0, np.sqrt(3.0) / 2.0 * a, 0.0],
                    [0.0, 0.0, c],
                ]
            )
        raise ValueError(f"unknown crystal system: {self.system}")


# The six structure types. Lattice parameters are representative of real
# materials but are randomised per sample by the simulator, so absolute scale
# is never a usable classification cue.
_STRUCTURES: dict[str, Structure] = {}


def _register(structure: Structure) -> None:
    _STRUCTURES[structure.name] = structure


# Principal low-index zone axes: the orientations a microscopist actually
# aligns to. Restricting to these keeps the zone-axis pattern well posed, since
# the visible reflections depend jointly on structure and beam direction.
_CUBIC_ZONES = ((0, 0, 1), (0, 1, 1), (1, 1, 1))
_HEX_ZONES = ((0, 0, 1), (1, 0, 0), (1, 1, 0))

_register(
    Structure(
        name="sc",
        system="cubic",
        a=3.34,
        c_over_a=1.0,
        basis=((0.0, 0.0, 0.0, 84),),  # alpha-Po, simple cubic
        zone_axes=_CUBIC_ZONES,
    )
)
_register(
    Structure(
        name="bcc",
        system="cubic",
        a=2.87,
        c_over_a=1.0,
        basis=((0.0, 0.0, 0.0, 26), (0.5, 0.5, 0.5, 26)),  # alpha-Fe
        zone_axes=_CUBIC_ZONES,
    )
)
_register(
    Structure(
        name="fcc",
        system="cubic",
        a=3.61,
        c_over_a=1.0,
        basis=(
            (0.0, 0.0, 0.0, 29),
            (0.5, 0.5, 0.0, 29),
            (0.5, 0.0, 0.5, 29),
            (0.0, 0.5, 0.5, 29),
        ),  # Cu
        zone_axes=_CUBIC_ZONES,
    )
)
_register(
    Structure(
        name="diamond",
        system="cubic",
        a=5.43,
        c_over_a=1.0,
        basis=(
            (0.0, 0.0, 0.0, 14),
            (0.5, 0.5, 0.0, 14),
            (0.5, 0.0, 0.5, 14),
            (0.0, 0.5, 0.5, 14),
            (0.25, 0.25, 0.25, 14),
            (0.75, 0.75, 0.25, 14),
            (0.75, 0.25, 0.75, 14),
            (0.25, 0.75, 0.75, 14),
        ),  # Si, FCC lattice with a two-atom motif
        zone_axes=_CUBIC_ZONES,
    )
)
_register(
    Structure(
        name="rocksalt",
        system="cubic",
        a=5.64,
        c_over_a=1.0,
        basis=(
            (0.0, 0.0, 0.0, 11),
            (0.5, 0.5, 0.0, 11),
            (0.5, 0.0, 0.5, 11),
            (0.0, 0.5, 0.5, 11),
            (0.5, 0.0, 0.0, 17),
            (0.0, 0.5, 0.0, 17),
            (0.0, 0.0, 0.5, 17),
            (0.5, 0.5, 0.5, 17),
        ),  # NaCl, FCC of Na interpenetrating FCC of Cl
        zone_axes=_CUBIC_ZONES,
    )
)
_register(
    Structure(
        name="hcp",
        system="hexagonal",
        a=2.95,
        c_over_a=1.633,
        basis=((0.0, 0.0, 0.0, 22), (1.0 / 3.0, 2.0 / 3.0, 0.5, 22)),  # alpha-Ti
        zone_axes=_HEX_ZONES,
    )
)

STRUCTURE_NAMES: tuple[str, ...] = tuple(_STRUCTURES.keys())


def get_structure(name: str) -> Structure:
    """Return the registered :class:`Structure` for ``name``."""
    try:
        return _STRUCTURES[name]
    except KeyError as exc:
        raise KeyError(f"unknown structure {name!r}; choose from {STRUCTURE_NAMES}") from exc


def reciprocal_matrix(lattice: np.ndarray) -> np.ndarray:
    """Return the crystallographic reciprocal matrix ``B = inv(A).T``.

    Columns of the result are the reciprocal basis vectors ``b1, b2, b3`` in
    inverse angstroms, satisfying ``b_i . a_j = delta_ij``.
    """
    return np.linalg.inv(lattice).T


@dataclass
class Reflections:
    """A set of reciprocal-lattice reflections with kinematical intensities.

    Attributes:
        hkl: Integer Miller indices, shape ``(n, 3)``.
        g_cart: Cartesian reciprocal vectors in inverse angstroms, shape ``(n, 3)``.
        g_len: Reflection magnitudes ``|g|``, shape ``(n,)``.
        intensity: Kinematical intensity ``|F|^2 * f-envelope``, shape ``(n,)``.
    """

    hkl: np.ndarray
    g_cart: np.ndarray
    g_len: np.ndarray
    intensity: np.ndarray = field(default_factory=lambda: np.empty(0))


def _scattering_factor(z: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Electron atomic scattering factor ``f(s) = Z * exp(-B_dw * s^2)``."""
    return z[None, :] * np.exp(-B_DW * (s[:, None] ** 2))


def structure_factors(structure: Structure, hkl: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Compute kinematical intensities ``|F|^2`` for a set of reflections.

    Args:
        structure: The structure providing the atomic basis.
        hkl: Integer Miller indices, shape ``(n, 3)``.
        lattice: Real-space lattice matrix for the (possibly rescaled) cell.

    Returns:
        Intensities ``|F(hkl)|^2`` including the angle-dependent scattering
        envelope, shape ``(n,)``.
    """
    b = reciprocal_matrix(lattice)
    g_cart = hkl @ b.T
    g_len = np.linalg.norm(g_cart, axis=1)
    s = g_len / 2.0

    coords = np.array([(x, y, z) for x, y, z, _ in structure.basis])
    zvals = np.array([z for *_, z in structure.basis], dtype=float)
    f = _scattering_factor(zvals, s)  # (n, n_atoms)
    phase = 2.0 * np.pi * (hkl @ coords.T)  # (n, n_atoms)
    amplitude = np.sum(f * np.exp(1j * phase), axis=1)
    return np.abs(amplitude) ** 2


def zone_axis_reflections(
    structure: Structure,
    zone_axis: tuple[int, int, int],
    a: float,
    g_max: float,
    hkl_range: int = 8,
    intensity_floor: float = 1e-6,
) -> Reflections:
    """Enumerate zero-order Laue zone reflections for a zone-axis pattern.

    The zero-order Laue zone contains reflections ``(h, k, l)`` with
    ``h u + k v + l w = 0`` (they lie in the plane through the origin
    perpendicular to the beam ``[u, v, w]``). Their Cartesian reciprocal
    vectors are already perpendicular to the beam, so they map directly onto
    the detector plane.

    Args:
        structure: The crystal structure.
        zone_axis: Beam direction ``[u, v, w]`` in the conventional cell.
        a: Lattice parameter for this sample (angstroms).
        g_max: Maximum ``|g|`` to include (inverse angstroms).
        hkl_range: Half-range of Miller indices to enumerate.
        intensity_floor: Drop reflections weaker than this fraction of the
            strongest reflection (numerical and systematic absences).

    Returns:
        A :class:`Reflections` with the surviving reflections, the direct beam
        ``(0, 0, 0)`` excluded.
    """
    lattice = structure.lattice_matrix(a)
    b = reciprocal_matrix(lattice)

    rng = range(-hkl_range, hkl_range + 1)
    grid = np.array([(h, k, ll) for h in rng for k in rng for ll in rng])
    u, v, w = zone_axis
    on_zone = (grid[:, 0] * u + grid[:, 1] * v + grid[:, 2] * w) == 0
    nonzero = np.any(grid != 0, axis=1)
    hkl = grid[on_zone & nonzero]

    g_cart = hkl @ b.T
    g_len = np.linalg.norm(g_cart, axis=1)
    keep = g_len <= g_max
    hkl, g_cart, g_len = hkl[keep], g_cart[keep], g_len[keep]

    intensity = structure_factors(structure, hkl, lattice)
    if intensity.size and intensity.max() > 0:
        strong = intensity > intensity_floor * intensity.max()
        hkl, g_cart, g_len, intensity = (
            hkl[strong],
            g_cart[strong],
            g_len[strong],
            intensity[strong],
        )
    return Reflections(hkl=hkl, g_cart=g_cart, g_len=g_len, intensity=intensity)
