"""Kinematical electron diffraction simulator with exact ground truth.

Given a structure type, the simulator draws a zone axis, a lattice parameter,
an orientation, and imaging conditions, then renders a zone-axis diffraction
pattern as a 2D spot image and its azimuthally averaged 1D radial profile.
Every distortion is physically motivated and every label is exact.

Design choices that keep the classification honest:

* **Image size is constant** (``size`` pixels) for every class, so it can never
  be a cue.
* **Absolute scale is randomised** per pattern (the ``scale_frac`` camera-length
  jitter), so the pattern's overall size in pixels does not encode the class.
* **Ring positions are scale-normalised**: reflections are kept out to a multiple
  of the first allowed reflection ``g1``, so the rings land on the same pixel
  radii whatever the lattice parameter. Note this holds for ring *geometry* only,
  not for ring *brightness*: see the scale cue below.
* **Background is randomised** per pattern, so a constant background cannot be a
  cue.

What remains, and what the classifier must use, is the true physics: the ratios
of ring radii, the angular arrangement of spots, and the systematic absences
and weak reflections that separate simple cubic, BCC, FCC, diamond, rock salt,
and HCP.

One cue this does **not** remove, and which is measured rather than assumed: each
structure carries a preset lattice parameter, and the scattering envelope
``exp(-B_dw s^2)`` makes the ring-to-ring intensity fall-off a readout of the
absolute ``|g|``, hence of ``a``. Randomising the camera length removes the
*pixel-scale* cue but not this *intensity-decay* one. Passing an explicit ``a``
(as :func:`crystalclass.datasets.make_scale_cue_dataset` does) is how the
benchmark severs the correlation and measures what it is worth. See
``configs/scale_cue.yaml`` and RESULTS.md.

Simplifications a microscopist should know about, stated plainly:

* There is no electron wavelength in this model, so the Ewald sphere is flat.
  Real zero-order-Laue-zone reflections have ``s_g ~ -lambda |g|^2 / 2`` even at
  perfect zone-axis alignment, which is why high-``|g|`` reflections dim and why
  the Laue circle exists. Here ``s_g`` is zero for every reflection at zero tilt,
  and the only ``|g|``-dependent dimming is the scattering envelope. No
  accelerating voltage is modelled, and there are no higher-order Laue zones.
* Scattering is kinematical (single). No dynamical redistribution of intensity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crystalclass.structures import STRUCTURE_NAMES, Structure, get_structure, zone_axis_reflections

# Excitation-error width (inverse angstroms). Off-zone tilt damps a reflection
# by exp(-(s_g / EXC_WIDTH)^2) where s_g is its excitation error. First-order
# model: s_g = |g| sin(tilt) cos(phi_g - phi_tilt), where phi_tilt is the azimuth
# the crystal tilts *towards*. Exact only for small tilt, and it omits the
# Ewald-curvature term -lambda |g|^2 / 2 entirely (see the module docstring).
EXC_WIDTH = 0.06


@dataclass
class SimConfig:
    """Configuration for one simulated diffraction pattern.

    Any field left as ``None`` is drawn from a physically sensible range by the
    simulator, which is how domain randomisation is applied.

    Attributes:
        structure: Structure name, or ``None`` to draw one uniformly.
        size: Image side length in pixels (constant across all patterns).
        a: Lattice parameter in angstroms, or ``None`` to jitter around the
            preset value.
        zone_axis: Beam direction, or ``None`` to draw from the structure's
            low-index zone axes.
        dose: Counts at the brightest diffracted spot; lower means noisier.
        orientation_spread: Scale parameter (in degrees) of the off-zone tilt.
            The tilt is drawn as ``abs(Normal(0, orientation_spread))``, so this
            is the sigma of the underlying normal *before* the absolute value,
            not the standard deviation of the realised tilt. The resulting
            half-normal has mean ``0.80 * orientation_spread`` and standard
            deviation ``0.60 * orientation_spread``.
        keep_fraction: Fraction of allowed spots retained. Spots are dropped
            **independently of their intensity**, so this models unpredictable
            spot loss (detector gaps, beam stop, aperture) and acts as a
            robustness augmentation. It is deliberately *not* a detection
            threshold: a real detection limit loses the weakest reflections
            first, which this does not do.
        ring_span: Reflections kept out to ``ring_span * g1``; controls how many
            rings are visible independent of lattice parameter.
        spot_sigma: Spot radius in pixels.
        background_level: Constant background pedestal (intensity units).
        diffuse_level: Amplitude of the smooth central diffuse background.
        readout_counts: Gaussian readout noise in absolute counts, independent
            of dose, as a detector's readout floor actually behaves. It
            therefore dominates only at low dose, where Poisson counting noise
            is small.
        scale_frac: Fraction of the half-frame that ``ring_span * g1`` maps to;
            the camera-length jitter. ``None`` draws it per pattern.
        blank_spots: If True, skip rendering the diffracted spots and keep only
            the central beam and background. Used by the leakage control: with
            the physics removed, any honest classifier must fall to chance.
    """

    structure: str | None = None
    size: int = 128
    a: float | None = None
    zone_axis: tuple[int, int, int] | None = None
    dose: float = 200.0
    orientation_spread: float = 3.0
    keep_fraction: float = 1.0
    ring_span: float | None = None
    spot_sigma: float = 1.6
    background_level: float | None = None
    diffuse_level: float | None = None
    readout_counts: float = 1.0
    scale_frac: float | None = None
    blank_spots: bool = False


@dataclass
class Pattern:
    """A simulated pattern with its exact ground truth.

    Attributes:
        image: Noisy 2D pattern in counts, shape ``(size, size)``.
        profile: Azimuthally averaged radial profile, shape ``(size // 2,)``.
        label: Structure name (the classification target).
        label_index: Index of ``label`` in :data:`LABELS`.
        spots_xy: Pixel coordinates of the rendered diffracted spots, ``(m, 2)``.
        spots_intensity: Normalised intensities of those spots, ``(m,)``.
        spots_hkl: Miller indices of those spots, ``(m, 3)``.
        meta: Dictionary of the realised generative parameters.
    """

    image: np.ndarray
    profile: np.ndarray
    label: str
    label_index: int
    spots_xy: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    spots_intensity: np.ndarray = field(default_factory=lambda: np.empty(0))
    spots_hkl: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    meta: dict = field(default_factory=dict)


LABELS: tuple[str, ...] = STRUCTURE_NAMES
LABEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(LABELS)}


def _draw_lattice_parameter(structure: Structure, rng: np.random.Generator) -> float:
    """Jitter the lattice parameter by +-8% around the preset value."""
    return float(structure.a * rng.uniform(0.92, 1.08))


def _radial_profile(image: np.ndarray) -> np.ndarray:
    """Azimuthal average of ``image`` about its centre.

    Returns:
        Mean intensity in integer-radius annuli from 0 to ``size // 2 - 1``.
    """
    size = image.shape[0]
    cy = cx = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    nbins = size // 2
    sums = np.bincount(r.ravel(), weights=image.ravel(), minlength=size)
    counts = np.bincount(r.ravel(), minlength=size)
    counts = np.where(counts == 0, 1, counts)
    return (sums / counts)[:nbins].astype(np.float32)


def simulate(config: SimConfig, rng: np.random.Generator) -> Pattern:
    """Simulate one diffraction pattern from ``config``.

    Args:
        config: The (partially randomised) generative configuration.
        rng: A NumPy random generator; the sole source of randomness.

    Returns:
        A :class:`Pattern` with the noisy image, radial profile, exact label,
        rendered spot list, and the realised generative parameters.
    """
    name = config.structure if config.structure is not None else str(rng.choice(LABELS))
    structure = get_structure(name)
    a = config.a if config.a is not None else _draw_lattice_parameter(structure, rng)
    zone = (
        config.zone_axis
        if config.zone_axis is not None
        else tuple(structure.zone_axes[rng.integers(len(structure.zone_axes))])
    )
    ring_span = config.ring_span if config.ring_span is not None else float(rng.uniform(3.2, 4.4))

    # Enumerate reflections generously, then keep out to ring_span * g1.
    wide = zone_axis_reflections(structure, zone, a, g_max=12.0 / a)
    if wide.g_len.size == 0:
        # Degenerate zone axis (no in-plane reflections); fall back to [001].
        zone = tuple(structure.zone_axes[0])
        wide = zone_axis_reflections(structure, zone, a, g_max=12.0 / a)
    g1 = float(wide.g_len.min())
    keep = wide.g_len <= ring_span * g1
    hkl = wide.hkl[keep]
    g_cart = wide.g_cart[keep]
    g_len = wide.g_len[keep]
    intensity = wide.intensity[keep].astype(float)

    # Two-dimensional detector coordinates: project g onto a basis of the plane
    # perpendicular to the beam. The reflections already lie in this plane.
    lattice = structure.lattice_matrix(a)
    beam = lattice @ np.array(zone, dtype=float)
    beam = beam / np.linalg.norm(beam)
    ref = np.array([0.0, 0.0, 1.0]) if abs(beam[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(beam, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(beam, e1)
    coords2d = np.column_stack([g_cart @ e1, g_cart @ e2])

    # Off-zone tilt: excitation-error damping (first-order).
    tilt = abs(rng.normal(0.0, np.deg2rad(config.orientation_spread)))
    phi_axis = rng.uniform(0.0, 2.0 * np.pi)
    phi_g = np.arctan2(coords2d[:, 1], coords2d[:, 0])
    s_g = g_len * np.sin(tilt) * np.cos(phi_g - phi_axis)
    intensity = intensity * np.exp(-((s_g / EXC_WIDTH) ** 2))

    # In-plane azimuthal orientation.
    theta = rng.uniform(0.0, 2.0 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords2d = coords2d @ rot.T

    # Missing reflections: randomly drop spots below the detection threshold.
    if config.keep_fraction < 1.0 and intensity.size:
        keep_mask = rng.random(intensity.size) < config.keep_fraction
        coords2d, g_len, hkl, intensity = (
            coords2d[keep_mask],
            g_len[keep_mask],
            hkl[keep_mask],
            intensity[keep_mask],
        )

    if intensity.size and intensity.max() > 0:
        intensity = intensity / intensity.max()

    # Map reciprocal coordinates to pixels with a randomised camera length.
    size = config.size
    center = (size - 1) / 2.0
    scale_frac = config.scale_frac if config.scale_frac is not None else float(rng.uniform(0.64, 0.78))
    g_edge = ring_span * g1
    pix_per_ig = scale_frac * (size / 2.0) / g_edge
    spots_xy = center + coords2d * pix_per_ig

    # Render the clean intensity image: diffracted spots + central beam.
    clean = np.zeros((size, size), dtype=float)
    yy, xx = np.mgrid[0:size, 0:size]
    inside = (
        (spots_xy[:, 0] >= 0) & (spots_xy[:, 0] < size) & (spots_xy[:, 1] >= 0) & (spots_xy[:, 1] < size)
        if spots_xy.size
        else np.zeros(0, dtype=bool)
    )
    spots_xy, g_len, hkl, intensity = (
        spots_xy[inside],
        g_len[inside],
        hkl[inside],
        intensity[inside],
    )
    two_sig2 = 2.0 * config.spot_sigma**2
    if not config.blank_spots:
        for (px, py), amp in zip(spots_xy, intensity):
            clean += amp * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / two_sig2)
    # Central undiffracted beam: fixed brightness, slightly broader, same for all.
    beam_sig2 = 2.0 * (config.spot_sigma * 1.4) ** 2
    clean += 1.0 * np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / beam_sig2)

    # Randomised background: pedestal plus a smooth central diffuse hump.
    ped = config.background_level if config.background_level is not None else float(rng.uniform(0.01, 0.05))
    diff_amp = config.diffuse_level if config.diffuse_level is not None else float(rng.uniform(0.02, 0.10))
    r = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    clean = clean + ped + diff_amp * np.exp(-r / (size / 6.0))

    # Poisson photon noise scaled by dose, plus a dose-independent Gaussian
    # readout floor in absolute counts (a detector property, not an exposure
    # property), so readout matters most where the signal is weakest.
    photons = np.clip(clean * config.dose, 0, None)
    noisy = rng.poisson(photons).astype(np.float32)
    noisy += rng.normal(0.0, config.readout_counts, size=noisy.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, None)

    profile = _radial_profile(noisy)
    meta = {
        "structure": name,
        "zone_axis": tuple(int(z) for z in zone),
        "a": a,
        "dose": config.dose,
        "orientation_spread": config.orientation_spread,
        "tilt_deg": float(np.rad2deg(tilt)),
        "keep_fraction": config.keep_fraction,
        "ring_span": ring_span,
        "scale_frac": scale_frac,
        "n_spots": int(spots_xy.shape[0]),
    }
    return Pattern(
        image=noisy,
        profile=profile,
        label=name,
        label_index=LABEL_INDEX[name],
        spots_xy=spots_xy,
        spots_intensity=intensity,
        spots_hkl=hkl,
        meta=meta,
    )
