"""Rotation- and scale-invariant features for the classical baseline.

A crystallographer indexes a pattern from the *ratios* of ring radii and the
angular arrangement of spots, not their absolute size. These features encode
exactly that, so the classical baseline is given a fair, physically grounded
representation rather than raw pixels:

* a radial profile resampled onto a scale-free axis ``r / r1`` (``r1`` is the
  first ring), which captures ring spacings and relative intensities;
* the ring-radius ratios and relative ring heights;
* the multiplicity and angular regularity of the inner spot ring, which recover
  the in-plane symmetry (four-fold, six-fold, two-fold) that azimuthal
  averaging throws away.

Everything is invariant to in-plane rotation and to the camera-length jitter,
so nothing here can exploit the absolute scale of a pattern.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

from crystalclass.sim import _radial_profile

N_PROFILE = 48  # resampled scale-free radial-profile length
N_RINGS = 6  # ring-ratio and ring-height features
U_MAX = 5.0  # resample the profile out to r / r1 = U_MAX
FEATURE_DIM = N_PROFILE + 2 * N_RINGS + 4


def _find_rings(profile: np.ndarray, r_min: int) -> tuple[np.ndarray, np.ndarray]:
    """Detect ring peaks in a radial profile beyond the central beam.

    Returns:
        Sorted ring radii (pixels) and their heights, strongest excluded of the
        central beam region ``r < r_min``.
    """
    tail = profile[r_min:]
    if tail.size < 3 or tail.max() <= 0:
        return np.empty(0), np.empty(0)
    height = 0.05 * tail.max()
    idx, props = find_peaks(tail, height=height, distance=2)
    if idx.size == 0:
        return np.empty(0), np.empty(0)
    radii = idx + r_min
    heights = props["peak_heights"]
    order = np.argsort(radii)
    return radii[order].astype(float), heights[order]


def _spot_peaks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect diffracted spots as local maxima above a background estimate.

    Returns:
        Spot pixel coordinates ``(m, 2)`` as ``(row, col)`` and their heights,
        with the central beam removed.
    """
    smooth = ndimage.gaussian_filter(image, 4.0)
    resid = image - smooth
    if resid.max() <= 0:
        return np.empty((0, 2)), np.empty(0)
    thresh = 0.15 * resid.max()
    local_max = ndimage.maximum_filter(resid, size=3)
    peaks = (resid == local_max) & (resid > thresh)
    coords = np.column_stack(np.nonzero(peaks))
    if coords.size == 0:
        return np.empty((0, 2)), np.empty(0)
    center = (image.shape[0] - 1) / 2.0
    r = np.sqrt((coords[:, 0] - center) ** 2 + (coords[:, 1] - center) ** 2)
    beam = r > 3.0
    coords = coords[beam]
    heights = resid[coords[:, 0], coords[:, 1]] if coords.size else np.empty(0)
    return coords.astype(float), heights


def _inner_ring_geometry(coords: np.ndarray, center: float) -> tuple[float, float]:
    """Multiplicity and angular-gap regularity of the innermost spot ring.

    Returns:
        ``(multiplicity, angular_gap_std)``. The multiplicity is the number of
        spots within 25% of the smallest spot radius; the angular-gap std (in
        radians) measures how evenly they are spaced, a proxy for symmetry.
    """
    if coords.shape[0] < 2:
        return 0.0, 0.0
    r = np.sqrt((coords[:, 0] - center) ** 2 + (coords[:, 1] - center) ** 2)
    r_inner = r.min()
    on_inner = r <= 1.25 * r_inner
    ring = coords[on_inner]
    if ring.shape[0] < 2:
        return float(ring.shape[0]), 0.0
    ang = np.sort(np.arctan2(ring[:, 0] - center, ring[:, 1] - center))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return float(ring.shape[0]), float(np.std(gaps))


def extract_features(image: np.ndarray) -> np.ndarray:
    """Extract the scale- and rotation-invariant feature vector for one pattern.

    Args:
        image: A 2D diffraction pattern.

    Returns:
        A length-:data:`FEATURE_DIM` feature vector.
    """
    profile = _radial_profile(image)
    r_min = max(3, int(0.05 * image.shape[0]))
    radii, heights = _find_rings(profile, r_min)

    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    if radii.size == 0:
        return feat

    r1 = radii[0]
    # Scale-free resampled radial profile on the axis u = r / r1.
    px = np.arange(profile.size)
    u = px / r1
    grid = np.linspace(0.0, U_MAX, N_PROFILE)
    resampled = np.interp(grid, u, profile)
    amp = resampled[grid > 0.5].max() if np.any(grid > 0.5) else resampled.max()
    if amp > 0:
        resampled = resampled / amp
    feat[:N_PROFILE] = resampled

    # Ring-radius ratios and relative heights.
    ratios = (radii[:N_RINGS] / r1)[:N_RINGS]
    feat[N_PROFILE : N_PROFILE + ratios.size] = ratios
    rel_h = (heights[:N_RINGS] / heights.max()) if heights.max() > 0 else heights[:N_RINGS]
    feat[N_PROFILE + N_RINGS : N_PROFILE + N_RINGS + rel_h.size] = rel_h

    # Scalar 2D geometry features.
    coords, _ = _spot_peaks(image)
    center = (image.shape[0] - 1) / 2.0
    mult, ang_std = _inner_ring_geometry(coords, center)
    tail = N_PROFILE + 2 * N_RINGS
    feat[tail + 0] = min(coords.shape[0], 200) / 200.0
    feat[tail + 1] = min(mult, 12.0) / 12.0
    feat[tail + 2] = ang_std
    feat[tail + 3] = min(radii.size, 12) / 12.0
    return feat


def extract_features_batch(images: np.ndarray) -> np.ndarray:
    """Extract features for a stack of images, shape ``(n, FEATURE_DIM)``."""
    return np.stack([extract_features(img) for img in images])
