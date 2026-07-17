"""Small CPU-friendly CNNs for diffraction-pattern classification.

Two learned models mirror the two representations the classical baseline uses:

* :class:`RadialCNN` is a 1D convolutional network over the radial profile,
  directly comparable to the classical model on the same rotation-invariant
  signal. It isolates the question "does a convnet help over a random forest on
  the same 1D input?"
* :class:`PatternCNN` is a small 2D convolutional network over the polar-Fourier
  map (see :func:`polar_fourier`), which is invariant to the always-present
  in-plane rotation yet keeps the angular symmetry order that azimuthal averaging
  discards.

Preprocessing matters here. The direct beam is masked, and a ``log1p``
compression brings faint reflections (the ones that separate diamond from FCC,
say) into range before standardisation. Both models are tiny enough to train on
CPU in minutes.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from torch import nn

from crystalclass.sim import LABELS

N_CLASSES = len(LABELS)
BEAM_MASK_RADIUS = 5.0  # pixels of the direct beam removed before featurising
N_R = 48  # radial samples in the polar transform
N_THETA = 180  # angular samples in the polar transform (dense, to limit aliasing)
N_FREQ = 24  # angular-frequency bins kept after the real FFT (symmetry lives low)


def polar_fourier(image: np.ndarray) -> np.ndarray:
    """Rotation-invariant polar-Fourier map of a diffraction pattern.

    The pattern is resampled onto a polar grid centred on the direct beam, then
    the magnitude of the FFT along the angular axis is taken. An in-plane
    rotation of the crystal is a cyclic shift along that axis, so its Fourier
    magnitude is unchanged: the map is rotation invariant. It keeps both the ring
    radii (the radial axis) and the angular symmetry order, four-fold versus
    six-fold, which appears as energy at specific angular frequencies. The camera
    length is only mildly randomised in the data, and indexing the radial axis to
    a noisy first-ring estimate was measured to hurt more than it helped, so the
    radial axis is left in absolute pixels here.

    Args:
        image: A 2D diffraction pattern.

    Returns:
        A ``(N_R, N_FREQ)`` float32 map, log-compressed and standardised.
    """
    image = np.clip(image.astype(np.float32), 0, None)
    size = image.shape[0]
    c = (size - 1) / 2.0
    r_max = size / 2.0 - 1.0
    radii = np.linspace(BEAM_MASK_RADIUS, r_max, N_R)
    thetas = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    rr, tt = np.meshgrid(radii, thetas, indexing="ij")
    ys = c + rr * np.sin(tt)
    xs = c + rr * np.cos(tt)
    polar = map_coordinates(image, [ys, xs], order=1, mode="constant", cval=0.0)
    mag = np.abs(np.fft.rfft(polar, axis=1))[:, :N_FREQ]  # rotation-invariant, low freqs
    mag = np.log1p(mag)
    return ((mag - mag.mean()) / (mag.std() + 1e-6)).astype(np.float32)


# The 2D model consumes the rotation-invariant polar-Fourier map.
preprocess_image = polar_fourier


def preprocess_profile(profile: np.ndarray) -> np.ndarray:
    """Zero the beam bins, log-compress, and standardise a radial profile."""
    profile = profile.astype(np.float32).copy()
    beam_bins = int(np.ceil(BEAM_MASK_RADIUS))
    profile[:beam_bins] = 0.0
    profile = np.log1p(np.clip(profile, 0, None))
    return ((profile - profile.mean()) / (profile.std() + 1e-6)).astype(np.float32)


# Backwards-compatible aliases used elsewhere in the package.
standardize_image = preprocess_image
standardize_profile = preprocess_profile


class RadialCNN(nn.Module):
    """1D CNN over the radial profile with a spatial-preserving head."""

    def __init__(self, length: int = 64, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        reduced = length // 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * reduced, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of profiles ``(b, 1, length)`` to class logits."""
        return self.head(self.features(x))


class PatternCNN(nn.Module):
    """2D CNN over the rotation-invariant polar-Fourier map of a pattern.

    Input is the ``(N_R, N_FREQ)`` map from :func:`polar_fourier`: the radial
    axis carries ring radii, the angular-frequency axis carries symmetry order.
    A standard CNN with a flattening head reads both, so this model can use the
    angular geometry that the 1D radial profile discards.
    """

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48x24 -> 24x12
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24x12 -> 12x6
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x6 -> 6x3
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(48 * 6 * 3, 96),
            nn.ReLU(),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of polar-Fourier maps ``(b, 1, N_R, N_FREQ)`` to logits."""
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_radial(model: RadialCNN, profiles: np.ndarray, batch: int = 256) -> np.ndarray:
    """Predict class indices from a stack of radial profiles."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(profiles), batch):
            chunk = profiles[i : i + batch]
            x = np.stack([preprocess_profile(p) for p in chunk])[:, None, :]
            logits = model(torch.from_numpy(x.astype(np.float32)))
            out.append(logits.argmax(1).numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=int)


def predict_pattern(model: PatternCNN, images: np.ndarray, batch: int = 128) -> np.ndarray:
    """Predict class indices from a stack of patterns."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i : i + batch]
            x = np.stack([preprocess_image(im) for im in chunk])[:, None, :, :]
            logits = model(torch.from_numpy(x.astype(np.float32)))
            out.append(logits.argmax(1).numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=int)
