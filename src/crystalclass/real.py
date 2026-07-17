"""Bring-your-own-data loader for real electron diffraction images.

The models are trained purely on simulation, so this path is for running them on
your own experimental pattern. There is no ground truth for an arbitrary image,
so the output is a predicted structure and class probabilities, not an accuracy.

Requirements on the input image, documented plainly because they matter:

* The direct beam must be at the image centre. Pass ``center`` if it is not, or
  crop the image so it is.
* The pattern is resampled to the model's input size. The ring geometry is what
  the model reads, so the whole pattern should be visible and roughly centred.
* Bright-field or real-space images are not diffraction patterns; this will
  return a meaningless label for them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from crystalclass.net import PatternCNN


def load_diffraction_image(
    path: str | Path,
    size: int = 128,
    center: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load a real diffraction image and resample it to the model input size.

    Args:
        path: Path to a PNG, TIFF, or JPEG image.
        size: Output side length.
        center: ``(row, col)`` of the direct beam; defaults to the image centre.

    Returns:
        A ``(size, size)`` float32 image with the beam centred.
    """
    img = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    h, w = img.shape
    cy, cx = center if center is not None else (h // 2, w // 2)
    half = min(cy, cx, h - cy, w - cx)
    crop = img[cy - half : cy + half, cx - half : cx + half]
    resized = np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR), dtype=np.float32)
    return resized


def classify_image(
    model: PatternCNN,
    image: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Return the predicted class index and softmax probabilities for an image."""
    import torch

    from crystalclass.net import preprocess_image

    x = preprocess_image(image)[None, None, :, :].astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return int(probs.argmax()), probs
