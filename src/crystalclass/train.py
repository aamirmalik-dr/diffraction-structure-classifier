"""Training loops for the radial and pattern CNNs.

Models are trained on a domain-randomised pool of simulated patterns with a
fixed seed, so every committed weight file regenerates exactly. Training runs on
CPU in a few minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from crystalclass.datasets import Dataset, make_training_dataset
from crystalclass.net import (
    PatternCNN,
    RadialCNN,
    preprocess_image,
    preprocess_profile,
)


@dataclass
class TrainSettings:
    """Hyperparameters and data-randomisation switches for a training run.

    Attributes:
        model: ``"radial"`` or ``"pattern"``.
        pool_size: Number of patterns in the training pool.
        epochs: Passes over the pool.
        batch: Minibatch size.
        lr: Adam learning rate.
        seed: Master seed for data generation and initialisation.
        size: Image side length.
        randomize_scale: Randomise the camera-length scale in training data.
        randomize_background: Randomise the background in training data.
    """

    model: str = "pattern"
    pool_size: int = 3000
    epochs: int = 18  # matches the CLI default and the committed weights
    batch: int = 32
    lr: float = 1e-3
    seed: int = 0
    size: int = 128
    randomize_scale: bool = True
    randomize_background: bool = True


def _preprocess_batch(arr: np.ndarray, is_image: bool) -> np.ndarray:
    fn = preprocess_image if is_image else preprocess_profile
    stacked = np.stack([fn(a) for a in arr])
    if is_image:
        return stacked[:, None, :, :].astype(np.float32)
    return stacked[:, None, :].astype(np.float32)


def train_model(
    settings: TrainSettings,
    pool: Dataset | None = None,
    verbose: bool = False,
) -> tuple[nn.Module, dict]:
    """Train a CNN and return the model and a training-history dict.

    Args:
        settings: Training configuration.
        pool: Optional pre-built training pool; generated from ``settings`` if
            not given.
        verbose: If True, print the loss every epoch.

    Returns:
        ``(model, history)`` where ``history`` records the per-epoch mean loss
        and final training accuracy.
    """
    torch.manual_seed(settings.seed)
    is_image = settings.model == "pattern"
    if pool is None:
        pool = make_training_dataset(
            settings.pool_size,
            seed=settings.seed,
            size=settings.size,
            randomize_scale=settings.randomize_scale,
            randomize_background=settings.randomize_background,
        )

    x = _preprocess_batch(pool.images if is_image else pool.profiles, is_image)
    y = pool.labels.astype(np.int64)
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)

    if settings.model == "pattern":
        model: nn.Module = PatternCNN()
    elif settings.model == "radial":
        model = RadialCNN(length=x.shape[-1])
    else:
        raise ValueError(f"unknown model {settings.model!r}")

    opt = torch.optim.Adam(model.parameters(), lr=settings.lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(settings.seed)
    n = len(y)

    losses = []
    model.train()
    for epoch in range(settings.epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for i in range(0, n, settings.batch):
            idx = order[i : i + settings.batch]
            opt.zero_grad()
            logits = model(x_t[idx])
            loss = loss_fn(logits, y_t[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        losses.append(epoch_loss)
        if verbose:
            print(f"epoch {epoch + 1}/{settings.epochs}  loss {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        train_pred = model(x_t).argmax(1).numpy()
    train_acc = float(np.mean(train_pred == y))
    history = {"loss": losses, "train_accuracy": train_acc, "model": settings.model}
    return model, history


def save_model(model: nn.Module, path: str) -> None:
    """Save model weights to ``path``."""
    torch.save(model.state_dict(), path)


def load_pattern_model(path: str) -> PatternCNN:
    """Load a :class:`PatternCNN` from ``path``."""
    model = PatternCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_radial_model(path: str, length: int) -> RadialCNN:
    """Load a :class:`RadialCNN` from ``path``."""
    model = RadialCNN(length=length)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
