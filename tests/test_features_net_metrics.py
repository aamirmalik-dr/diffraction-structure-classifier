import numpy as np
import torch

from crystalclass.features import FEATURE_DIM, extract_features, extract_features_batch
from crystalclass.metrics import accuracy, confusion_matrix, macro_f1, per_class_scores
from crystalclass.net import N_FREQ, N_R, PatternCNN, RadialCNN, count_parameters, polar_fourier
from crystalclass.sim import LABELS, SimConfig, simulate


def test_feature_dim_and_finite():
    p = simulate(SimConfig(structure="fcc", dose=300), np.random.default_rng(0))
    feat = extract_features(p.image)
    assert feat.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(feat))
    assert feat.any()  # a clean pattern yields non-zero ring features


def test_feature_batch_shape():
    imgs = np.stack([simulate(SimConfig(structure="bcc"), np.random.default_rng(i)).image for i in range(4)])
    feats = extract_features_batch(imgs)
    assert feats.shape == (4, FEATURE_DIM)


def test_radial_cnn_forward():
    model = RadialCNN(length=64)
    out = model(torch.randn(3, 1, 64))
    assert out.shape == (3, len(LABELS))
    assert count_parameters(model) < 100_000


def test_pattern_cnn_forward():
    model = PatternCNN()
    out = model(torch.randn(2, 1, N_R, N_FREQ))
    assert out.shape == (2, len(LABELS))
    assert count_parameters(model) < 400_000


def test_polar_fourier_rotation_invariant():
    # The polar-Fourier map should be far less sensitive to an in-plane rotation
    # than the raw image is. Sparse-spot discretisation keeps it from being exact.
    from scipy.ndimage import rotate

    p = simulate(SimConfig(structure="fcc", dose=2000, orientation_spread=0.0), np.random.default_rng(0))
    rotated = rotate(p.image, 37.0, reshape=False, order=1)
    m0, m1 = polar_fourier(p.image), polar_fourier(rotated)
    map_rel = np.linalg.norm(m0 - m1) / (np.linalg.norm(m0) + 1e-6)

    def std_img(a):
        a = a.astype(float)
        return (a - a.mean()) / (a.std() + 1e-6)

    img_rel = np.linalg.norm(std_img(p.image) - std_img(rotated)) / (np.linalg.norm(std_img(p.image)) + 1e-6)
    assert m0.shape == (N_R, N_FREQ)
    assert map_rel < img_rel  # the transform genuinely suppresses rotation sensitivity


def test_metrics_basic():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.sum() == 6
    assert np.isclose(accuracy(y_true, y_pred), 5 / 6)
    scores = per_class_scores(cm)
    assert set(scores) == set(LABELS)
    assert 0.0 <= macro_f1(cm) <= 1.0
