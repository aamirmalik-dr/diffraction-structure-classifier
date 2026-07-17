import numpy as np

from crystalclass.classical import train_classical
from crystalclass.datasets import make_dataset, make_training_dataset
from crystalclass.io import load_pattern, save_pattern
from crystalclass.metrics import accuracy
from crystalclass.sim import LABELS, SimConfig, simulate
from crystalclass.train import TrainSettings, train_model


def test_dataset_class_balance():
    ds = make_dataset(len(LABELS) * 3, seed=0)
    counts = np.bincount(ds.labels, minlength=len(LABELS))
    assert np.all(counts == 3)
    assert ds.images.shape == (len(LABELS) * 3, 128, 128)
    assert ds.profiles.shape[0] == len(LABELS) * 3


def test_io_roundtrip(tmp_path):
    p = simulate(SimConfig(structure="rocksalt", dose=100), np.random.default_rng(1))
    path = tmp_path / "s.npz"
    save_pattern(path, p)
    q = load_pattern(path)
    assert q.label == p.label
    assert np.array_equal(q.image, p.image)
    # JSON metadata round-trips tuples as lists, so compare element-wise.
    assert tuple(q.meta["zone_axis"]) == tuple(p.meta["zone_axis"])


def test_classical_beats_chance():
    train = make_training_dataset(180, seed=0)
    test = make_training_dataset(60, seed=1)
    model = train_classical(train.images, train.labels, kind="rf", default=True)
    acc = accuracy(test.labels, model.predict(test.images))
    assert acc > 1.0 / len(LABELS)


def test_cnn_trains_and_beats_chance():
    settings = TrainSettings(model="radial", pool_size=180, epochs=3, seed=0)
    model, history = train_model(settings)
    assert np.isfinite(history["loss"]).all()
    assert history["train_accuracy"] > 1.0 / len(LABELS)


def test_real_image_roundtrip(tmp_path):
    from PIL import Image

    from crystalclass.real import classify_image, load_diffraction_image
    from crystalclass.train import train_model

    p = simulate(SimConfig(structure="fcc", dose=300), np.random.default_rng(0))
    img8 = (255 * p.image / p.image.max()).astype("uint8")
    png = tmp_path / "pattern.png"
    Image.fromarray(img8).save(png)
    loaded = load_diffraction_image(png, size=128)
    assert loaded.shape == (128, 128)

    model, _ = train_model(TrainSettings(model="pattern", pool_size=120, epochs=2, seed=0))
    idx, probs = classify_image(model, loaded)
    assert 0 <= idx < len(LABELS)
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)
