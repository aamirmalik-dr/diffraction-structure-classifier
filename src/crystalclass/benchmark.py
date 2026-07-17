"""Config-driven benchmark harness.

A YAML config selects a mode and the methods to compare. Every mode builds its
test data from fixed seeds, so the committed JSON results and figures regenerate
exactly.

Modes:
    sweep       vary one physical parameter (dose, keep_fraction, or
                orientation_spread) and record accuracy and macro-F1 per method.
    compare     one test set, full confusion matrices and per-class scores, plus
                the classical default-vs-tuned gap.
    leakage     blank the diffracted spots (physics removed) and a label-shuffle
                control; both must collapse to chance for an honest model.
    ablation    retrain the pattern CNN with one randomisation component removed
                and re-score on the standard test set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

from crystalclass.classical import train_classical
from crystalclass.datasets import Dataset, make_dataset, make_training_dataset
from crystalclass.metrics import accuracy, macro_f1, summarize
from crystalclass.net import predict_pattern, predict_radial
from crystalclass.sim import LABELS, SimConfig
from crystalclass.train import (
    TrainSettings,
    load_pattern_model,
    load_radial_model,
    train_model,
)

Method = Callable[[Dataset], np.ndarray]

# Reference imaging conditions held fixed while another axis is swept.
REFERENCE = {"dose": 150.0, "orientation_spread": 2.0, "keep_fraction": 1.0}


def _test_config(overrides: dict) -> SimConfig:
    base = SimConfig(
        dose=REFERENCE["dose"],
        orientation_spread=REFERENCE["orientation_spread"],
        keep_fraction=REFERENCE["keep_fraction"],
    )
    return replace(base, **overrides)


def load_methods(specs: list[dict], models_dir: Path, profile_len: int) -> dict[str, Method]:
    """Instantiate prediction callables from method specs.

    Args:
        specs: List of ``{"name", "kind", "path"}`` entries. ``kind`` is one of
            ``classical``, ``radial``, ``pattern``.
        models_dir: Directory holding the committed artifacts.
        profile_len: Radial-profile length, needed to rebuild the 1D CNN.

    Returns:
        Mapping from method name to a callable ``Dataset -> predictions``.
    """
    methods: dict[str, Method] = {}
    for spec in specs:
        kind, path = spec["kind"], models_dir / spec["path"]
        if kind == "classical":
            model = joblib.load(path)
            methods[spec["name"]] = lambda ds, m=model: m.predict(ds.images)
        elif kind == "radial":
            model = load_radial_model(str(path), length=profile_len)
            methods[spec["name"]] = lambda ds, m=model: predict_radial(m, ds.profiles)
        elif kind == "pattern":
            model = load_pattern_model(str(path))
            methods[spec["name"]] = lambda ds, m=model: predict_pattern(m, ds.images)
        else:
            raise ValueError(f"unknown method kind {kind!r}")
    return methods


def _run_sweep(config: dict, methods: dict[str, Method]) -> dict:
    axis = config["axis"]  # "dose" | "keep_fraction" | "orientation_spread"
    values = config["values"]
    n_per_class = config.get("n_per_class", 30)
    seed = config.get("seed", 100)
    rows = []
    for j, value in enumerate(values):
        ds = make_dataset(
            n_per_class * len(LABELS),
            seed=seed + j,
            base=_test_config({axis: value}),
        )
        entry = {"value": value}
        for name, method in methods.items():
            pred = method(ds)
            entry[name] = {
                "accuracy": accuracy(ds.labels, pred),
                "macro_f1": macro_f1(_confusion(ds.labels, pred)),
            }
        rows.append(entry)
    return {"mode": "sweep", "axis": axis, "reference": REFERENCE, "rows": rows}


def _confusion(y_true, y_pred):
    from crystalclass.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred)


def _run_compare(config: dict, methods: dict[str, Method], models_dir: Path) -> dict:
    n_per_class = config.get("n_per_class", 60)
    seed = config.get("seed", 200)
    overrides = config.get("test_overrides", {})
    ds = make_dataset(n_per_class * len(LABELS), seed=seed, base=_test_config(overrides))
    out = {"mode": "compare", "test_overrides": overrides, "methods": {}}
    for name, method in methods.items():
        pred = method(ds)
        out["methods"][name] = summarize(ds.labels, pred)

    # Fair-tuning artifact: how much does grid-search tuning buy the baseline?
    if config.get("classical_default_check"):
        train = make_training_dataset(config.get("tune_pool", 1200), seed=config.get("tune_seed", 7))
        tuned = train_classical(train.images, train.labels, kind="rf", default=False)
        default = train_classical(train.images, train.labels, kind="rf", default=True)
        out["classical_tuning"] = {
            "tuned_accuracy": accuracy(ds.labels, tuned.predict(ds.images)),
            "default_accuracy": accuracy(ds.labels, default.predict(ds.images)),
            "best_params": tuned.best_params,
            "cv_score": tuned.cv_score,
        }
    return out


def _run_leakage(config: dict, methods: dict[str, Method]) -> dict:
    """Physics-removed and label-shuffle controls; both should hit chance."""
    n_per_class = config.get("n_per_class", 40)
    seed = config.get("seed", 300)
    chance = 1.0 / len(LABELS)

    # 1. Blank-spots control: keep beam + background, remove all diffraction.
    blank = make_dataset(
        n_per_class * len(LABELS),
        seed=seed,
        base=_test_config({"blank_spots": True}),
    )
    blank_scores = {name: accuracy(blank.labels, method(blank)) for name, method in methods.items()}

    # 2. Trained-on-blank control: fit a fresh classical model on blanked data;
    # if it beats chance, the background leaks the class.
    blank_train = make_training_dataset(config.get("train_pool", 900), seed=seed + 1)
    blank_imgs = make_dataset(
        len(blank_train.labels),
        seed=seed + 2,
        base=_test_config({"blank_spots": True}),
    )
    trained = train_classical(blank_imgs.images, blank_train.labels, kind="rf", default=True)
    blank_test = make_dataset(
        n_per_class * len(LABELS),
        seed=seed + 3,
        base=_test_config({"blank_spots": True}),
    )
    trained_on_blank_acc = accuracy(blank_test.labels, trained.predict(blank_test.images))

    # 3. Label-shuffle control on real patterns: classical fit to shuffled labels.
    real_train = make_training_dataset(config.get("train_pool", 900), seed=seed + 4)
    rng = np.random.default_rng(seed + 5)
    shuffled = real_train.labels.copy()
    rng.shuffle(shuffled)
    shuffle_model = train_classical(real_train.images, shuffled, kind="rf", default=True)
    real_test = make_dataset(n_per_class * len(LABELS), seed=seed + 6, base=_test_config({}))
    shuffle_acc = accuracy(real_test.labels, shuffle_model.predict(real_test.images))

    return {
        "mode": "leakage",
        "chance_level": chance,
        "blank_spots_accuracy": blank_scores,
        "trained_on_blank_accuracy": trained_on_blank_acc,
        "shuffled_label_accuracy": shuffle_acc,
    }


def _run_ablation(config: dict) -> dict:
    """Retrain the pattern CNN with a randomisation component removed."""
    n_per_class = config.get("n_per_class", 40)
    seed = config.get("seed", 400)
    epochs = config.get("epochs", 8)
    pool_size = config.get("pool_size", 3000)
    variants = {
        "full": {},
        "fixed_scale": {"randomize_scale": False},
        "fixed_background": {"randomize_background": False},
    }
    test = make_dataset(n_per_class * len(LABELS), seed=seed, base=_test_config({}))
    hard = make_dataset(n_per_class * len(LABELS), seed=seed + 1, base=_test_config({"dose": 20.0}))
    results = {}
    for variant, kwargs in variants.items():
        settings = TrainSettings(model="pattern", pool_size=pool_size, epochs=epochs, seed=seed, **kwargs)
        model, history = train_model(settings)
        results[variant] = {
            "test_accuracy": accuracy(test.labels, predict_pattern(model, test.images)),
            "low_dose_accuracy": accuracy(hard.labels, predict_pattern(model, hard.images)),
            "train_accuracy": history["train_accuracy"],
        }
    return {"mode": "ablation", "reference": REFERENCE, "variants": results}


def run_config(config: dict, models_dir: str | Path, profile_len: int = 64) -> dict:
    """Dispatch a benchmark config to its mode and return the result dict."""
    models_dir = Path(models_dir)
    mode = config["mode"]
    if mode == "ablation":
        return _run_ablation(config)
    methods = load_methods(config["methods"], models_dir, profile_len)
    if mode == "sweep":
        return _run_sweep(config, methods)
    if mode == "compare":
        return _run_compare(config, methods, models_dir)
    if mode == "leakage":
        return _run_leakage(config, methods)
    raise ValueError(f"unknown benchmark mode {mode!r}")
