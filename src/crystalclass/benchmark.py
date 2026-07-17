"""Config-driven benchmark harness.

A YAML config selects a mode and the methods to compare. Every mode builds its
test data from fixed seeds, so the committed JSON results and figures regenerate
exactly.

Modes:
    sweep       vary one physical parameter (dose, keep_fraction, or
                orientation_spread) and record accuracy and macro-F1 per method.
    compare     one test set, full confusion matrices and per-class scores, 95%
                confidence intervals, paired McNemar tests between every method
                pair, and the classical fair-tuning check (tuned vs default RF,
                tuned RF vs tuned SVM).
    leakage     blank the diffracted spots (physics removed) and a label-shuffle
                control; both must collapse to chance for an honest model.
    ablation    retrain the pattern CNN with one randomisation component removed
                and re-score on the standard test set. Note this measures the
                robustness value of domain randomisation (train narrow, test
                wide); it is not a shortcut test.
    scale_cue   score every method on a test set where the lattice parameter is
                decorrelated from the class, measuring how much accuracy comes
                from material identity rather than structure-type geometry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

from crystalclass.classical import train_classical
from crystalclass.datasets import (
    Dataset,
    make_dataset,
    make_scale_cue_dataset,
    make_training_dataset,
)
from crystalclass.metrics import accuracy, macro_f1, mcnemar, summarize, wilson_interval
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


def reference_config(overrides: dict) -> SimConfig:
    """Build a test SimConfig from the reference imaging conditions.

    Public because the tutorial notebook and docs build test sets with it; the
    benchmark holds :data:`REFERENCE` fixed on every axis it is not sweeping.

    Args:
        overrides: Fields to override on top of :data:`REFERENCE`.

    Returns:
        A :class:`~crystalclass.sim.SimConfig`.
    """
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
            base=reference_config({axis: value}),
        )
        entry = {"value": value, "n_test": int(len(ds.labels))}
        for name, method in methods.items():
            pred = method(ds)
            n_correct = int(np.sum(pred == ds.labels))
            lo, hi = wilson_interval(n_correct, len(ds.labels))
            entry[name] = {
                "accuracy": accuracy(ds.labels, pred),
                "accuracy_ci95": [lo, hi],
                "macro_f1": macro_f1(_confusion(ds.labels, pred)),
            }
        rows.append(entry)
    return {"mode": "sweep", "axis": axis, "reference": REFERENCE, "rows": rows}


def _confusion(y_true, y_pred):
    from crystalclass.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred)


def _pairwise_tests(labels: np.ndarray, preds: dict[str, np.ndarray]) -> list[dict]:
    """Paired McNemar test for every ordered method pair, plus the raw delta."""
    names = list(preds)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            test = mcnemar(labels, preds[a], preds[b])
            rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "delta_accuracy": accuracy(labels, preds[a]) - accuracy(labels, preds[b]),
                    **test,
                }
            )
    return rows


def _run_compare(config: dict, methods: dict[str, Method], models_dir: Path) -> dict:
    n_per_class = config.get("n_per_class", 60)
    seed = config.get("seed", 200)
    overrides = config.get("test_overrides", {})
    ds = make_dataset(n_per_class * len(LABELS), seed=seed, base=reference_config(overrides))
    out = {"mode": "compare", "test_overrides": overrides, "n_test": int(len(ds.labels)), "methods": {}}
    preds = {name: method(ds) for name, method in methods.items()}
    for name, pred in preds.items():
        out["methods"][name] = summarize(ds.labels, pred)

    # Every headline ordering claim rests on these, not on the point estimates.
    out["pairwise_tests"] = _pairwise_tests(ds.labels, preds)

    # Fair-tuning artifact: how much does grid-search tuning buy the baseline, and
    # does a different classical estimator do better on the same features?
    if config.get("classical_default_check"):
        train = make_training_dataset(config.get("tune_pool", 1200), seed=config.get("tune_seed", 7))
        tuned = train_classical(train.images, train.labels, kind="rf", default=False)
        default = train_classical(train.images, train.labels, kind="rf", default=True)
        svm = train_classical(train.images, train.labels, kind="svm", default=False)
        tuning = {
            "tune_pool": config.get("tune_pool", 1200),
            "tune_seed": config.get("tune_seed", 7),
            "tuned_accuracy": accuracy(ds.labels, tuned.predict(ds.images)),
            "default_accuracy": accuracy(ds.labels, default.predict(ds.images)),
            "svm_accuracy": accuracy(ds.labels, svm.predict(ds.images)),
            "best_params": tuned.best_params,
            "svm_best_params": svm.best_params,
            "cv_score": tuned.cv_score,
            "svm_cv_score": svm.cv_score,
        }
        # Is the second classical estimator meaningfully different from the first?
        tuning["svm_vs_rf"] = mcnemar(ds.labels, svm.predict(ds.images), tuned.predict(ds.images))
        out["classical_tuning"] = tuning
    return out


def _run_scale_cue(config: dict, methods: dict[str, Method]) -> dict:
    """Measure how much accuracy comes from the class-correlated lattice parameter.

    Each structure carries one preset lattice parameter jittered only +-8%, and
    the small-cell and large-cell classes do not overlap, so absolute cell size
    is very nearly a class label. The camera-length randomisation hides it from
    the *pixel* scale, but the scattering envelope still encodes it in the
    ring-to-ring intensity fall-off. Scoring the same models on a test set whose
    lattice parameter is drawn from one common range isolates the difference.
    """
    n_per_class = config.get("n_per_class", 60)
    seed = config.get("seed", 500)
    overrides = config.get("test_overrides", {})
    a_range = tuple(config.get("a_range", [2.6, 6.1]))
    n = n_per_class * len(LABELS)

    standard = make_dataset(n, seed=seed, base=reference_config(overrides))
    decorrelated = make_scale_cue_dataset(n, seed=seed, base=reference_config(overrides), a_range=a_range)

    out: dict = {
        "mode": "scale_cue",
        "test_overrides": overrides,
        "a_range": list(a_range),
        "n_test": int(n),
        "methods": {},
    }
    dec_preds = {}
    for name, method in methods.items():
        pred_std = method(standard)
        pred_dec = method(decorrelated)
        dec_preds[name] = pred_dec
        out["methods"][name] = {
            "standard": summarize(standard.labels, pred_std),
            "decorrelated": summarize(decorrelated.labels, pred_dec),
            "accuracy_drop": accuracy(standard.labels, pred_std) - accuracy(decorrelated.labels, pred_dec),
        }
    # Do the method orderings survive once the cue is gone?
    out["pairwise_tests_decorrelated"] = _pairwise_tests(decorrelated.labels, dec_preds)
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
        base=reference_config({"blank_spots": True}),
    )
    blank_scores = {name: accuracy(blank.labels, method(blank)) for name, method in methods.items()}

    # 2. Trained-on-blank control: fit a fresh classical model on blanked data;
    # if it beats chance, the background leaks the class. The images and their
    # labels must come from the same draw: pairing images with labels from a
    # separate dataset would silently turn this into a label-shuffle control,
    # which also returns chance and would therefore look like a pass.
    blank_imgs = make_dataset(
        config.get("train_pool", 900),
        seed=seed + 2,
        base=reference_config({"blank_spots": True}),
    )
    trained = train_classical(blank_imgs.images, blank_imgs.labels, kind="rf", default=True)
    blank_test = make_dataset(
        n_per_class * len(LABELS),
        seed=seed + 3,
        base=reference_config({"blank_spots": True}),
    )
    trained_on_blank_acc = accuracy(blank_test.labels, trained.predict(blank_test.images))

    # 3. Label-shuffle control on real patterns: classical fit to shuffled labels.
    real_train = make_training_dataset(config.get("train_pool", 900), seed=seed + 4)
    rng = np.random.default_rng(seed + 5)
    shuffled = real_train.labels.copy()
    rng.shuffle(shuffled)
    shuffle_model = train_classical(real_train.images, shuffled, kind="rf", default=True)
    real_test = make_dataset(n_per_class * len(LABELS), seed=seed + 6, base=reference_config({}))
    shuffle_acc = accuracy(real_test.labels, shuffle_model.predict(real_test.images))

    return {
        "mode": "leakage",
        "chance_level": chance,
        "blank_spots_accuracy": blank_scores,
        "trained_on_blank_accuracy": trained_on_blank_acc,
        "shuffled_label_accuracy": shuffle_acc,
    }


def _run_ablation(config: dict) -> dict:
    """Retrain the pattern CNN with a randomisation component removed.

    What this measures: the component is removed from the *training* pool only,
    and the model is re-scored on the standard (still fully randomised) test set.
    That is a train-narrow / test-wide generalisation test, so it quantifies the
    robustness that domain randomisation buys.

    What this does **not** measure: whether a component is a shortcut. A shortcut
    would show up as a *gain* when the cue is held fixed in training and test
    together, which this design cannot produce. The shortcut tests in this repo
    are ``leakage`` (blank the physics) and ``scale_cue`` (sever the lattice
    parameter correlation).

    A single training run of this small CNN is seed-sensitive (an occasional run
    underfits, especially a fixed variant), so each variant is trained over
    ``n_seeds`` seeds and the mean and spread are reported. Reading one seed alone
    would turn an unlucky training run into a spurious "this cue is critical"
    conclusion.
    """
    n_per_class = config.get("n_per_class", 40)
    seed = config.get("seed", 400)
    epochs = config.get("epochs", 8)
    pool_size = config.get("pool_size", 3000)
    n_seeds = config.get("n_seeds", 3)
    variants = {
        "full": {},
        "fixed_scale": {"randomize_scale": False},
        "fixed_background": {"randomize_background": False},
    }
    test = make_dataset(n_per_class * len(LABELS), seed=seed, base=reference_config({}))
    hard = make_dataset(n_per_class * len(LABELS), seed=seed + 1, base=reference_config({"dose": 20.0}))
    results = {}
    for variant, kwargs in variants.items():
        test_accs, low_accs, train_accs = [], [], []
        for k in range(n_seeds):
            settings = TrainSettings(
                model="pattern", pool_size=pool_size, epochs=epochs, seed=seed + 100 * k, **kwargs
            )
            model, history = train_model(settings)
            test_accs.append(accuracy(test.labels, predict_pattern(model, test.images)))
            low_accs.append(accuracy(hard.labels, predict_pattern(model, hard.images)))
            train_accs.append(history["train_accuracy"])
        results[variant] = {
            "test_accuracy": float(np.mean(test_accs)),
            "test_accuracy_std": float(np.std(test_accs)),
            "low_dose_accuracy": float(np.mean(low_accs)),
            "low_dose_accuracy_std": float(np.std(low_accs)),
            "train_accuracy": float(np.mean(train_accs)),
            "n_seeds": n_seeds,
            "test_accuracy_per_seed": [float(x) for x in test_accs],
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
    if mode == "scale_cue":
        return _run_scale_cue(config, methods)
    raise ValueError(f"unknown benchmark mode {mode!r}")
