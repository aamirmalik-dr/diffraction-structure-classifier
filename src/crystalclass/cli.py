"""Command-line interface for crystalclass.

Subcommands:
    crystalclass simulate    simulate one pattern to .npz and/or .png
    crystalclass classify    classify an .npz sample or a real image
    crystalclass train       train the pattern CNN, radial CNN, or classical model
    crystalclass benchmark   run a YAML benchmark config, save JSON (+ figure)
    crystalclass samples     regenerate the committed sample patterns
    crystalclass gallery     save a labelled gallery of the six structures
    crystalclass demo        classify every committed sample with the pattern CNN
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")  # headless: the CLI only writes figure files, never shows them

from crystalclass.benchmark import run_config
from crystalclass.datasets import make_training_dataset
from crystalclass.io import load_pattern, save_pattern
from crystalclass.net import predict_pattern, predict_radial
from crystalclass.plots import plot_pattern_gallery
from crystalclass.sim import LABELS, SimConfig, simulate

DEFAULT_MODELS = {
    "pattern": "models/pattern_cnn.pt",
    "radial": "models/radial_cnn.pt",
    "classical": "models/classical_rf.joblib",
}

# (name, structure, dose, seed): the committed reference samples.
SAMPLES = (
    ("fcc_cu", "fcc", 120.0, 0),
    ("bcc_fe", "bcc", 120.0, 1),
    ("diamond_si", "diamond", 120.0, 2),
    ("rocksalt_nacl", "rocksalt", 120.0, 3),
    ("hcp_ti", "hcp", 120.0, 4),
    ("sc_po", "sc", 120.0, 5),
)


def _cmd_simulate(args: argparse.Namespace) -> None:
    zone = tuple(int(v) for v in args.zone.split(",")) if args.zone else None
    cfg = SimConfig(
        structure=args.structure,
        dose=args.dose,
        orientation_spread=args.orientation,
        keep_fraction=args.keep,
        zone_axis=zone,
    )
    pattern = simulate(cfg, np.random.default_rng(args.seed))
    print(f"{pattern.label}  zone {pattern.meta['zone_axis']}  {pattern.meta['n_spots']} spots")
    if args.out:
        save_pattern(args.out, pattern)
        print(f"wrote {args.out}")
    if args.figure:
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
        a1.imshow(pattern.image, cmap="magma", vmax=np.percentile(pattern.image, 99.5))
        a1.set_title(f"{pattern.label} pattern")
        a1.axis("off")
        a2.plot(pattern.profile, color="#4c72b0")
        a2.set_title("radial profile")
        a2.set_xlabel("radius (px)")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=130)
        print(f"wrote {args.figure}")


def _load_input(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        return load_pattern(path).image
    from crystalclass.real import load_diffraction_image

    return load_diffraction_image(path)


def _cmd_classify(args: argparse.Namespace) -> None:
    image = _load_input(args.input)
    if args.method == "classical":
        model = joblib.load(args.model or DEFAULT_MODELS["classical"])
        pred = int(model.predict(image[None])[0])
    elif args.method == "radial":
        from crystalclass.sim import _radial_profile
        from crystalclass.train import load_radial_model

        prof = _radial_profile(image)
        model = load_radial_model(args.model or DEFAULT_MODELS["radial"], length=len(prof))
        pred = int(predict_radial(model, prof[None])[0])
    else:
        from crystalclass.train import load_pattern_model

        model = load_pattern_model(args.model or DEFAULT_MODELS["pattern"])
        pred = int(predict_pattern(model, image[None])[0])
    print(f"predicted structure: {LABELS[pred]}")


def _cmd_train(args: argparse.Namespace) -> None:
    out = Path(args.out or DEFAULT_MODELS[args.model])
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.model == "classical":
        from crystalclass.classical import train_classical

        pool = make_training_dataset(args.pool, seed=args.seed)
        model = train_classical(pool.images, pool.labels, kind=args.kind, default=False)
        joblib.dump(model, out, compress=3)
        print(f"trained classical {args.kind}: cv={model.cv_score:.4f}  params={model.best_params}")
        print(f"wrote {out}")
        return
    from crystalclass.train import TrainSettings, save_model, train_model

    settings = TrainSettings(model=args.model, pool_size=args.pool, epochs=args.epochs, seed=args.seed)
    model, history = train_model(settings, verbose=True)
    save_model(model, str(out))
    print(f"train accuracy {history['train_accuracy']:.4f}")
    print(f"wrote {out}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text())
    result = run_config(config, models_dir=args.models_dir)
    out = Path(args.out or f"results/{Path(args.config).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    print(json.dumps(result, indent=2)[:800])


def _cmd_samples(args: argparse.Namespace) -> None:
    for name, structure, dose, seed in SAMPLES:
        cfg = SimConfig(structure=structure, dose=dose)
        pattern = simulate(cfg, np.random.default_rng(seed))
        path = Path("data/sample") / f"{name}.npz"
        save_pattern(path, pattern)
        print(f"wrote {path}  ({pattern.label}, {pattern.meta['n_spots']} spots)")


def _cmd_gallery(args: argparse.Namespace) -> None:
    patterns = []
    for name, structure, dose, seed in SAMPLES:
        cfg = SimConfig(structure=structure, dose=dose)
        patterns.append(simulate(cfg, np.random.default_rng(seed)))
    out = args.out or "figures/pattern_gallery.png"
    plot_pattern_gallery(patterns, out, title="Simulated zone-axis diffraction, one per structure")
    print(f"wrote {out}")


def _cmd_demo(args: argparse.Namespace) -> None:
    from crystalclass.train import load_pattern_model

    model = load_pattern_model(args.model or DEFAULT_MODELS["pattern"])
    correct = 0
    for name, structure, dose, seed in SAMPLES:
        pattern = load_pattern(Path("data/sample") / f"{name}.npz")
        pred = LABELS[int(predict_pattern(model, pattern.image[None])[0])]
        ok = pred == pattern.label
        correct += ok
        print(f"{name:16s} true={pattern.label:9s} pred={pred:9s} {'OK' if ok else 'X'}")
    print(f"{correct}/{len(SAMPLES)} correct")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    p = argparse.ArgumentParser(prog="crystalclass", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="simulate one pattern")
    s.add_argument("--structure", choices=LABELS, default=None)
    s.add_argument("--dose", type=float, default=150.0)
    s.add_argument("--orientation", type=float, default=2.0)
    s.add_argument("--keep", type=float, default=1.0)
    s.add_argument("--zone", type=str, default=None, help="comma-separated, e.g. 0,0,1")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=str, default=None)
    s.add_argument("--figure", type=str, default=None)
    s.set_defaults(func=_cmd_simulate)

    c = sub.add_parser("classify", help="classify a sample or real image")
    c.add_argument("input")
    c.add_argument("--method", choices=["pattern", "radial", "classical"], default="pattern")
    c.add_argument("--model", type=str, default=None)
    c.set_defaults(func=_cmd_classify)

    t = sub.add_parser("train", help="train a model")
    t.add_argument("--model", choices=["pattern", "radial", "classical"], default="pattern")
    t.add_argument("--kind", choices=["rf", "svm"], default="rf", help="classical estimator")
    t.add_argument("--pool", type=int, default=3000)
    t.add_argument("--epochs", type=int, default=18)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=str, default=None)
    t.set_defaults(func=_cmd_train)

    b = sub.add_parser("benchmark", help="run a YAML benchmark config")
    b.add_argument("config")
    b.add_argument("--out", type=str, default=None)
    b.add_argument("--models-dir", type=str, default=".")
    b.set_defaults(func=_cmd_benchmark)

    sub.add_parser("samples", help="regenerate committed samples").set_defaults(func=_cmd_samples)

    g = sub.add_parser("gallery", help="save the structure gallery figure")
    g.add_argument("--out", type=str, default=None)
    g.set_defaults(func=_cmd_gallery)

    d = sub.add_parser("demo", help="classify committed samples with the pattern CNN")
    d.add_argument("--model", type=str, default=None)
    d.set_defaults(func=_cmd_demo)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
