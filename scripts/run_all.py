"""End-to-end reproduction: samples, models, benchmarks, metrics, figures.

    python scripts/run_all.py

Regenerates the committed samples and model weights, runs every benchmark
config, aggregates the headline numbers into ``results/metrics.json``, and draws
the figures. Fixed seeds throughout, so the outputs match the committed ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"


def _run(args: list[str]) -> None:
    print(">>", " ".join(args))
    subprocess.run([sys.executable, "-m", "crystalclass.cli", *args], cwd=ROOT, check=True)


def _benchmark(name: str) -> None:
    from crystalclass.benchmark import run_config

    config = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    result = run_config(config, models_dir=ROOT)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(f"   wrote results/{name}.json")


def aggregate_metrics() -> None:
    """Collect the headline numbers from every benchmark into metrics.json."""

    def load(name: str) -> dict:
        return json.loads((RESULTS / f"{name}.json").read_text())

    compare = load("compare")
    dose = load("dose_sweep")
    leakage = load("leakage")
    ablation = load("ablation")
    scale_cue = load("scale_cue")

    methods = ["classical_rf", "radial_cnn", "pattern_cnn"]
    dose_curve = {m: {str(r["value"]): round(r[m]["accuracy"], 4) for r in dose["rows"]} for m in methods}
    metrics = {
        "task": "6-way crystal-structure classification from simulated electron diffraction",
        "labels": compare["methods"]["classical_rf"]["labels"],
        "chance_level": round(1.0 / 6.0, 4),
        "compare_conditions": compare["test_overrides"],
        "compare_accuracy": {m: round(compare["methods"][m]["accuracy"], 4) for m in methods},
        "compare_macro_f1": {m: round(compare["methods"][m]["macro_f1"], 4) for m in methods},
        "n_test_compare": compare["n_test"],
        "compare_accuracy_ci95": {
            m: [round(v, 4) for v in compare["methods"][m]["accuracy_ci95"]] for m in methods
        },
        "compare_pairwise_tests": compare["pairwise_tests"],
        "classical_tuning": compare.get("classical_tuning", {}),
        "scale_cue": {
            "a_range": scale_cue["a_range"],
            "n_test": scale_cue["n_test"],
            "accuracy_standard": {
                m: round(scale_cue["methods"][m]["standard"]["accuracy"], 4) for m in methods
            },
            "accuracy_decorrelated": {
                m: round(scale_cue["methods"][m]["decorrelated"]["accuracy"], 4) for m in methods
            },
            "accuracy_drop": {m: round(scale_cue["methods"][m]["accuracy_drop"], 4) for m in methods},
            "pairwise_tests_decorrelated": scale_cue["pairwise_tests_decorrelated"],
        },
        "dose_accuracy": dose_curve,
        "leakage": {
            "chance_level": leakage["chance_level"],
            "blank_spots_accuracy": {k: round(v, 4) for k, v in leakage["blank_spots_accuracy"].items()},
            "trained_on_blank_accuracy": round(leakage["trained_on_blank_accuracy"], 4),
            "shuffled_label_accuracy": round(leakage["shuffled_label_accuracy"], 4),
        },
        "ablation": {
            v: {k: (round(x, 4) if isinstance(x, (int, float)) else x) for k, x in d.items()}
            for v, d in ablation["variants"].items()
        },
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("   wrote results/metrics.json")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    _run(["samples"])
    _run(["train", "--model", "classical"])
    _run(["train", "--model", "radial"])
    _run(["train", "--model", "pattern"])
    for name in [
        "dose_sweep",
        "reflection_sweep",
        "orientation_sweep",
        "compare",
        "scale_cue",
        "leakage",
        "ablation",
    ]:
        _benchmark(name)
    aggregate_metrics()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py")], check=True)
    print("done")


if __name__ == "__main__":
    main()
