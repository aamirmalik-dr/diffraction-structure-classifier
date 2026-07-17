# Python API

`crystalclass` is a small library. This page shows the pieces you are most
likely to call directly, each with a runnable example. Everything is importable
from the submodules; the most common names are re-exported from the top level.

## Simulate one pattern

```python
import numpy as np
from crystalclass.sim import SimConfig, simulate

cfg = SimConfig(structure="diamond", dose=120, orientation_spread=3.0)
pattern = simulate(cfg, np.random.default_rng(0))

pattern.image      # (128, 128) noisy pattern in counts
pattern.profile    # (64,) azimuthally averaged radial profile
pattern.label      # "diamond"  (exact ground truth)
pattern.meta       # {'zone_axis': (1, 1, 0), 'a': 5.31, 'dose': 120, ...}
```

Leave any `SimConfig` field as `None` to have it drawn from a physical range;
that is how domain randomisation is applied. The structure, zone axis, lattice
parameter, camera-length scale, and background all randomise by default.

## Inspect the physics directly

```python
import numpy as np
from crystalclass.structures import get_structure, structure_factors

fcc = get_structure("fcc")
lattice = fcc.lattice_matrix()
structure_factors(fcc, np.array([(1, 1, 1)]), lattice)  # allowed, > 0
structure_factors(fcc, np.array([(1, 1, 0)]), lattice)  # extinct, ~ 0
```

Systematic absences are never hard coded. They fall out of the structure-factor
phase sum being zero, so the class labels and the allowed reflections come from
the same first-principles calculation.

## Build a seeded dataset

```python
from crystalclass.datasets import make_dataset, make_training_dataset

test = make_dataset(600, seed=0)              # class-balanced, one draw per class
train = make_training_dataset(3000, seed=0)   # full per-sample domain randomisation

train.images    # (3000, 128, 128) float32
train.profiles  # (3000, 64) float32
train.labels    # (3000,) int64
```

## The classical baseline

```python
from crystalclass.classical import train_classical
from crystalclass.metrics import accuracy

model = train_classical(train.images, train.labels, kind="rf")  # grid-search tuned
preds = model.predict(test.images)
accuracy(test.labels, preds)
model.best_params   # the tuned hyperparameters
```

Pass `default=True` to skip tuning and fit the library-default estimator; the
difference between the two is what the fair-tuning check in the benchmark
reports.

## The learned models

```python
from crystalclass.train import TrainSettings, train_model
from crystalclass.net import predict_pattern, predict_radial

model, history = train_model(TrainSettings(model="pattern"))  # defaults match the committed weights
preds = predict_pattern(model, test.images)

radial, _ = train_model(TrainSettings(model="radial"))
preds_1d = predict_radial(radial, test.profiles)
```

Both networks standardise their input after masking the direct beam and
log-compressing the dynamic range (`crystalclass.net.preprocess_image`).

## Score predictions

```python
from crystalclass.metrics import summarize

report = summarize(test.labels, preds)
report["accuracy"], report["macro_f1"]
report["per_class"]["diamond"]        # precision / recall / f1 / support
report["confusion_matrix"]            # nested list, true classes on rows
```

## Run a benchmark config

```python
import yaml
from crystalclass.benchmark import run_config

config = yaml.safe_load(open("configs/dose_sweep.yaml"))
result = run_config(config, models_dir=".")
```

The same thing from the command line, saving JSON:

```
crystalclass benchmark configs/dose_sweep.yaml --out results/dose_sweep.json
```

## Classify your own image

```python
from crystalclass.real import load_diffraction_image, classify_image
from crystalclass.train import load_pattern_model

model = load_pattern_model("models/pattern_cnn.pt")
image = load_diffraction_image("your_pattern.png", size=128)
index, probs = classify_image(model, image)
```

See `crystalclass/real.py` for the requirements the input image must meet
(centred direct beam, whole pattern visible, actually a diffraction pattern).
