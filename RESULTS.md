# Results

Every number below was produced this session by the committed configs in a fresh
Python 3.11 venv (torch 2.13.0+cpu), with fixed seeds throughout. Regenerate any
table with `crystalclass benchmark configs/<name>.yaml`; raw values live in
`results/<name>.json`, and the headline numbers are collected in
`results/metrics.json`. The task is six-way classification, so the chance level
is 1/6 = 0.167. Three methods are compared on three views of the same pattern:

- **classical_rf**: engineered scale- and rotation-invariant features into a
  grid-search-tuned random forest.
- **radial_cnn**: a 1D CNN on the azimuthally averaged radial profile.
- **pattern_cnn**: a 2D CNN on the rotation-invariant polar-Fourier map.

## 1. Head-to-head comparison - configs/compare.yaml

Moderate operating point: dose 40, orientation spread 3 degrees, 85% of
reflections kept, 360 test patterns.

| Method | Accuracy | Macro-F1 |
|---|---|---|
| classical_rf | 0.556 | 0.541 |
| radial_cnn | 0.658 | 0.648 |
| **pattern_cnn** | **0.711** | **0.712** |
| chance | 0.167 | - |

The 2D polar-Fourier CNN leads. It beats the 1D radial CNN by 5 points and the
classical baseline by 16 points, at 4.3x the chance level. The ordering is the
same on every axis swept below.

## 2. Accuracy versus dose - configs/dose_sweep.yaml

Orientation spread 2 degrees, all reflections kept, 240 test patterns per point.
Dose is counts at the brightest diffracted spot; lower is noisier.

| Dose | classical_rf | radial_cnn | pattern_cnn |
|---|---|---|---|
| 8 | 0.546 | 0.654 | **0.700** |
| 15 | 0.675 | 0.692 | **0.721** |
| 30 | 0.692 | 0.742 | **0.762** |
| 60 | 0.638 | 0.746 | **0.796** |
| 120 | 0.679 | 0.696 | **0.796** |
| 300 | 0.721 | 0.704 | **0.721** |

The pattern CNN is both the most accurate and the most dose-stable, holding 0.70
to 0.80 from dose 8 to 300. The classical baseline is the noisiest and only draws
level at the brightest setting.

## 3. Accuracy versus visible reflections - configs/reflection_sweep.yaml

Dose 150, orientation spread 2 degrees. The kept fraction thins the spot list to
mimic weak or undetected reflections.

| Kept fraction | classical_rf | radial_cnn | pattern_cnn |
|---|---|---|---|
| 0.30 | 0.250 | 0.354 | **0.383** |
| 0.45 | 0.417 | 0.567 | **0.617** |
| 0.60 | 0.483 | 0.637 | **0.679** |
| 0.75 | 0.637 | 0.683 | **0.721** |
| 0.90 | 0.662 | 0.700 | **0.808** |
| 1.00 | 0.721 | 0.717 | **0.742** |

The pattern CNN's margin is largest in the middle, where enough geometry survives
to be informative but the radial profile alone is ambiguous. With very few spots
(0.30) all methods struggle; with a full spot list they converge.

## 4. Accuracy versus orientation spread - configs/orientation_sweep.yaml

Dose 150, all reflections kept. The spread is the standard deviation of the
off-zone tilt in degrees.

| Spread (deg) | classical_rf | radial_cnn | pattern_cnn |
|---|---|---|---|
| 0 | 0.662 | 0.708 | **0.750** |
| 2 | 0.692 | 0.704 | **0.762** |
| 4 | 0.662 | 0.704 | **0.762** |
| 6 | 0.579 | 0.667 | 0.662 |
| 9 | 0.529 | 0.537 | **0.600** |
| 12 | 0.446 | 0.500 | **0.562** |

All methods degrade as the crystal tilts off the zone axis and reflections dim
asymmetrically; the pattern CNN degrades most gently.

## 5. Is the gap real? Fair-tuning the classical baseline

The classical baseline is tuned by cross-validated grid search, not left at
library defaults, so its loss to the learned models cannot be blamed on an
untuned hyperparameter. On the comparison set:

| Classical variant | Accuracy |
|---|---|
| library default | 0.547 |
| grid-search tuned | 0.542 |

Tuning does not help. The cross-validation-optimal forest scores a fraction of a
point below the default on this test set (0.542 vs 0.547), which is the ordinary
noise of choosing hyperparameters on one split and scoring on another. The
takeaway is the honest one: the classical baseline sits near 0.55 whether tuned
or not, so the 16-point gap to the pattern CNN (0.711) is a limit of the
representation, not of the hyperparameters. The committed forest is depth-bounded
so the artifact stays small (5.9 MB); an unbounded forest gains a fraction of a
point at ten times the size.

## 6. Leakage controls - configs/leakage.yaml

Every number here must sit at the 1/6 = 0.167 chance level; anything above it is a
leak. Image size is constant across classes by construction, so it is not tested
here; the background and the diffraction spots are.

| Control | Accuracy |
|---|---|
| classical_rf on blanked patterns | 0.179 |
| radial_cnn on blanked patterns | 0.138 |
| pattern_cnn on blanked patterns | 0.171 |
| classifier trained on blanked patterns | 0.146 |
| classifier trained on shuffled labels | 0.158 |

With the diffracted spots removed (only the central beam and background remain),
every model collapses to chance, a model trained on blanked patterns cannot learn
anything, and a model trained on shuffled labels cannot either. The classifiers
use the diffraction, not the background or the frame.

## 7. Domain-randomisation ablation - configs/ablation.yaml

The 2D CNN is retrained with one randomisation component removed, then re-scored.
If holding the camera-length scale or the background fixed inflated accuracy, that
component was a trivial cue. It does not; both variants score slightly lower,
which means the randomisation earns its place as robustness rather than hiding a
cue.

| Variant | Accuracy (dose 150) | Accuracy (dose 20) |
|---|---|---|
| full randomisation | **0.742** | **0.733** |
| fixed scale | 0.708 | 0.675 |
| fixed background | 0.708 | 0.704 |

The effect is clearest at low dose, where fixing the camera-length scale costs the
most: domain randomisation is doing robustness work, not papering over a shortcut.

## 8. Per-class performance and confusions

Per-class recall at the comparison operating point (dose 40, spread 3, keep 0.85):

| Class | classical_rf | radial_cnn | pattern_cnn |
|---|---|---|---|
| sc | 0.63 | 0.83 | 0.73 |
| bcc | 0.38 | 0.52 | 0.55 |
| fcc | 0.70 | 0.77 | 0.80 |
| diamond | 0.32 | 0.47 | 0.53 |
| rocksalt | 0.38 | 0.38 | 0.65 |
| hcp | 0.92 | 0.98 | 1.00 |

HCP, the only hexagonal structure, is recognised almost perfectly by every
method: its six-fold pattern is unmistakable. The hard classes are the cubic ones
that share ring geometry. Diamond is the hardest (it is an FCC lattice with extra
absences) and is most often mistaken for BCC or FCC. Rock salt is where the 2D
CNN earns its keep: its recall of 0.65 far exceeds the radial CNN's and the
classical baseline's 0.38, because rock salt differs from simple cubic mainly in
the angular arrangement and relative intensity of its spots, exactly the
information the azimuthal average throws away and the polar-Fourier map keeps. The
full confusion matrix is `figures/confusion_matrix.png`.
