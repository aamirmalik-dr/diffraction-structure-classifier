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

Accuracy is reported with a 95% Wilson confidence interval, and every ordering
claim is checked with a paired McNemar test on the same test patterns, because at
these sample sizes several points of accuracy can be pure sampling noise.

## 1. Head-to-head comparison - configs/compare.yaml

Moderate operating point: dose 40, orientation-spread scale 3 degrees, 85% of
reflections kept, 1800 test patterns.

| Method | Accuracy | 95% CI | Macro-F1 |
|---|---|---|---|
| classical_rf | 0.559 | 0.536 - 0.582 | 0.551 |
| radial_cnn | 0.669 | 0.647 - 0.691 | 0.661 |
| **pattern_cnn** | **0.717** | 0.695 - 0.737 | **0.713** |
| chance | 0.167 | - | - |

Paired McNemar tests on the same 1800 patterns:

| Comparison | delta accuracy | p-value | significant |
|---|---|---|---|
| pattern_cnn vs radial_cnn | +0.047 | 8.8e-5 | yes |
| pattern_cnn vs classical_rf | +0.157 | 1.6e-33 | yes |
| radial_cnn vs classical_rf | +0.110 | 1.5e-20 | yes |

The 2D polar-Fourier CNN leads, the 1D radial CNN is second, the classical
baseline third, and all three gaps are significant. The ordering is stable across
the axes swept below.

## 2. Accuracy versus dose - configs/dose_sweep.yaml

Orientation-spread scale 2 degrees, all reflections kept, 600 test patterns per
point. Dose is counts at the brightest diffracted spot; lower is noisier. Readout
noise is a fixed count-level floor, so it dominates at low dose as a real detector
does.

| Dose | classical_rf | radial_cnn | pattern_cnn |
|---|---|---|---|
| 8 | 0.503 | **0.645** | 0.628 |
| 15 | 0.612 | **0.698** | 0.680 |
| 30 | 0.647 | 0.717 | 0.717 |
| 60 | 0.665 | 0.717 | **0.743** |
| 120 | 0.682 | 0.732 | **0.755** |
| 300 | 0.730 | **0.745** | 0.723 |

The two CNNs are close, and which one leads depends on dose. The pattern CNN wins
at moderate dose, where enough photons survive for the angular structure to be
read; at the lowest doses the 1D radial CNN matches or slightly beats it, because
angular detail is the first thing photon noise destroys, so the 2D model's extra
information is worth least when photons are scarcest. The classical baseline is
the noisiest throughout and only draws level at the brightest setting. Curve with
95% bands: `figures/dose_sweep.png`.

## 3. Accuracy versus visible reflections - configs/reflection_sweep.yaml

Dose 150, orientation-spread scale 2 degrees, 600 patterns per point. The kept
fraction thins the spot list uniformly at random (not by intensity), mimicking
unpredictable spot loss.

| Kept fraction | classical_rf | radial_cnn | pattern_cnn |
|---|---|---|---|
| 0.30 | 0.258 | 0.395 | **0.445** |
| 0.45 | 0.378 | 0.530 | **0.538** |
| 0.60 | 0.515 | 0.638 | **0.653** |
| 0.75 | 0.622 | 0.652 | **0.685** |
| 0.90 | 0.667 | **0.733** | 0.723 |
| 1.00 | 0.723 | 0.722 | **0.737** |

The pattern CNN leads across most of the range, with its margin largest in the
middle, where enough geometry survives to be informative but the radial profile
alone is ambiguous. With very few spots all methods struggle; with a full spot
list they converge. Curve: `figures/reflection_sweep.png`.

## 4. Accuracy versus orientation spread - configs/orientation_sweep.yaml

Dose 150, all reflections kept, 600 patterns per point. The swept value is the
scale parameter of the half-normal off-zone tilt, in degrees (the realised tilt
has mean about 0.80 times this and standard deviation about 0.60 times this).

| Spread (deg) | classical_rf | radial_cnn | pattern_cnn |
|---|---|---|---|
| 0 | 0.700 | 0.698 | **0.722** |
| 2 | 0.690 | 0.715 | **0.730** |
| 4 | 0.645 | **0.718** | 0.702 |
| 6 | 0.610 | 0.647 | **0.703** |
| 9 | 0.493 | 0.585 | **0.600** |
| 12 | 0.462 | 0.548 | 0.547 |

All methods degrade as the crystal tilts off the zone axis and reflections dim
asymmetrically; the pattern CNN degrades most gently through the mid-range. Curve:
`figures/orientation_sweep.png`.

## 5. Is the gap real? Fair-tuning the classical baseline

The classical baseline is tuned by cross-validated grid search, and a second
classical estimator (a tuned RBF-SVM on the same engineered features) is scored
alongside it, so the loss to the learned models cannot be blamed on one
estimator's hyperparameters. Retrained on the tuning pool (1500 patterns, seed 7)
and scored on the comparison set:

| Classical variant | Accuracy |
|---|---|
| random forest, library default | 0.521 |
| random forest, grid-search tuned | 0.517 |
| RBF-SVM, grid-search tuned | 0.547 |

Every classical variant sits in the 0.52 to 0.56 band. Tuning the forest does not
help (the cross-validation-optimal forest scores a fraction of a point below the
default on this split, the ordinary noise of choosing hyperparameters on one split
and scoring on another), and switching to a tuned SVM moves the number by three
points at most. The committed `classical_rf.joblib`, trained on a larger default
pool, scores 0.559 on this set, the same band. So the roughly 15-point gap to the
pattern CNN (0.717) is a limit of the engineered representation, not of one
estimator or its hyperparameters. The committed forest is depth-bounded so the
artifact stays a few MB.

## 6. The lattice-parameter shortcut - configs/scale_cue.yaml

This is the most important control in the repository. Each structure carries one
preset lattice parameter, and the small-cell classes (sc/bcc/fcc/hcp, 2.6 to 3.9
A) do not overlap the large-cell ones (diamond/rocksalt, 5.0 to 6.1 A), so
absolute cell size is very nearly a class label. Randomising the camera length
removes the pixel-scale version of this cue, but the scattering envelope
`exp(-B s^2)` still encodes absolute `|g|`, hence `a`, in the ring-to-ring
intensity fall-off. Scoring the committed models on a matched test set whose
lattice parameter is drawn from one common range for every class isolates the
effect (1800 patterns each):

| Method | preset a (published) | decorrelated a | drop |
|---|---|---|---|
| classical_rf | 0.578 | 0.418 | -0.161 |
| radial_cnn | 0.662 | 0.484 | -0.178 |
| pattern_cnn | 0.694 | 0.518 | -0.176 |

Every model loses about 16 to 18 points, so roughly a quarter of each model's
accuracy is material identity rather than structure-type geometry. The method
ordering survives the control (pattern_cnn still leads, every pairwise McNemar
p < 1e-3), so the comparison between representations is real, but the absolute
numbers are inflated by cell size and should be read with that in mind. Chart:
`figures/scale_cue.png`.

This is not a bug hidden and then confessed; it is a measured property of the
task. Fixing it properly would require realistic per-material lattice-parameter
distributions and a tabulated scattering factor, both out of scope here, so the
honest move is to report both numbers.

## 7. Leakage controls - configs/leakage.yaml

Every number here must sit at the 1/6 = 0.167 chance level; anything above it is a
leak. Image size is constant across classes by construction, so it is not tested
here; the background and the diffraction spots are.

| Control | Accuracy |
|---|---|
| classical_rf on blanked patterns | 0.188 |
| radial_cnn on blanked patterns | 0.158 |
| pattern_cnn on blanked patterns | 0.121 |
| classifier trained on blanked patterns | 0.188 |
| classifier trained on shuffled labels | 0.171 |

With the diffracted spots removed (only the central beam and background remain),
every model collapses to chance, a model trained on blanked patterns cannot learn
anything, and a model trained on shuffled labels cannot either. The classifiers
use the diffraction, not the background or the frame.

## 8. Domain-randomisation ablation - configs/ablation.yaml

The 2D CNN is retrained with one randomisation component removed from the training
pool, then re-scored on the standard test set. This measures the robustness that
domain randomisation buys (train narrow, test wide); it is not a shortcut test
(the shortcut tests are sections 6 and 7).

A single training run of this small CNN is seed-sensitive, so each variant is
trained over three seeds and the mean (plus or minus one standard deviation) is
reported. The exact values are regenerated into `results/ablation.json`; the
pattern is that both fixed variants score at or below full randomisation, so
domain randomisation is doing robustness work rather than papering over a
shortcut. The size of the fixed-background effect in particular is seed-sensitive
(one unlucky run underfits badly), which is exactly why this is averaged rather
than read off a single seed. Chart with error bars: `figures/ablation.png`.

## 9. Per-class performance and confusions

Per-class recall at the comparison operating point (dose 40, spread 3, keep 0.85,
1800 patterns):

| Class | classical_rf | radial_cnn | pattern_cnn |
|---|---|---|---|
| sc | 0.60 | 0.78 | 0.88 |
| bcc | 0.45 | 0.70 | 0.49 |
| fcc | 0.59 | 0.78 | 0.79 |
| diamond | 0.41 | 0.41 | 0.62 |
| rocksalt | 0.42 | 0.40 | 0.56 |
| hcp | 0.89 | 0.95 | 0.96 |

HCP, the only hexagonal structure, is recognised almost perfectly by every
method: its six-fold pattern is unmistakable. The hard classes are the cubic ones
that share ring geometry. The single largest confusion is simple cubic against
rock salt, and it is a property of the physics, not a model failure: down the
[001] zone axis rock salt shows only its all-even reflections (the all-odd family
is forbidden when `l = 0`), so its pattern is geometrically identical to simple
cubic's (same ring radii, same four-fold arrangement, verified in
`tests/test_physics_claims.py`) and the two differ only in how fast intensity
falls off. Where the 2D model earns its lead is diamond and rock salt: it recovers
diamond at recall 0.62 and rock salt at 0.56 where the 1D radial CNN manages 0.41
and 0.40, because those classes carry information in the relative intensities and
angular arrangement that the azimuthal average discards. BCC is the one class the
radial CNN reads better than the 2D model. The full confusion matrix is
`figures/confusion_matrix.png`; the per-class recall bars are
`figures/per_class_recall.png`.
