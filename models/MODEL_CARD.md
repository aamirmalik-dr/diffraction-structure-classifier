# Model card: crystalclass classifiers

## What these files are

| File | Model | Input | Size |
|---|---|---|---|
| `pattern_cnn.pt` | 2D CNN over the rotation-invariant polar-Fourier map | full diffraction pattern | ~0.4 MB |
| `radial_cnn.pt` | 1D CNN over the radial profile | azimuthal average | ~0.14 MB |
| `classical_rf.joblib` | grid-search-tuned random forest over engineered features | ring ratios + spot geometry | ~5.9 MB (compressed) |

All three predict one of six structure types: `sc`, `bcc`, `fcc`, `diamond`,
`rocksalt`, `hcp`.

## Representations

The three models are deliberately given three different views of the same
pattern so the comparison is about representation, not luck:

- **Classical**: scale- and rotation-invariant features (a radial profile
  resampled onto `r / r1`, ring-radius ratios and relative heights, and the
  multiplicity and angular regularity of the inner spot ring), fed to a random
  forest whose hyperparameters are chosen by cross-validated grid search.
- **Radial CNN**: a small 1D convnet over the raw azimuthally averaged radial
  profile. Rotation invariant by construction, but blind to angular geometry.
- **Pattern CNN**: a small 2D convnet over the polar-Fourier map. The pattern is
  resampled to polar coordinates and the FFT magnitude is taken along the
  angular axis, so an in-plane rotation (a cyclic shift in angle) leaves the map
  unchanged. The map keeps both the ring radii and the angular symmetry order,
  four-fold versus six-fold, so this model can use the geometry the radial
  profile discards.

## Architectures

- **Pattern CNN**: three Conv-BatchNorm-ReLU-MaxPool blocks (16, 32, 48
  channels) over the 48x24 polar-Fourier map, then a 96-unit dense layer to six
  logits. About 100k parameters.
- **Radial CNN**: three Conv-BatchNorm-ReLU-MaxPool blocks (16, 32, 48 channels)
  over the length-64 profile, then a 64-unit dense layer. About 40k parameters.

Both standardise their input after masking the direct beam (5 px) and applying a
`log1p` compression so faint reflections are not swamped by the bright rings.

## Training regime

Both CNNs train for a few minutes on CPU on a domain-randomised pool of 3000
simulated patterns (seed 0), 12 to 20 epochs, Adam at 1e-3, batch 32. Each
pattern draws an independent structure, zone axis (from the principal low-index
set), lattice parameter (+-8%), dose (log-uniform 8 to 400 counts), orientation
spread (0 to 6 degrees), missing-reflection fraction (down to 0.75), camera-length
scale, and background. Reproduce with:

```
crystalclass train --model pattern
crystalclass train --model radial
crystalclass train --model classical
```

The random forest is tuned by 3-fold grid search over the number of trees, depth,
and minimum leaf size, on features extracted from the same pool.

## Measured performance

All measured this session with fixed seeds; see `RESULTS.md` and
`results/metrics.json`. At a moderate operating point (dose 40, 3 degrees tilt,
85% of reflections), accuracy is 0.711 for the pattern CNN, 0.658 for the radial
CNN, and 0.556 for the tuned classical baseline, against a chance level of 0.167.
The pattern CNN leads at every dose, tilt, and reflection count swept, holds
accuracy 0.70 to 0.80 across a 40x dose range, and recovers rock salt at recall
0.65 versus the radial CNN's 0.38. HCP is near-perfect for every method (recall
0.93 to 1.00); diamond is the hardest. The confusion matrix, the
accuracy-versus-dose curve, and the per-class recalls are in `figures/`.

## Integrity checks

- **Leakage control** (`configs/leakage.yaml`): with the diffracted spots
  blanked out (only the central beam and background remain), every model sits at
  the 1/6 chance level, and a model trained on blanked patterns cannot beat
  chance. A label-shuffle control also collapses to chance. The models use the
  diffraction, not the background or the image size (which is constant).
- **Ablation** (`configs/ablation.yaml`): retraining the pattern CNN with the
  camera-length scale or the background held fixed does not inflate accuracy,
  confirming neither is a trivial cue.

## Intended use and limitations

Intended: identifying the structure type of a simulated single-crystal
zone-axis electron diffraction pattern within the modelling assumptions below,
and serving as a reproducible baseline for representation comparisons on this
task.

Not intended: quantitative phase identification on experimental data without
validation, patterns from zone axes far outside the principal low-index set,
multi-phase or textured samples, or convergent-beam diffraction.

Modelling assumptions and domain gap: the simulator is **kinematical** (single
scattering) with a Gaussian spot shape, a first-order excitation-error model for
off-zone tilt, and Poisson noise. Real patterns add dynamical (multiple)
scattering, which redistributes intensity and can make forbidden reflections
appear, plus higher-order Laue zones, inelastic background, and detector effects.
The models were trained purely on simulation and have never seen a real pattern.
Expect the domain gap to be real; treat any prediction on experimental data as a
hypothesis to check against a trusted reference.
