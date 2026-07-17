# Model card: crystalclass classifiers

## What these files are

| File | Model | Input | Size |
|---|---|---|---|
| `pattern_cnn.pt` | 2D CNN over the rotation-invariant polar-Fourier map | full diffraction pattern | ~0.4 MB |
| `radial_cnn.pt` | 1D CNN over the radial profile | azimuthal average | ~0.14 MB |
| `classical_rf.joblib` | grid-search-tuned random forest over engineered features | ring ratios + spot geometry | ~5.2 MB (compressed) |

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
  logits. 102,486 parameters.
- **Radial CNN**: three Conv-BatchNorm-ReLU-MaxPool blocks (16, 32, 48 channels)
  over the length-64 profile, then a 64-unit dense layer. 32,598 parameters.

Both standardise their input after masking the direct beam (5 px) and applying a
`log1p` compression so faint reflections are not swamped by the bright rings.
(The classical baseline does not mask the beam: its resampled radial profile
starts at `r = 0`. This is harmless, because the beam is rendered at the same
amplitude for every class and so carries no class information, but it is a real
difference between the pipelines.)

## Training regime

Both CNNs train for a few minutes on CPU on a domain-randomised pool of 3000
simulated patterns (seed 0), 18 epochs, Adam at 1e-3, batch 32. Each
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

All measured this session with fixed seeds on 1800 test patterns; see `RESULTS.md`
and `results/metrics.json`. At a moderate operating point (dose 40, tilt scale 3
degrees, 85% of reflections), accuracy is 0.717 (95% CI 0.695 to 0.737) for the
pattern CNN, 0.669 for the radial CNN, and 0.559 for the tuned classical baseline,
against a chance level of 0.167. Every pairwise difference is significant by a
paired McNemar test (p < 1e-4). The pattern CNN leads across the swept axes, but
the two CNNs are close: at the lowest doses the 1D radial CNN matches or slightly
beats the 2D one, since angular detail is the first thing photon noise destroys.
The 2D model's clearest per-class wins are diamond (recall 0.62 versus the radial
CNN's 0.41) and rock salt (0.56 versus 0.40). HCP is near-perfect for every method
(recall 0.89 to 0.96); BCC and diamond are the hardest. The confusion matrix, the
accuracy-versus-dose curve, and the per-class recalls are in `figures/`.

**Caveat, measured and reported (`configs/scale_cue.yaml`):** each structure
carries a preset lattice parameter, and the small-cell and large-cell classes do
not overlap, so absolute cell size is a partial class label that the scattering
envelope leaks through the ring intensities. Decorrelating the lattice parameter
from the class costs every model about 16 to 18 accuracy points (pattern CNN 0.694
to 0.518), so roughly a quarter of the accuracy above is material identity rather
than structure-type geometry. The method ordering survives the control.

## Integrity checks

- **Lattice-parameter shortcut** (`configs/scale_cue.yaml`): quantifies the cell-
  size cue above; the ordering between methods survives decorrelation even though
  the absolute numbers fall.
- **Leakage control** (`configs/leakage.yaml`): with the diffracted spots
  blanked out (only the central beam and background remain), every model sits at
  the 1/6 chance level, a model trained on blanked patterns cannot beat chance,
  and a label-shuffle control also collapses to chance. The models use the
  diffraction, not the background or the image size (which is constant).
- **Ablation** (`configs/ablation.yaml`): retraining the pattern CNN with the
  camera-length scale or the background held fixed in training degrades accuracy
  rather than inflating it (averaged over three seeds, since a single run is
  seed-sensitive), so neither randomisation is hiding a shortcut. This measures
  the robustness value of domain randomisation, not the presence of a shortcut.

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
off-zone tilt, Poisson photon noise, and a dose-independent Gaussian readout
floor. There is no electron wavelength, so the Ewald sphere is flat and there are
no higher-order Laue zones, and the atomic scattering factor is a single-Gaussian
approximation that models relative intensities only qualitatively. Real patterns
add dynamical (multiple) scattering, which redistributes intensity and can make
forbidden reflections appear, plus higher-order Laue zones, inelastic background,
and detector effects. Absolute cell size is also a partial class label here (see
the scale-cue caveat above), which it would not be for an arbitrary real sample.
The models were trained purely on simulation and have never seen a real pattern.
Expect the domain gap to be real; treat any prediction on experimental data as a
hypothesis to check against a trusted reference.
