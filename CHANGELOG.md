# Changelog

## 0.1.1 (2026-09-26)

- Package metadata completed: keywords, classifiers, and project URLs in `pyproject.toml`.
- Citation file (`CITATION.cff`) and this changelog added.
- CI badge added to the README.
- Related repositories section in the README linking the six sibling electron-microscopy repositories.
- Test guarding the package `__version__` against the installed distribution metadata.

## 0.1.0 (2026-07-17)

- Kinematical zone-axis diffraction simulator for six structure types with structure factors, excitation-error dimming, missing reflections, Gaussian spots, a randomised background, Poisson noise, and a dose-independent readout floor; every pattern carries its exact label and a radial profile.
- Three classifiers on three views of the same pattern: a random forest and an RBF-SVM tuned by cross-validated grid search on scale- and rotation-invariant features, a 1D CNN on the radial profile, and a 2D CNN on a rotation-invariant polar-Fourier map.
- Accuracy with 95% Wilson intervals, macro-F1, confusion matrices, and paired McNemar tests for every ordering claim.
- Controls: a lattice-parameter shortcut control that decorrelates cell size from the class, a blank-pattern and label-shuffle leakage control, and a seed-averaged domain-randomisation ablation; dose, visible-reflection, and orientation-spread sweeps.
- The `crystalclass` CLI, committed CNN and random-forest artifacts, sample patterns, results JSON, figures including the zone-axis gallery and shortcut chart, model card, API docs, executed tutorial, and a CI workflow.
- Maintenance after publication: ruff and black pinned to exact versions, README expanded and restructured, pattern gallery re-exported at higher resolution.
