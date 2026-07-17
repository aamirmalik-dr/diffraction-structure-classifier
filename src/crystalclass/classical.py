"""Classical baseline: engineered features into a fair-tuned classifier.

Two estimators are offered, a random forest and an RBF support vector machine.
Both are tuned by grid search with cross-validation on the training features, so
the comparison against the learned models is against a *tuned* baseline, not a
default one. This matters: an untuned classical method can lose to a network for
reasons that have nothing to do with the network being better, and this project
credits the gap only after the baseline has had its fair chance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from crystalclass.features import extract_features_batch

_GRIDS: dict[str, tuple[Pipeline, dict]] = {
    "rf": (
        Pipeline([("clf", RandomForestClassifier(random_state=0, n_jobs=-1))]),
        {
            # Depth is bounded so the committed forest stays small (a few MB); an
            # unbounded forest on this pool balloons past the size budget for a
            # fraction of a point of accuracy.
            "clf__n_estimators": [150, 250],
            "clf__max_depth": [10, 16],
            "clf__min_samples_leaf": [2, 4],
        },
    ),
    "svm": (
        Pipeline([("scaler", StandardScaler()), ("clf", SVC(random_state=0))]),
        {
            "clf__C": [1.0, 10.0, 100.0],
            "clf__gamma": ["scale", 0.01, 0.1],
        },
    ),
}


@dataclass
class ClassicalModel:
    """A fitted classical classifier over engineered features."""

    kind: str
    estimator: Pipeline
    best_params: dict
    cv_score: float

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict class indices for a stack of images."""
        return self.estimator.predict(extract_features_batch(images))


def train_classical(
    images: np.ndarray,
    labels: np.ndarray,
    kind: str = "rf",
    cv: int = 3,
    default: bool = False,
) -> ClassicalModel:
    """Fit and fair-tune a classical classifier on engineered features.

    Args:
        images: Training image stack.
        labels: Integer class labels.
        kind: ``"rf"`` or ``"svm"``.
        cv: Cross-validation folds for the grid search.
        default: If True, skip tuning and fit the library-default estimator.
            Used to quantify how much the fair tuning is worth.

    Returns:
        A fitted :class:`ClassicalModel`.
    """
    if kind not in _GRIDS:
        raise ValueError(f"unknown classical kind {kind!r}; choose from {list(_GRIDS)}")
    feats = extract_features_batch(images)
    pipeline, grid = _GRIDS[kind]
    if default:
        pipeline.fit(feats, labels)
        return ClassicalModel(kind=kind, estimator=pipeline, best_params={}, cv_score=float("nan"))
    search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=-1, scoring="accuracy")
    search.fit(feats, labels)
    return ClassicalModel(
        kind=kind,
        estimator=search.best_estimator_,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
    )
