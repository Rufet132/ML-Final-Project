"""Preprocessing utilities used by the experiment scripts.

Covers exactly what the project brief asks for: feature standardization
fitted on training data only, simple missing-value handling, one-hot
encoding of categorical columns (so the continuous-only DecisionTree can
consume mixed datasets such as Adult Income), and random oversampling as
the documented treatment for severely imbalanced datasets.
"""

from __future__ import annotations

import numpy as np


class StandardScaler:
    """Zero-mean unit-variance scaler (fit on train, transform anywhere).

    Constant columns get a standard deviation of 1 so they transform to
    exactly zero instead of dividing by zero.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Learn per-feature mean and standard deviation from ``X``."""
        X = _validate_matrix(X)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize ``X`` with the statistics learned in :meth:`fit`."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must be fitted before transform")
        X = _validate_matrix(X)
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("X has a different number of features than the fitted data")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on ``X`` and return its standardized version."""
        return self.fit(X).transform(X)


def drop_missing_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove samples containing NaN features.

    Row removal (rather than imputation) is the documented choice for
    this project: the affected datasets lose well under 10% of rows, and
    dropping keeps the feature distributions untouched for the
    bias-variance analysis.

    Args:
        X: Feature matrix ``(n_samples, n_features)``; may contain NaN.
        y: Labels ``(n_samples,)``.

    Returns:
        The ``(X, y)`` pair restricted to fully observed rows.
    """
    X = _validate_matrix(X, allow_nan=True)
    y = np.asarray(y)
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    keep = ~np.isnan(X).any(axis=1)
    return X[keep], y[keep]


def one_hot_encode(columns: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Turn named categorical columns into 0/1 indicator features.

    Args:
        columns: Mapping of column name to a 1-D array of category
            values (any dtype); all columns must share one length.

    Returns:
        Tuple of the stacked indicator matrix (float, one column per
        distinct category) and the generated feature names in
        ``"column=value"`` form.
    """
    if not columns:
        raise ValueError("at least one column is required")
    lengths = {values.shape[0] for values in map(np.asarray, columns.values())}
    if len(lengths) != 1:
        raise ValueError("all columns must have the same length")

    blocks: list[np.ndarray] = []
    names: list[str] = []
    for column_name, raw_values in columns.items():
        values = np.asarray(raw_values)
        for category in np.unique(values):
            blocks.append((values == category).astype(float))
            names.append(f"{column_name}={category}")
    return np.column_stack(blocks), names


def random_oversample(X: np.ndarray, y: np.ndarray,
                      random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes by resampling minority classes with replacement.

    Every class is oversampled up to the majority-class count. This is
    the imbalance treatment documented in the report: it is simple,
    model-agnostic, and unlike SMOTE it introduces no synthetic feature
    values that a from-scratch tree has never seen. Must be applied to
    the *training* split only, after the train/test split, so the test
    distribution stays untouched.

    Args:
        X: Feature matrix ``(n_samples, n_features)``.
        y: Labels ``(n_samples,)``.
        random_state: Seed making the resampling reproducible.

    Returns:
        The oversampled ``(X, y)`` pair with equal class counts, shuffled.
    """
    X = _validate_matrix(X)
    y = np.asarray(y)
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max())

    chosen: list[np.ndarray] = []
    for label, count in zip(classes, counts):
        class_indices = np.flatnonzero(y == label)
        if count < target:
            extra = rng.choice(class_indices, size=target - count, replace=True)
            class_indices = np.concatenate([class_indices, extra])
        chosen.append(class_indices)

    order = rng.permutation(np.concatenate(chosen))
    return X[order], y[order]


def stratified_subsample(X: np.ndarray, y: np.ndarray, n_samples: int,
                         random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw a class-proportional subset of at most ``n_samples`` rows.

    Class proportions are preserved (each class keeps at least one
    sample), which is how the experiments shrink Adult and Covertype to
    tractable sizes without destroying their natural imbalance.

    Args:
        X: Feature matrix ``(n_samples, n_features)``.
        y: Labels ``(n_samples,)``.
        n_samples: Desired subset size; returns the data unchanged when
            it is already this small.
        random_state: Seed making the draw reproducible.

    Returns:
        The subsampled ``(X, y)`` pair, shuffled.
    """
    X = _validate_matrix(X)
    y = np.asarray(y)
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if y.shape[0] <= n_samples:
        return X, y

    rng = np.random.default_rng(random_state)
    fraction = n_samples / y.shape[0]
    chosen: list[np.ndarray] = []
    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        take = max(1, int(round(class_indices.shape[0] * fraction)))
        chosen.append(rng.choice(class_indices, size=min(take, class_indices.shape[0]),
                                 replace=False))
    order = rng.permutation(np.concatenate(chosen))
    return X[order], y[order]


def _validate_matrix(X: np.ndarray, allow_nan: bool = False) -> np.ndarray:
    """Coerce ``X`` to a 2-D float array, optionally permitting NaN."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2-dimensional array")
    if not allow_nan and not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X
