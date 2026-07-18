"""Data preprocessing helpers implemented with NumPy."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class StandardScaler:
    """Standardize features using statistics learned by :meth:`fit`."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        data = _as_feature_matrix(X)
        self.mean_ = np.mean(data, axis=0)
        scale = np.std(data, axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must be fitted before transform")
        data = _as_feature_matrix(X)
        if data.shape[1] != self.mean_.size:
            raise ValueError("X has a different number of features than fitted data")
        return (data - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _as_feature_matrix(X: np.ndarray) -> np.ndarray:
    data = np.asarray(X, dtype=float)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if data.shape[0] == 0:
        raise ValueError("X must not be empty")
    return data


def _validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(X)
    target = np.asarray(y).reshape(-1)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if data.shape[0] != target.size:
        raise ValueError("X and y must have the same number of samples")
    return data, target


def drop_missing_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop samples containing a NaN feature."""
    data, target = _validate_xy(X, y)
    try:
        missing = np.isnan(data.astype(float)).any(axis=1)
    except (TypeError, ValueError):
        missing = np.equal(data, None).any(axis=1)  # noqa: E711
    return data[~missing], target[~missing]


def one_hot_encode(columns: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """One-hot encode named categorical columns in mapping order."""
    if not columns:
        raise ValueError("at least one column is required")
    lengths = {np.asarray(values).reshape(-1).size for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("all columns must have the same length")

    matrices: list[np.ndarray] = []
    names: list[str] = []
    for name, values in columns.items():
        values = np.asarray(values).reshape(-1)
        categories = np.unique(values)
        matrices.append((values[:, None] == categories[None, :]).astype(float))
        names.extend(f"{name}={category}" for category in categories)
    return np.column_stack(matrices), names


def random_oversample(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly duplicate minority samples until every class is balanced."""
    data, target = _validate_xy(X, y)
    classes, counts = np.unique(target, return_counts=True)
    if classes.size == 0:
        raise ValueError("y must not be empty")
    rng = np.random.default_rng(random_state)
    target_count = int(np.max(counts))
    sampled_indices: list[np.ndarray] = []
    for label, count in zip(classes, counts):
        indices = np.flatnonzero(target == label)
        extras = rng.choice(indices, size=target_count - int(count), replace=True)
        sampled_indices.append(np.concatenate([indices, extras]))
    combined = np.concatenate(sampled_indices)
    rng.shuffle(combined)
    return data[combined], target[combined]


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return at most ``n_samples`` while approximately preserving class ratios."""
    data, target = _validate_xy(X, y)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if data.shape[0] <= n_samples:
        return data.copy(), target.copy()

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(target, return_counts=True)
    quotas = counts * (n_samples / target.size)
    allocations = np.floor(quotas).astype(int)
    remainder = n_samples - int(np.sum(allocations))
    priorities = np.argsort(-(quotas - allocations), kind="stable")
    allocations[priorities[:remainder]] += 1

    selected = [
        rng.choice(np.flatnonzero(target == label), size=int(amount), replace=False)
        for label, amount in zip(classes, allocations)
        if amount > 0
    ]
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return data[indices], target[indices]
