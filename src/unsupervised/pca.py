"""Principal component analysis implemented with NumPy."""

from typing import Optional

import numpy as np


class PCA:
    """Principal component analysis using covariance eigendecomposition."""

    def __init__(self, n_components: int) -> None:
        if not isinstance(n_components, (int, np.integer)) or n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        self.n_components = int(n_components)
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    @staticmethod
    def _validate_X(X: np.ndarray, *, min_samples: int = 1) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if X.shape[0] < min_samples or X.shape[1] == 0:
            raise ValueError(f"X must contain at least {min_samples} sample(s) and one feature")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit PCA and retain the leading ``n_components`` directions."""
        X = self._validate_X(X, min_samples=2)
        _, n_features = X.shape
        if self.n_components > n_features:
            raise ValueError("n_components must be less than or equal to n_features")

        self.n_features_in_ = n_features
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        covariance = centered.T @ centered / (X.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        total_variance = float(eigenvalues.sum())
        if total_variance <= np.finfo(float).eps:
            raise ValueError("PCA is undefined when the data has zero variance")

        self.explained_variance_ = eigenvalues[: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.components_ = eigenvectors[:, : self.n_components].T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project samples onto the fitted principal components."""
        if self.components_ is None or self.mean_ is None or self.n_features_in_ is None:
            raise ValueError("PCA must be fitted before calling transform")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and return the projected training samples."""
        return self.fit(X).transform(X)
