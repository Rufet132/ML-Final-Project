"""K-Means clustering implemented with NumPy."""

from typing import Optional

import numpy as np


class KMeans:
    """Lloyd's K-Means algorithm with reproducible random initialization."""

    def __init__(self, n_clusters: int, max_iter: int = 300,
                 tol: float = 1e-4, random_state: Optional[int] = None) -> None:
        if not isinstance(n_clusters, (int, np.integer)) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if tol < 0 or not np.isfinite(tol):
            raise ValueError("tol must be a finite non-negative number")
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2-dimensional array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def fit(self, X: np.ndarray) -> "KMeans":
        """Cluster ``X`` using assignment and centroid-update steps."""
        X = self._validate_X(X)
        n_samples, n_features = X.shape
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters must be less than or equal to n_samples")

        rng = np.random.RandomState(self.random_state)
        centroids = X[rng.choice(n_samples, self.n_clusters, replace=False)].copy()

        for iteration in range(1, self.max_iter + 1):
            squared_distances = self._squared_distances(X, centroids)
            labels = np.argmin(squared_distances, axis=1)
            new_centroids = np.empty((self.n_clusters, n_features), dtype=float)
            nearest_distance = squared_distances[np.arange(n_samples), labels]

            for cluster in range(self.n_clusters):
                members = X[labels == cluster]
                if members.size:
                    new_centroids[cluster] = members.mean(axis=0)
                else:
                    # Re-seed an empty cluster at the currently worst represented point.
                    replacement = int(np.argmax(nearest_distance))
                    new_centroids[cluster] = X[replacement]
                    nearest_distance[replacement] = -np.inf

            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            if shift <= self.tol:
                break

        squared_distances = self._squared_distances(X, centroids)
        labels = np.argmin(squared_distances, axis=1)
        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = float(squared_distances[np.arange(n_samples), labels].sum())
        self.n_iter_ = iteration
        return self

    @staticmethod
    def _squared_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        differences = X[:, None, :] - centroids[None, :, :]
        return np.einsum("nkd,nkd->nk", differences, differences)

    @staticmethod
    def _compute_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compatibility helper returning Euclidean sample-centroid distances."""
        return np.sqrt(KMeans._squared_distances(X, centroids))
