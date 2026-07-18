"""DBSCAN density-based clustering implemented with NumPy."""

from collections import deque
from typing import Optional

import numpy as np


class DBSCAN:
    """Density-based clustering; label ``-1`` represents noise."""

    def __init__(self, eps: float, min_samples: int) -> None:
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a finite positive number")
        if not isinstance(min_samples, (int, np.integer)) or min_samples <= 0:
            raise ValueError("min_samples must be a positive integer")
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2-dimensional array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """Find density-connected components in ``X``."""
        X = self._validate_X(X)
        neighbors = self._get_neighbors(X)
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        is_core = np.fromiter(
            (len(indices) >= self.min_samples for indices in neighbors),
            dtype=bool,
            count=n_samples,
        )
        cluster_id = 0

        for point in range(n_samples):
            if visited[point]:
                continue
            visited[point] = True
            if not is_core[point]:
                continue

            labels[point] = cluster_id
            queue = deque(neighbors[point])
            queued = np.zeros(n_samples, dtype=bool)
            queued[neighbors[point]] = True
            while queue:
                neighbor = queue.popleft()
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if is_core[neighbor]:
                        for candidate in neighbors[neighbor]:
                            if not queued[candidate]:
                                queue.append(candidate)
                                queued[candidate] = True
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(is_core)
        return self

    def _get_neighbors(self, X: np.ndarray) -> list[np.ndarray]:
        squared_norms = np.einsum("ij,ij->i", X, X)
        squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2 * X @ X.T
        np.maximum(squared_distances, 0.0, out=squared_distances)
        radius_squared = self.eps * self.eps
        return [np.flatnonzero(row <= radius_squared) for row in squared_distances]
