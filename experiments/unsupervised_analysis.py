"""Reproducible PCA, K-Means, and DBSCAN analysis for Module 4."""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.unsupervised.dbscan import DBSCAN
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA


DEFAULT_FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def _safe_name(name: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_"
                      for character in name.strip())
    return cleaned or "dataset"


def _validate_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("X must contain at least two samples and two features")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be one-dimensional with one label per sample")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X, y


def run_pca_analysis(X: np.ndarray, y: np.ndarray, dataset_name: str,
                     random_state: int = 42,
                     figures_dir: Optional[Path] = None) -> Tuple[PCA, np.ndarray, int]:
    """Create the cumulative scree plot and true-label PCA projection."""
    del random_state  # PCA is deterministic.
    X, y = _validate_inputs(X, y)
    output = Path(figures_dir or DEFAULT_FIGURES_DIR)
    output.mkdir(parents=True, exist_ok=True)
    name = _safe_name(dataset_name)

    full_pca = PCA(n_components=X.shape[1]).fit(X)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    reached = np.flatnonzero(cumulative >= 0.90)
    components_90 = int(reached[0] + 1) if reached.size else X.shape[1]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.arange(1, X.shape[1] + 1), cumulative, marker="o", markersize=3)
    axis.axhline(0.90, color="tab:red", linestyle="--", label="90% threshold")
    axis.axvline(components_90, color="tab:green", linestyle="--",
                 label=f"{components_90} components")
    axis.set(xlabel="Number of principal components", ylabel="Cumulative explained variance",
             title=f"PCA scree plot — {dataset_name}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / f"pca_scree_{name}.png", dpi=150)
    plt.close(fig)

    pca_2d = PCA(n_components=2)
    projection = pca_2d.fit_transform(X)
    _plot_projection(projection, y, f"True classes — {dataset_name}", "Class",
                     output / f"pca_true_labels_{name}.png")
    return pca_2d, projection, components_90


def _plot_projection(projection: np.ndarray, labels: np.ndarray, title: str,
                     colorbar_label: str, path: Path, noise_label: bool = False,
                     centroids: Optional[np.ndarray] = None) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    labels = np.asarray(labels)
    if noise_label:
        regular = labels != -1
        if np.any(regular):
            scatter = axis.scatter(projection[regular, 0], projection[regular, 1],
                                   c=labels[regular], cmap="viridis", alpha=0.7, s=28)
            fig.colorbar(scatter, ax=axis, label=colorbar_label)
        if np.any(~regular):
            axis.scatter(projection[~regular, 0], projection[~regular, 1], marker="x",
                         color="tab:red", s=35, label="Noise")
            axis.legend()
    else:
        scatter = axis.scatter(projection[:, 0], projection[:, 1], c=labels,
                               cmap="viridis", alpha=0.7, s=28)
        fig.colorbar(scatter, ax=axis, label=colorbar_label)
    if centroids is not None:
        axis.scatter(centroids[:, 0], centroids[:, 1], color="tab:red", marker="X",
                     s=150, edgecolor="black", label="Centroids")
        axis.legend()
    axis.set(xlabel="PC1", ylabel="PC2", title=title)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_kmeans_analysis(X: np.ndarray, y: np.ndarray, projection: np.ndarray,
                        pca: PCA, dataset_name: str, random_state: int = 42,
                        figures_dir: Optional[Path] = None,
                        n_restarts: int = 10) -> Tuple[KMeans, int, dict]:
    """Run the elbow experiment and keep the lowest-inertia restart for each k."""
    X, y = _validate_inputs(X, y)
    output = Path(figures_dir or DEFAULT_FIGURES_DIR)
    output.mkdir(parents=True, exist_ok=True)
    name = _safe_name(dataset_name)
    max_k = min(10, X.shape[0])
    k_values = np.arange(1, max_k + 1)
    inertias: list[float] = []
    ari_scores: list[float] = []
    best_models: list[KMeans] = []

    for k in k_values:
        candidates = [KMeans(int(k), random_state=random_state + restart).fit(X)
                      for restart in range(n_restarts)]
        best = min(candidates, key=lambda model: float(model.inertia_))
        best_models.append(best)
        inertias.append(float(best.inertia_))
        ari_scores.append(float(adjusted_rand_score(y, best.labels_)))

    # "Best k" is explicitly evaluated against known classes for the requested ARI study.
    best_index = int(np.argmax(ari_scores))
    model = best_models[best_index]
    best_k = int(k_values[best_index])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(k_values, inertias, marker="o")
    axes[0].axvline(best_k, color="tab:red", linestyle="--", label=f"best k={best_k}")
    axes[0].set(xlabel="Number of clusters (k)", ylabel="Inertia", title="Elbow curve")
    axes[1].plot(k_values, ari_scores, marker="o", color="tab:green")
    axes[1].axvline(best_k, color="tab:red", linestyle="--")
    axes[1].set(xlabel="Number of clusters (k)", ylabel="ARI", title="ARI against true classes")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle(f"K-Means analysis — {dataset_name}")
    fig.tight_layout()
    fig.savefig(output / f"kmeans_elbow_{name}.png", dpi=150)
    plt.close(fig)

    centroid_projection = pca.transform(model.centroids_)
    _plot_projection(projection, model.labels_, f"K-Means clusters (k={best_k}) — {dataset_name}",
                     "Cluster", output / f"pca_kmeans_{name}.png",
                     centroids=centroid_projection)
    details = {"k_values": k_values.tolist(), "inertias": inertias, "ari_scores": ari_scores}
    return model, best_k, details


def _k_distances(X: np.ndarray, min_samples: int) -> np.ndarray:
    squared_norms = np.einsum("ij,ij->i", X, X)
    distances_squared = squared_norms[:, None] + squared_norms[None, :] - 2 * X @ X.T
    np.maximum(distances_squared, 0.0, out=distances_squared)
    # min_samples includes the point itself, matching the DBSCAN definition.
    kth = np.partition(distances_squared, min_samples - 1, axis=1)[:, min_samples - 1]
    return np.sort(np.sqrt(kth))


def _knee_index(values: np.ndarray) -> int:
    if values.size < 3 or np.isclose(values[-1], values[0]):
        return max(0, values.size - 1)
    x = np.linspace(0.0, 1.0, values.size)
    y = (values - values[0]) / (values[-1] - values[0])
    # Sorted k-distances normally form a convex curve below the endpoint chord.
    return int(np.argmax(x - y))


def run_dbscan_analysis(X: np.ndarray, y: np.ndarray, projection: np.ndarray,
                        dataset_name: str, random_state: int = 42,
                        figures_dir: Optional[Path] = None,
                        min_samples: Optional[int] = None) -> Tuple[DBSCAN, float, dict]:
    """Create a k-distance plot and evaluate epsilon values around its knee."""
    del random_state  # DBSCAN is deterministic.
    X, y = _validate_inputs(X, y)
    output = Path(figures_dir or DEFAULT_FIGURES_DIR)
    output.mkdir(parents=True, exist_ok=True)
    name = _safe_name(dataset_name)
    if min_samples is None:
        min_samples = min(max(5, 2 * X.shape[1]), X.shape[0])
    if not 1 <= min_samples <= X.shape[0]:
        raise ValueError("min_samples must be between 1 and n_samples")

    distances = _k_distances(X, min_samples)
    knee = _knee_index(distances)
    knee_eps = max(float(distances[knee]), np.finfo(float).eps)
    positive = distances[distances > 0]
    if positive.size:
        candidates = np.unique(np.quantile(positive, np.linspace(0.70, 0.98, 15)))
        candidates = np.unique(np.append(candidates, knee_eps))
    else:
        candidates = np.array([knee_eps])

    models = [DBSCAN(float(eps), min_samples).fit(X) for eps in candidates]
    ari_scores = [float(adjusted_rand_score(y, model.labels_)) for model in models]
    best_index = int(np.argmax(ari_scores))
    model = models[best_index]
    best_eps = float(candidates[best_index])

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.arange(distances.size), distances)
    axis.axhline(knee_eps, color="tab:orange", linestyle="--", label=f"knee ε={knee_eps:.3g}")
    axis.axhline(best_eps, color="tab:red", linestyle=":", label=f"best ε={best_eps:.3g}")
    axis.set(xlabel="Points sorted by distance", ylabel=f"{min_samples}-nearest-neighbour distance",
             title=f"DBSCAN k-distance plot — {dataset_name}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / f"dbscan_kdistance_{name}.png", dpi=150)
    plt.close(fig)

    _plot_projection(projection, model.labels_,
                     f"DBSCAN clusters (ε={best_eps:.3g}) — {dataset_name}", "Cluster",
                     output / f"pca_dbscan_{name}.png", noise_label=True)
    noise_fraction = float(np.mean(model.labels_ == -1))
    details = {"min_samples": min_samples, "knee_eps": knee_eps,
               "eps_values": candidates.tolist(), "ari_scores": ari_scores,
               "noise_fraction": noise_fraction}
    return model, best_eps, details


def run_unsupervised_pipeline(X: np.ndarray, y: np.ndarray, dataset_name: str,
                              random_state: int = 42,
                              figures_dir: Optional[Path] = None) -> dict:
    """Run every required Module 4 analysis and return report-ready metrics."""
    X, y = _validate_inputs(X, y)
    pca, projection, components_90 = run_pca_analysis(
        X, y, dataset_name, random_state, figures_dir)
    kmeans, best_k, kmeans_details = run_kmeans_analysis(
        X, y, projection, pca, dataset_name, random_state, figures_dir)
    dbscan, best_eps, dbscan_details = run_dbscan_analysis(
        X, y, projection, dataset_name, random_state, figures_dir)

    result = {
        "dataset": dataset_name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "components_90": components_90,
        "pca_2d_variance": float(pca.explained_variance_ratio_.sum()),
        "pca": pca,
        "X_pca": projection,
        "kmeans": kmeans,
        "optimal_k": best_k,
        "kmeans_ari": float(adjusted_rand_score(y, kmeans.labels_)),
        "kmeans_analysis": kmeans_details,
        "dbscan": dbscan,
        "best_eps": best_eps,
        "dbscan_ari": float(adjusted_rand_score(y, dbscan.labels_)),
        "dbscan_noise_fraction": dbscan_details["noise_fraction"],
        "dbscan_analysis": dbscan_details,
    }
    result["better_clustering"] = (
        "K-Means" if result["kmeans_ari"] >= result["dbscan_ari"] else "DBSCAN")
    print(f"{dataset_name}: PCA90={components_90}, K-Means ARI={result['kmeans_ari']:.4f}, "
          f"DBSCAN ARI={result['dbscan_ari']:.4f}, noise={result['dbscan_noise_fraction']:.2%}")
    return result
