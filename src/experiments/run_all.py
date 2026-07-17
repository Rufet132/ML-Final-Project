"""One-command reproduction entry point for the implemented project experiments.

Currently Module 4 is complete; the remaining experiment modules are still repository
placeholders. Run from the repository root with:

    python src/experiments/run_all.py
"""

from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unsupervised_analysis import run_unsupervised_pipeline


def _datasets() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return three bundled datasets with different size/dimensionality profiles."""
    datasets = []
    for name, loader in (("breast_cancer", load_breast_cancer),
                         ("wine", load_wine), ("digits", load_digits)):
        dataset = loader()
        X = StandardScaler().fit_transform(np.asarray(dataset.data, dtype=float))
        datasets.append((name, X, np.asarray(dataset.target)))
    return datasets


def main() -> None:
    figures_dir = ROOT / "figures" / "unsupervised"
    print("Running Module 4: unsupervised analysis")
    print(f"Figures will be written to {figures_dir}")
    results = [run_unsupervised_pipeline(X, y, name, random_state=42,
                                         figures_dir=figures_dir)
               for name, X, y in _datasets()]
    print("\nSummary")
    print("dataset          PCA90   K-Means ARI   DBSCAN ARI   noise")
    for result in results:
        print(f"{result['dataset']:<16} {result['components_90']:>5}   "
              f"{result['kmeans_ari']:>11.4f}   {result['dbscan_ari']:>10.4f}   "
              f"{result['dbscan_noise_fraction']:>6.1%}")


if __name__ == "__main__":
    main()
