"""Experiment 5 — Robustness to label noise.

A fraction eta of *training* labels is flipped to a different random
class, AdaBoost (100 stumps) and Random Forest (100 trees) are trained
on the corrupted data, and both are evaluated on the untouched clean
test split. Boosting is expected to degrade faster: its reweighting
concentrates on the (now mislabeled) hard samples, while bagging's
averaging dilutes them.
"""

from __future__ import annotations

import numpy as np

from src.bagging.random_forest import RandomForestClassifier
from src.boosting.adaboost import AdaBoostClassifier
from src.experiments.utils import (
    Dataset,
    RANDOM_SEED,
    default_n_jobs,
    Timer,
    figure_dir,
    load_datasets,
    prepare_split,
    save_results_csv,
)

#: Label-flip fractions required by the brief, plus the clean reference.
NOISE_LEVELS: list[float] = [0.0, 0.05, 0.10, 0.20]

#: Ensemble sizes in full and fast mode (brief: 100).
N_ESTIMATORS_FULL: int = 100
N_ESTIMATORS_FAST: int = 20

#: Worker processes for our forest (ML_N_JOBS overrides).
N_JOBS: int = default_n_jobs()


def flip_labels(y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of ``y`` with ``fraction`` of entries flipped.

    Each corrupted entry receives a *different* class drawn uniformly
    from the remaining labels, so binary datasets get the complementary
    class and multi-class datasets get genuine confusion.
    """
    y_noisy = y.copy()
    n_flip = int(round(fraction * y.shape[0]))
    if n_flip == 0:
        return y_noisy
    classes = np.unique(y)
    flip_indices = rng.choice(y.shape[0], size=n_flip, replace=False)
    for index in flip_indices:
        alternatives = classes[classes != y_noisy[index]]
        y_noisy[index] = rng.choice(alternatives)
    return y_noisy


def run(datasets: list[Dataset], fast: bool = False,
        seed: int = RANDOM_SEED) -> list[dict]:
    """Train both ensembles at every noise level and record clean-test accuracy."""
    output = figure_dir("noise_robustness")
    n_estimators = N_ESTIMATORS_FAST if fast else N_ESTIMATORS_FULL
    rows: list[dict] = []

    for dataset in datasets:
        X_train, X_test, y_train, y_test = prepare_split(dataset, random_state=seed)
        class_weight = "balanced" if dataset.imbalanced else None
        print(f"  noise robustness on {dataset.name} ({n_estimators} estimators)")
        curves: dict[str, list[float]] = {"adaboost": [], "random_forest": []}

        for eta in NOISE_LEVELS:
            rng = np.random.default_rng(seed + int(eta * 1000))
            y_noisy = flip_labels(y_train, eta, rng)

            with Timer(f"eta={eta:.2f} adaboost"):
                boost = AdaBoostClassifier(n_estimators=n_estimators,
                                           random_state=seed).fit(X_train, y_noisy)
                boost_accuracy = float(np.mean(boost.predict(X_test) == y_test))
            with Timer(f"eta={eta:.2f} random forest"):
                forest = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=dataset.ensemble_max_depth,
                    n_jobs=N_JOBS, random_state=seed, class_weight=class_weight,
                ).fit(X_train, y_noisy)
                forest_accuracy = float(np.mean(forest.predict(X_test) == y_test))

            curves["adaboost"].append(boost_accuracy)
            curves["random_forest"].append(forest_accuracy)
            rows.append({"dataset": dataset.name, "noise": eta,
                         "adaboost_accuracy": boost_accuracy,
                         "random_forest_accuracy": forest_accuracy})
            print(f"    eta={eta:.2f}: adaboost={boost_accuracy:.4f}, "
                  f"rf={forest_accuracy:.4f}")

        _plot(dataset.name, curves, output)

    save_results_csv(rows, output / "noise_robustness.csv")
    return rows


def _plot(dataset_name: str, curves: dict[str, list[float]], output) -> None:
    """Accuracy degradation curves for both ensembles."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    for model_name, values in curves.items():
        axis.plot(NOISE_LEVELS, values, marker="o", label=model_name)
    axis.set(xlabel="Fraction of flipped training labels",
             ylabel="Clean-test accuracy",
             title=f"Experiment 5 — Noise robustness on {dataset_name}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / f"noise_robustness_{dataset_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point: ``python -m src.experiments.noise_robustness [--fast]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced sizes")
    args = parser.parse_args()
    run(load_datasets(fast=args.fast), fast=args.fast)


if __name__ == "__main__":
    main()
