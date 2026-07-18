"""Experiment 6 — Bias-variance decomposition on a balanced binary dataset.

Following Breiman (1996) adapted to 0-1 loss: B bootstrap replicates of
the training set each train one model, all models predict a fixed test
set, and for every test point the "main prediction" is the majority
vote across replicates. Then

* bias      = error of the main prediction against the true labels,
* variance  = average disagreement of individual replicates with the
  main prediction.

Boosting is expected to show lower bias, bagging lower variance —
that contrast is the answer to the project's central question.
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
from src.trees.decision_tree import DecisionTree

#: Bootstrap replicates (brief: 100).
N_REPLICATES_FULL: int = 100
N_REPLICATES_FAST: int = 20

#: Ensemble size per replicate; smaller than the head-to-head 100 purely
#: to keep B x n_estimators tractable, identical for both ensembles so
#: the bias/variance contrast stays fair.
ENSEMBLE_SIZE_FULL: int = 50
ENSEMBLE_SIZE_FAST: int = 10

#: The balanced binary dataset used for the decomposition.
TARGET_DATASET: str = "breast_cancer"

#: Worker processes for our forest (ML_N_JOBS overrides).
N_JOBS: int = default_n_jobs()


def decompose(predictions: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float]:
    """0-1-loss bias/variance decomposition of replicate predictions.

    Args:
        predictions: Matrix of shape ``(n_replicates, n_test)``.
        y_test: True test labels.

    Returns:
        ``(bias, variance, mean_error)`` where ``mean_error`` is the
        average replicate test error (reported for context).
    """
    n_replicates, n_test = predictions.shape
    main_prediction = np.empty(n_test, dtype=predictions.dtype)
    for column in range(n_test):
        values, counts = np.unique(predictions[:, column], return_counts=True)
        main_prediction[column] = values[np.argmax(counts)]

    bias = float(np.mean(main_prediction != y_test))
    variance = float(np.mean(predictions != main_prediction[np.newaxis, :]))
    mean_error = float(np.mean(predictions != y_test[np.newaxis, :]))
    return bias, variance, mean_error


def run(datasets: list[Dataset], fast: bool = False,
        seed: int = RANDOM_SEED) -> list[dict]:
    """Run the decomposition for a single tree, AdaBoost, and Random Forest."""
    output = figure_dir("bias_variance")
    n_replicates = N_REPLICATES_FAST if fast else N_REPLICATES_FULL
    ensemble_size = ENSEMBLE_SIZE_FAST if fast else ENSEMBLE_SIZE_FULL

    dataset = next((d for d in datasets if d.name == TARGET_DATASET), None)
    if dataset is None:
        raise ValueError(f"bias_variance requires the {TARGET_DATASET!r} dataset")

    X_train, X_test, y_train, y_test = prepare_split(dataset, random_state=seed)
    n_train = X_train.shape[0]
    print(f"  bias-variance on {dataset.name} "
          f"({n_replicates} replicates, ensembles of {ensemble_size})")

    def make_models(replicate_seed: int) -> dict:
        return {
            "decision_tree": DecisionTree(random_state=replicate_seed),
            "adaboost": AdaBoostClassifier(n_estimators=ensemble_size,
                                           random_state=replicate_seed),
            "random_forest": RandomForestClassifier(
                n_estimators=ensemble_size, n_jobs=N_JOBS,
                random_state=replicate_seed),
        }

    model_names = list(make_models(0))
    predictions = {name: np.empty((n_replicates, X_test.shape[0]), dtype=y_test.dtype)
                   for name in model_names}

    rng = np.random.default_rng(seed)
    with Timer(f"{n_replicates} bootstrap replicates x {len(model_names)} models"):
        for replicate in range(n_replicates):
            indices = rng.integers(0, n_train, size=n_train)
            X_boot, y_boot = X_train[indices], y_train[indices]
            for name, model in make_models(seed + replicate).items():
                model.fit(X_boot, y_boot)
                predictions[name][replicate] = model.predict(X_test)

    rows: list[dict] = []
    for name in model_names:
        bias, variance, mean_error = decompose(predictions[name], y_test)
        rows.append({"dataset": dataset.name, "model": name,
                     "n_replicates": n_replicates, "ensemble_size": ensemble_size,
                     "bias": bias, "variance": variance, "mean_error": mean_error})
        print(f"    {name}: bias={bias:.4f}, variance={variance:.4f}, "
              f"mean_error={mean_error:.4f}")

    _plot(rows, dataset.name, output)
    save_results_csv(rows, output / "bias_variance.csv")
    return rows


def _plot(rows: list[dict], dataset_name: str, output) -> None:
    """Grouped bar chart of bias and variance per model."""
    import matplotlib.pyplot as plt

    models = [row["model"] for row in rows]
    positions = np.arange(len(models))
    width = 0.35

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(positions - width / 2, [row["bias"] for row in rows], width, label="Bias")
    axis.bar(positions + width / 2, [row["variance"] for row in rows], width,
             label="Variance")
    axis.set_xticks(positions)
    axis.set_xticklabels(models, rotation=10)
    axis.set_ylabel("0-1 loss component")
    axis.set_title(f"Experiment 6 — Bias-variance decomposition on {dataset_name}")
    axis.grid(alpha=0.25, axis="y")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / f"bias_variance_{dataset_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point: ``python -m src.experiments.bias_variance [--fast]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced sizes")
    args = parser.parse_args()
    run(load_datasets(fast=args.fast, names=[TARGET_DATASET]), fast=args.fast)


if __name__ == "__main__":
    main()
