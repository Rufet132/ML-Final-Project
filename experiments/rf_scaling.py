"""Experiment 3 — Random Forest scaling curves.

(a) Test and OOB accuracy vs number of trees: one forest with the
    maximum tree count is fitted, then every prefix of its trees is
    evaluated by accumulating per-tree probability votes — 200 points
    for the cost of one fit.
(b) Test accuracy vs ``max_depth`` at a fixed number of trees.
"""

from __future__ import annotations

import numpy as np

from src.bagging.random_forest import RandomForestClassifier
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

#: Tree counts for curve (a) in full and fast mode (brief: 1..200).
MAX_TREES_FULL: int = 200
MAX_TREES_FAST: int = 40

#: Fixed forest size for the depth sweep (b) (brief: 100).
DEPTH_SWEEP_TREES_FULL: int = 100
DEPTH_SWEEP_TREES_FAST: int = 20

#: Depth grid for sweep (b) (brief: 1..20).
DEPTHS_FULL: list[int] = list(range(1, 21))
DEPTHS_FAST: list[int] = [1, 2, 3, 4, 6, 8, 10, 12]

#: Worker processes for tree training (ML_N_JOBS overrides).
N_JOBS: int = default_n_jobs()


def _prefix_test_accuracy(forest: RandomForestClassifier, X_test: np.ndarray,
                          y_test: np.ndarray) -> np.ndarray:
    """Accuracy of the first k trees for every k, from one fit."""
    assert forest.classes_ is not None
    votes = np.zeros((X_test.shape[0], len(forest.classes_)))
    accuracies = np.empty(len(forest.estimators_))
    class_index = {label: i for i, label in enumerate(forest.classes_)}
    for k, tree in enumerate(forest.estimators_):
        proba = tree.predict_proba(X_test)
        assert tree.classes_ is not None
        for column, label in enumerate(tree.classes_):
            votes[:, class_index[label]] += proba[:, column]
        predictions = forest.classes_[np.argmax(votes, axis=1)]
        accuracies[k] = float(np.mean(predictions == y_test))
    return accuracies


def _prefix_oob_accuracy(forest: RandomForestClassifier, X_train: np.ndarray,
                         y_train: np.ndarray) -> np.ndarray:
    """OOB accuracy of the first k trees for every k.

    Uses the forest's stored bootstrap OOB masks (internal state of our
    own class): each tree votes only for the samples it never saw, and
    the prefix score counts samples with at least one OOB vote so far.
    """
    assert forest.classes_ is not None
    n_samples = X_train.shape[0]
    votes = np.zeros((n_samples, len(forest.classes_)))
    has_vote = np.zeros(n_samples, dtype=bool)
    accuracies = np.full(len(forest.estimators_), np.nan)
    class_index = {label: i for i, label in enumerate(forest.classes_)}

    for k, (tree, oob_mask) in enumerate(zip(forest.estimators_, forest._oob_masks)):
        if np.any(oob_mask):
            predictions = tree.predict(X_train[oob_mask])
            for local, sample in enumerate(np.flatnonzero(oob_mask)):
                votes[sample, class_index[predictions[local]]] += 1
            has_vote |= oob_mask
        if np.any(has_vote):
            oob_pred = forest.classes_[np.argmax(votes[has_vote], axis=1)]
            accuracies[k] = float(np.mean(oob_pred == y_train[has_vote]))
    return accuracies


def run(datasets: list[Dataset], fast: bool = False,
        seed: int = RANDOM_SEED) -> list[dict]:
    """Run both scaling studies for every dataset."""
    output = figure_dir("rf_scaling")
    max_trees = MAX_TREES_FAST if fast else MAX_TREES_FULL
    sweep_trees = DEPTH_SWEEP_TREES_FAST if fast else DEPTH_SWEEP_TREES_FULL
    depths = DEPTHS_FAST if fast else DEPTHS_FULL
    rows: list[dict] = []

    for dataset in datasets:
        X_train, X_test, y_train, y_test = prepare_split(dataset, random_state=seed)
        class_weight = "balanced" if dataset.imbalanced else None
        print(f"  rf scaling on {dataset.name} (max {max_trees} trees, "
              f"depth sweep {depths[0]}..{depths[-1]} x{sweep_trees})")

        # (a) accuracy vs number of trees, from a single fit.
        with Timer(f"forest fit ({max_trees} trees)"):
            forest = RandomForestClassifier(
                n_estimators=max_trees, max_depth=dataset.ensemble_max_depth,
                oob_score=True, n_jobs=N_JOBS, random_state=seed,
                class_weight=class_weight,
            ).fit(X_train, y_train)
        with Timer("prefix evaluation"):
            test_curve = _prefix_test_accuracy(forest, X_test, y_test)
            oob_curve = _prefix_oob_accuracy(forest, X_train, y_train)

        for k in range(max_trees):
            rows.append({"dataset": dataset.name, "study": "n_estimators",
                         "n_estimators": k + 1,
                         "max_depth": dataset.ensemble_max_depth,
                         "test_accuracy": float(test_curve[k]),
                         "oob_accuracy": float(oob_curve[k])})

        # (b) accuracy vs depth at a fixed number of trees.
        depth_accuracies: list[float] = []
        with Timer(f"depth sweep ({len(depths)} forests)"):
            for depth in depths:
                sweep_forest = RandomForestClassifier(
                    n_estimators=sweep_trees, max_depth=depth,
                    n_jobs=N_JOBS, random_state=seed, class_weight=class_weight,
                ).fit(X_train, y_train)
                accuracy = float(np.mean(sweep_forest.predict(X_test) == y_test))
                depth_accuracies.append(accuracy)
                rows.append({"dataset": dataset.name, "study": "max_depth",
                             "n_estimators": sweep_trees, "max_depth": depth,
                             "test_accuracy": accuracy,
                             "oob_accuracy": float("nan")})

        _plot(dataset.name, test_curve, oob_curve, depths, depth_accuracies,
              sweep_trees, output)

    save_results_csv(rows, output / "rf_scaling.csv")
    return rows


def _plot(dataset_name: str, test_curve: np.ndarray, oob_curve: np.ndarray,
          depths: list[int], depth_accuracies: list[float],
          sweep_trees: int, output) -> None:
    """Two-panel figure: trees sweep with OOB, and depth sweep."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    trees = np.arange(1, test_curve.shape[0] + 1)
    axes[0].plot(trees, test_curve, label="Test accuracy")
    axes[0].plot(trees, oob_curve, label="OOB accuracy", linestyle="--")
    axes[0].set(xlabel="Number of trees", ylabel="Accuracy",
                title="(a) Accuracy vs n_estimators")
    axes[1].plot(depths, depth_accuracies, marker="o")
    axes[1].set(xlabel="max_depth", ylabel="Accuracy",
                title=f"(b) Accuracy vs max_depth ({sweep_trees} trees)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle(f"Experiment 3 — Random Forest scaling on {dataset_name}")
    fig.tight_layout()
    fig.savefig(output / f"rf_scaling_{dataset_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point: ``python -m src.experiments.rf_scaling [--fast]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced sizes")
    args = parser.parse_args()
    run(load_datasets(fast=args.fast), fast=args.fast)


if __name__ == "__main__":
    main()
