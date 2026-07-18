"""Experiment 2 — AdaBoost learning curves over the number of stumps.

One AdaBoost ensemble with the maximum number of estimators is fitted
per dataset, and ``staged_predict`` recovers the train/test error after
every boosting round — so the whole 1..200 curve costs a single fit
instead of 200 refits.
"""

from __future__ import annotations

import numpy as np

from src.boosting.adaboost import AdaBoostClassifier
from src.experiments.utils import (
    Dataset,
    RANDOM_SEED,
    Timer,
    figure_dir,
    load_datasets,
    prepare_split,
    save_results_csv,
)

#: Boosting rounds in full and fast mode (brief: 1..200).
MAX_ESTIMATORS_FULL: int = 200
MAX_ESTIMATORS_FAST: int = 40


def _staged_errors(model: AdaBoostClassifier, X: np.ndarray,
                   y: np.ndarray) -> np.ndarray:
    """Misclassification rate after each boosting round."""
    return np.array([float(np.mean(stage != y))
                     for stage in model.staged_predict(X)])


def run(datasets: list[Dataset], fast: bool = False,
        seed: int = RANDOM_SEED) -> list[dict]:
    """Fit one max-size AdaBoost per dataset and record staged errors."""
    output = figure_dir("adaboost_scaling")
    n_estimators = MAX_ESTIMATORS_FAST if fast else MAX_ESTIMATORS_FULL
    rows: list[dict] = []

    for dataset in datasets:
        X_train, X_test, y_train, y_test = prepare_split(dataset, random_state=seed)
        print(f"  adaboost scaling on {dataset.name} (max {n_estimators} stumps)")

        with Timer("adaboost fit + staged evaluation"):
            model = AdaBoostClassifier(n_estimators=n_estimators,
                                       random_state=seed).fit(X_train, y_train)
            train_error = _staged_errors(model, X_train, y_train)
            test_error = _staged_errors(model, X_test, y_test)

        fitted_rounds = train_error.shape[0]
        if fitted_rounds < n_estimators:
            print(f"    early stop after {fitted_rounds} rounds "
                  f"(next stump no better than random)")

        rounds = np.arange(1, fitted_rounds + 1)
        for round_number, tr, te in zip(rounds, train_error, test_error):
            rows.append({"dataset": dataset.name, "n_estimators": int(round_number),
                         "train_error": float(tr), "test_error": float(te)})

        _plot(rounds, train_error, test_error, dataset.name, output)

    save_results_csv(rows, output / "adaboost_scaling.csv")
    return rows


def _plot(rounds: np.ndarray, train_error: np.ndarray, test_error: np.ndarray,
          dataset_name: str, output) -> None:
    """Train/test error vs number of boosting rounds."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(rounds, train_error, label="Train error")
    axis.plot(rounds, test_error, label="Test error")
    axis.set(xlabel="Number of estimators (boosting rounds)",
             ylabel="Misclassification error",
             title=f"Experiment 2 — AdaBoost scaling on {dataset_name}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / f"adaboost_scaling_{dataset_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point: ``python -m src.experiments.adaboost_scaling [--fast]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced sizes")
    args = parser.parse_args()
    run(load_datasets(fast=args.fast), fast=args.fast)


if __name__ == "__main__":
    main()
