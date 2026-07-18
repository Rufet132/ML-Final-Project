"""Bonus experiment — Gradient Boosting vs AdaBoost, round by round.

For the binary datasets, staged test error of three boosting flavours
with the same round budget:

* AdaBoost / SAMME (hard votes, sample reweighting),
* AdaBoost / SAMME.R (probability-weighted votes),
* our GradientBoostingClassifier (log-loss, Newton leaf steps).

The figure documents the bonus requirement "basic GBM with log-loss,
comparison against AdaBoost". MNIST is skipped here: depth-3 regression
trees scanning 784 features every round make GBM disproportionately
slow without adding to the comparison.
"""

from __future__ import annotations

import numpy as np

from src.boosting.adaboost import AdaBoostClassifier
from src.boosting.gradient_boosting import GradientBoostingClassifier
from src.experiments.utils import (
    Dataset,
    RANDOM_SEED,
    Timer,
    figure_dir,
    load_datasets,
    prepare_split,
    save_results_csv,
)

#: Boosting rounds in full and fast mode.
N_ROUNDS_FULL: int = 200
N_ROUNDS_FAST: int = 40

#: Binary datasets used for the comparison.
COMPARISON_DATASETS: list[str] = ["breast_cancer", "adult"]


def _staged_errors(model, X: np.ndarray, y: np.ndarray) -> list[float]:
    """Misclassification rate after each boosting round."""
    return [float(np.mean(stage != y)) for stage in model.staged_predict(X)]


def run(datasets: list[Dataset], fast: bool = False,
        seed: int = RANDOM_SEED) -> list[dict]:
    """Fit the three boosters per dataset and record staged test error."""
    output = figure_dir("gbm_comparison")
    n_rounds = N_ROUNDS_FAST if fast else N_ROUNDS_FULL
    rows: list[dict] = []

    for dataset in datasets:
        if dataset.name not in COMPARISON_DATASETS:
            continue
        X_train, X_test, y_train, y_test = prepare_split(dataset, random_state=seed)
        print(f"  gbm comparison on {dataset.name} ({n_rounds} rounds)")

        with Timer("adaboost SAMME"):
            samme = AdaBoostClassifier(n_estimators=n_rounds,
                                       random_state=seed).fit(X_train, y_train)
        with Timer("adaboost SAMME.R"):
            samme_r = AdaBoostClassifier(n_estimators=n_rounds, algorithm="SAMME.R",
                                         random_state=seed).fit(X_train, y_train)
        with Timer("gradient boosting"):
            gbm = GradientBoostingClassifier(n_estimators=n_rounds,
                                             random_state=seed).fit(X_train, y_train)

        curves = {
            "adaboost_samme": _staged_errors(samme, X_test, y_test),
            "adaboost_samme_r": _staged_errors(samme_r, X_test, y_test),
            "gbm_logloss": _staged_errors(gbm, X_test, y_test),
        }

        for model_name, errors in curves.items():
            for round_number, error in enumerate(errors, start=1):
                rows.append({"dataset": dataset.name, "model": model_name,
                             "round": round_number, "test_error": error})
            print(f"    {model_name}: final test error={errors[-1]:.4f} "
                  f"({len(errors)} rounds)")

        _plot(dataset.name, curves, output)

    save_results_csv(rows, output / "gbm_comparison.csv")
    return rows


def _plot(dataset_name: str, curves: dict[str, list[float]], output) -> None:
    """Staged test error of the three boosters on one dataset."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    for model_name, errors in curves.items():
        axis.plot(np.arange(1, len(errors) + 1), errors, label=model_name)
    axis.set(xlabel="Boosting round", ylabel="Test error",
             title=f"Bonus — GBM vs AdaBoost on {dataset_name}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / f"gbm_comparison_{dataset_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point: ``python -m src.experiments.gbm_comparison [--fast]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced sizes")
    args = parser.parse_args()
    run(load_datasets(fast=args.fast, names=COMPARISON_DATASETS), fast=args.fast)


if __name__ == "__main__":
    main()
