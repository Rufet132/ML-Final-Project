"""Experiment 4 — Head-to-head comparison at fixed resources.

Stratified k-fold cross-validation of four models on every dataset:
our single unpruned DecisionTree, AdaBoost (100 stumps), our Random
Forest (100 trees), and sklearn's RandomForestClassifier as an external
reference. Reports mean +/- standard deviation of accuracy, macro F1 and
ROC-AUC across folds, plus a box plot of the fold accuracies.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SkRandomForest
from sklearn.model_selection import StratifiedKFold

from src.bagging.random_forest import RandomForestClassifier
from src.boosting.adaboost import AdaBoostClassifier
from src.experiments.utils import (
    Dataset,
    RANDOM_SEED,
    default_n_jobs,
    Timer,
    figure_dir,
    load_datasets,
    save_results_csv,
)
from src.metrics.evaluation import classification_summary
from src.trees.decision_tree import DecisionTree
from src.utils.preprocessing import StandardScaler

#: Ensemble size fixed by the brief.
N_ESTIMATORS_FULL: int = 100
N_ESTIMATORS_FAST: int = 20

#: Cross-validation folds (brief: 5).
N_FOLDS_FULL: int = 5
N_FOLDS_FAST: int = 3

#: Worker processes for our forest (ML_N_JOBS overrides).
N_JOBS: int = default_n_jobs()


def _model_factory(dataset: Dataset, n_estimators: int, seed: int) -> dict:
    """Build the four competing models for one dataset."""
    class_weight = "balanced" if dataset.imbalanced else None
    return {
        "decision_tree": lambda: DecisionTree(random_state=seed),
        "adaboost": lambda: AdaBoostClassifier(n_estimators=n_estimators,
                                               random_state=seed),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=n_estimators, max_depth=dataset.ensemble_max_depth,
            n_jobs=N_JOBS, random_state=seed, class_weight=class_weight),
        "sklearn_rf": lambda: SkRandomForest(
            n_estimators=n_estimators, max_depth=dataset.ensemble_max_depth,
            random_state=seed, class_weight=class_weight),
    }


def _fold_metrics(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit one model on one fold and return its metric summary."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        summary = classification_summary(y_test, y_pred, model.predict_proba(X_test))
    except ValueError:
        # A fold can miss an extremely rare class, leaving AUC undefined.
        summary = classification_summary(y_test, y_pred)
        summary["roc_auc"] = float("nan")
    return summary


def run(datasets: list[Dataset], fast: bool = False,
        seed: int = RANDOM_SEED) -> list[dict]:
    """Cross-validate all four models on every dataset."""
    output = figure_dir("head_to_head")
    n_estimators = N_ESTIMATORS_FAST if fast else N_ESTIMATORS_FULL
    n_folds = N_FOLDS_FAST if fast else N_FOLDS_FULL
    rows: list[dict] = []
    fold_accuracies: dict[tuple[str, str], list[float]] = {}

    for dataset in datasets:
        print(f"  head-to-head on {dataset.name} "
              f"({n_folds}-fold CV, {n_estimators} estimators)")
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        factory = _model_factory(dataset, n_estimators, seed)
        per_model: dict[str, list[dict]] = {name: [] for name in factory}

        for fold, (train_index, test_index) in enumerate(
                splitter.split(dataset.X, dataset.y)):
            scaler = StandardScaler().fit(dataset.X[train_index])
            X_train = scaler.transform(dataset.X[train_index])
            X_test = scaler.transform(dataset.X[test_index])
            y_train, y_test = dataset.y[train_index], dataset.y[test_index]

            for model_name, build in factory.items():
                with Timer(f"fold {fold + 1}/{n_folds} {model_name}"):
                    summary = _fold_metrics(build(), X_train, y_train, X_test, y_test)
                per_model[model_name].append(summary)
                fold_accuracies.setdefault((dataset.name, model_name), []).append(
                    summary["accuracy"])

        for model_name, summaries in per_model.items():
            row = {"dataset": dataset.name, "model": model_name,
                   "n_folds": n_folds, "n_estimators": n_estimators}
            for metric in ("accuracy", "f1_macro", "roc_auc"):
                values = np.array([s[metric] for s in summaries], dtype=float)
                row[f"{metric}_mean"] = float(np.nanmean(values))
                row[f"{metric}_std"] = float(np.nanstd(values))
            rows.append(row)
            print(f"    {model_name}: acc={row['accuracy_mean']:.4f}"
                  f"+/-{row['accuracy_std']:.4f}  "
                  f"f1={row['f1_macro_mean']:.4f}  auc={row['roc_auc_mean']:.4f}")

        _plot(dataset.name, {name: fold_accuracies[(dataset.name, name)]
                             for name in factory}, output)

    save_results_csv(rows, output / "head_to_head.csv")
    return rows


def _plot(dataset_name: str, accuracies: dict[str, list[float]], output) -> None:
    """Box plot of per-fold accuracies for the four models."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(list(accuracies.values()), tick_labels=list(accuracies.keys()))
    axis.set(ylabel="Fold accuracy",
             title=f"Experiment 4 — Head-to-head on {dataset_name}")
    axis.grid(alpha=0.25, axis="y")
    plt.setp(axis.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(output / f"head_to_head_{dataset_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point: ``python -m src.experiments.head_to_head [--fast]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced sizes")
    args = parser.parse_args()
    run(load_datasets(fast=args.fast), fast=args.fast)


if __name__ == "__main__":
    main()
