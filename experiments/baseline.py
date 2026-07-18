"""Experiment 1 — Baseline single tree and stump vs the sklearn reference.

For every dataset: an 80/20 stratified split, then
* our unpruned ``DecisionTree``,
* a depth-1 ``DecisionStump``,
* ``sklearn.tree.DecisionTreeClassifier`` with identical parameters,
* additionally, for the imbalanced dataset, our tree trained with
  balanced class weights (the documented imbalance treatment).

Reports accuracy, macro F1 and ROC-AUC (macro one-vs-rest for
multi-class) and checks the accuracy parity with sklearn (2% rubric).
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.boosting.adaboost import DecisionStump
from src.experiments.utils import (
    Dataset,
    RANDOM_SEED,
    Timer,
    figure_dir,
    load_datasets,
    prepare_split,
    save_results_csv,
)
from src.metrics.evaluation import classification_summary
from src.trees.decision_tree import DecisionTree

#: Accuracy parity requirement against sklearn from the brief.
PARITY_TOLERANCE: float = 0.02


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Per-sample weights making every class carry equal total weight."""
    classes, counts = np.unique(y, return_counts=True)
    per_class = y.shape[0] / (classes.shape[0] * counts.astype(float))
    return per_class[np.searchsorted(classes, y)]


def _evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return one result row with the three standard metrics."""
    y_pred = model.predict(X_test)
    try:
        proba = model.predict_proba(X_test)
        summary = classification_summary(y_test, y_pred, proba)
    except ValueError:
        # A test fold can miss an extremely rare class, leaving AUC undefined.
        summary = classification_summary(y_test, y_pred)
        summary["roc_auc"] = float("nan")
    return {"model": name, **summary}


def run(datasets: list[Dataset], fast: bool = False,
        seed: int = RANDOM_SEED) -> list[dict]:
    """Run the baseline comparison for every dataset.

    Returns:
        One row per (dataset, model) with accuracy/F1/AUC plus the
        sklearn parity gap for the single tree.
    """
    del fast  # single trees are cheap; sizes come from the dataset layer
    output = figure_dir("baseline")
    rows: list[dict] = []

    for dataset in datasets:
        X_train, X_test, y_train, y_test = prepare_split(dataset, random_state=seed)
        print(f"  baseline on {dataset.name} "
              f"(train={X_train.shape[0]}, test={X_test.shape[0]})")

        with Timer("our unpruned tree"):
            ours = DecisionTree(random_state=seed).fit(X_train, y_train)
        with Timer("decision stump"):
            stump = DecisionStump(random_state=seed).fit(X_train, y_train)
        with Timer("sklearn tree"):
            reference = DecisionTreeClassifier(random_state=seed).fit(X_train, y_train)

        results = [
            _evaluate("decision_tree", ours, X_test, y_test),
            _evaluate("decision_stump", stump, X_test, y_test),
            _evaluate("sklearn_tree", reference, X_test, y_test),
        ]

        if dataset.imbalanced:
            with Timer("our tree + balanced class weights"):
                treated = DecisionTree(random_state=seed)
                treated.fit(X_train, y_train,
                            sample_weight=_balanced_sample_weight(y_train))
            results.append(_evaluate("decision_tree_balanced", treated, X_test, y_test))

        parity_gap = abs(results[0]["accuracy"] - results[2]["accuracy"])
        status = "OK" if parity_gap <= PARITY_TOLERANCE else "EXCEEDED"
        print(f"    sklearn parity gap: {parity_gap:.4f} ({status})")

        for row in results:
            row.update({"dataset": dataset.name, "parity_gap": parity_gap})
            rows.append(row)

        _plot(results, dataset.name, output)

    save_results_csv(rows, output / "baseline_metrics.csv")
    return rows


def _plot(results: list[dict], dataset_name: str, output) -> None:
    """Grouped bar chart of the three metrics per model."""
    import matplotlib.pyplot as plt

    metrics = ["accuracy", "f1_macro", "roc_auc"]
    models = [row["model"] for row in results]
    positions = np.arange(len(models))
    width = 0.26

    fig, axis = plt.subplots(figsize=(9, 5))
    for offset, metric in enumerate(metrics):
        values = [row.get(metric, float("nan")) for row in results]
        axis.bar(positions + (offset - 1) * width, values, width, label=metric)
    axis.set_xticks(positions)
    axis.set_xticklabels(models, rotation=15)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title(f"Experiment 1 — Baseline models on {dataset_name}")
    axis.grid(alpha=0.25, axis="y")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / f"baseline_{dataset_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point: ``python -m src.experiments.baseline [--fast]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced sizes")
    args = parser.parse_args()
    run(load_datasets(fast=args.fast), fast=args.fast)


if __name__ == "__main__":
    main()
