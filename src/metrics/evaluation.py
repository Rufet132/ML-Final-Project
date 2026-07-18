"""Evaluation metrics implemented with NumPy."""

from __future__ import annotations

import numpy as np


def _validate_targets(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.size == 0:
        raise ValueError("targets must not be empty")
    if true.size != pred.size:
        raise ValueError("targets must have the same length")
    return true, pred


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of exactly matching labels."""
    true, pred = _validate_targets(y_true, y_pred)
    return float(np.mean(true == pred))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return unweighted mean per-class F1 with zero for undefined scores."""
    true, pred = _validate_targets(y_true, y_pred)
    scores: list[float] = []
    for label in np.union1d(true, pred):
        true_positive = np.sum((true == label) & (pred == label))
        false_positive = np.sum((true != label) & (pred == label))
        false_negative = np.sum((true == label) & (pred != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else float(2 * true_positive / denominator))
    return float(np.mean(scores))


def _binary_roc_auc(binary_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary AUC using average ranks, correctly handling ties."""
    positives = int(np.sum(binary_true == 1))
    negatives = binary_true.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC is undefined with fewer than 2 classes")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positive_rank_sum = float(np.sum(ranks[binary_true == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return binary or macro one-vs-rest multiclass ROC-AUC."""
    true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=float)
    if true.size == 0:
        raise ValueError("targets must not be empty")
    classes = np.unique(true)
    if classes.size < 2:
        raise ValueError("ROC-AUC is undefined with fewer than 2 classes")
    if scores.shape[0] != true.size:
        raise ValueError("targets and scores must have the same length")

    if classes.size == 2:
        if scores.ndim == 2:
            if scores.shape[1] != 2:
                raise ValueError("scores must have one column per class")
            scores = scores[:, 1]
        elif scores.ndim != 1:
            raise ValueError("binary scores must be one-dimensional")
        binary_true = (true == classes[1]).astype(int)
        return float(_binary_roc_auc(binary_true, scores))

    if scores.ndim != 2:
        raise ValueError("multiclass ROC-AUC requires a probability matrix")
    if scores.shape[1] != classes.size:
        raise ValueError("scores must have one column per class")
    aucs = [
        _binary_roc_auc((true == label).astype(int), scores[:, index])
        for index, label in enumerate(classes)
    ]
    return float(np.mean(aucs))


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Collect the standard classification metrics used by experiments."""
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_macro(y_true, y_pred),
    }
    if y_proba is not None:
        summary["roc_auc"] = roc_auc_score(y_true, y_proba)
    return summary
