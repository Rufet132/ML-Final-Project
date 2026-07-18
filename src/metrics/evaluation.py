"""Evaluation metrics implemented from scratch with NumPy.

Provides the three metrics the experimental study reports for every
classifier: accuracy, macro-averaged F1, and ROC-AUC (binary, plus a
macro one-vs-rest extension for multi-class problems). sklearn.metrics
is deliberately not used here so the whole evaluation pipeline stays
self-contained; the unit tests compare these implementations against
sklearn as an external reference.
"""

from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that exactly match the true labels.

    Args:
        y_true: True labels, shape ``(n_samples,)``.
        y_pred: Predicted labels, shape ``(n_samples,)``.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    y_true, y_pred = _validate_labels(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1: the unweighted mean of per-class F1 scores.

    Uses the identity ``F1 = 2 TP / (2 TP + FP + FN)`` per class, where
    the denominator equals (samples predicted as the class) + (samples
    truly of the class). Classes with an empty denominator contribute 0,
    matching sklearn's ``zero_division=0`` convention. Macro averaging
    weighs every class equally, which is what makes this metric
    informative on the imbalanced datasets required by the brief.

    Args:
        y_true: True labels, shape ``(n_samples,)``.
        y_pred: Predicted labels, shape ``(n_samples,)``.

    Returns:
        Macro F1 in ``[0, 1]``.
    """
    y_true, y_pred = _validate_labels(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    scores = np.zeros(classes.shape[0], dtype=float)
    for index, label in enumerate(classes):
        true_positive = float(np.sum((y_pred == label) & (y_true == label)))
        predicted_positive = float(np.sum(y_pred == label))
        actual_positive = float(np.sum(y_true == label))
        denominator = predicted_positive + actual_positive
        if denominator > 0:
            scores[index] = 2.0 * true_positive / denominator
    return float(scores.mean()) if classes.size else 0.0


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve.

    Binary input (``y_score`` 1-D, scores for the positive class): AUC is
    computed with the rank-statistic identity
    ``AUC = U / (N_pos * N_neg)`` where ``U`` is the Mann-Whitney U
    statistic, using midranks so tied scores are handled exactly like
    the trapezoidal ROC construction.

    Multi-class input (``y_score`` 2-D with one column per class in
    sorted label order): returns the macro average of one-vs-rest binary
    AUCs, as required for the project's multi-class datasets.

    Args:
        y_true: True labels, shape ``(n_samples,)``.
        y_score: Positive-class scores ``(n_samples,)`` for binary, or
            class-probability matrix ``(n_samples, n_classes)``.

    Returns:
        AUC in ``[0, 1]``.

    Raises:
        ValueError: If only one class is present (AUC is undefined) or
            shapes are inconsistent.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim not in (1, 2):
        raise ValueError("y_score must be a 1-D score array or 2-D probability matrix")
    if y_true.ndim != 1 or y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have one entry per sample")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score must contain only finite values")

    classes = np.unique(y_true)
    if classes.size < 2:
        raise ValueError("ROC-AUC is undefined with fewer than 2 classes present")

    if y_score.ndim == 1:
        if classes.size != 2:
            raise ValueError("1-D y_score requires exactly 2 classes; pass a "
                             "probability matrix for multi-class AUC")
        return _binary_auc(y_true == classes[1], y_score)

    if y_score.shape[1] != classes.size:
        raise ValueError("y_score must have one column per class present in y_true")
    aucs = [_binary_auc(y_true == label, y_score[:, index])
            for index, label in enumerate(classes)]
    return float(np.mean(aucs))


def _binary_auc(is_positive: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC for one positive class, with midranks for ties."""
    n_positive = int(np.sum(is_positive))
    n_negative = is_positive.shape[0] - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC-AUC needs at least one positive and one negative sample")
    ranks = _midranks(scores)
    positive_rank_sum = float(ranks[is_positive].sum())
    u_statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return float(u_statistic / (n_positive * n_negative))


def _midranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks where tied values all receive their average rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=float)
    sorted_values = values[order]
    boundaries = np.flatnonzero(np.diff(sorted_values) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [values.shape[0]]])
    for start, end in zip(starts, ends):
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
    return ranks


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray,
                           y_proba: np.ndarray | None = None) -> dict[str, float]:
    """Bundle the report's three standard metrics into one dictionary.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Optional probability matrix ``(n_samples, n_classes)``;
            when given, ROC-AUC is included (positive-class column for
            binary problems, macro one-vs-rest for multi-class).

    Returns:
        Dict with keys ``accuracy``, ``f1_macro`` and, when probabilities
        are supplied, ``roc_auc``.
    """
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_macro(y_true, y_pred),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            summary["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            summary["roc_auc"] = roc_auc_score(y_true, y_proba)
    return summary


def _validate_labels(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce label arrays and check they are 1-D with matching length."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("labels must be 1-dimensional arrays")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.shape[0] == 0:
        raise ValueError("labels must not be empty")
    return y_true, y_pred
