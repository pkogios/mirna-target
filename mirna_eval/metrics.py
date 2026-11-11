"""Metrics for binary predictions given as probabilities.

Two families:

* threshold-free (ranking / probability quality): ROC-AUC, average precision,
  log loss, Brier score, expected calibration error;
* thresholded (decision quality): accuracy, balanced accuracy, precision,
  recall, specificity, F1, Matthews correlation coefficient.

The threshold must be chosen on validation data (``select_threshold``) and then
reused unchanged on the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm

HIGHER_IS_BETTER = {
    "roc_auc": True, "average_precision": True, "log_loss": False, "brier": False, "ece": False,
    "accuracy": True, "balanced_accuracy": True, "precision": True, "recall": True,
    "specificity": True, "f1": True, "mcc": True,
}
METRIC_NAMES = list(HIGHER_IS_BETTER)


def log_loss(labels, probabilities, eps: float = 1e-7) -> float:
    p = np.clip(probabilities, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def brier_score(labels, probabilities) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def calibration_table(labels, probabilities, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins: mean predicted probability vs observed positive rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append({
                "bin_low": edges[b], "bin_high": edges[b + 1], "n": int(mask.sum()),
                "mean_probability": float(probabilities[mask].mean()),
                "positive_rate": float(labels[mask].mean()),
            })
    return pd.DataFrame(rows)


def expected_calibration_error(labels, probabilities, n_bins: int = 10) -> float:
    table = calibration_table(labels, probabilities, n_bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["mean_probability"] - table["positive_rate"]).abs()).sum())


def compute_metrics(labels, probabilities, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    both_classes = len(np.unique(labels)) == 2
    return {
        "roc_auc": float(skm.roc_auc_score(labels, probabilities)) if both_classes else float("nan"),
        "average_precision": float(skm.average_precision_score(labels, probabilities)) if both_classes else float("nan"),
        "log_loss": log_loss(labels, probabilities),
        "brier": brier_score(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": float(mcc),
    }


def confusion_counts(labels, probabilities, threshold: float = 0.5) -> dict:
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    labels = np.asarray(labels, dtype=int)
    return {
        "tp": int(((predicted == 1) & (labels == 1)).sum()),
        "fp": int(((predicted == 1) & (labels == 0)).sum()),
        "tn": int(((predicted == 0) & (labels == 0)).sum()),
        "fn": int(((predicted == 0) & (labels == 1)).sum()),
    }


def select_threshold(labels, probabilities, criterion: str = "balanced_accuracy") -> float:
    """Pick the decision threshold that maximises ``criterion`` on validation data."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if criterion == "balanced_accuracy":  # equivalent to Youden's J
        fpr, tpr, thresholds = skm.roc_curve(labels, probabilities)
        score = (tpr + 1 - fpr) / 2
        score, thresholds = score[1:], thresholds[1:]  # drop the artificial "inf" threshold
    elif criterion == "f1":
        precision, recall, thresholds = skm.precision_recall_curve(labels, probabilities)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.nan_to_num(2 * precision * recall / (precision + recall))[:-1]
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    return float(thresholds[int(np.argmax(score))])


def metrics_by_group(labels, probabilities, groups, threshold: float = 0.5, min_size: int = 20) -> pd.DataFrame:
    """Metrics inside each group (e.g. per miRNA). Small or single-class groups are skipped."""
    labels, probabilities, groups = np.asarray(labels), np.asarray(probabilities), np.asarray(groups)
    rows = []
    for value in pd.unique(groups):
        mask = groups == value
        if mask.sum() < min_size or len(np.unique(labels[mask])) < 2:
            continue
        values = compute_metrics(labels[mask], probabilities[mask], threshold)
        rows.append({"group": value, "n": int(mask.sum()), "positive_rate": float(labels[mask].mean()), **values})
    table = pd.DataFrame(rows)
    return table.sort_values("n", ascending=False).reset_index(drop=True) if len(table) else table
