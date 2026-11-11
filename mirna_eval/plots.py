"""Figures: ROC, precision-recall, calibration and confusion matrix."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skm

from .metrics import calibration_table, confusion_counts


def plot_curves(labels, predictions: dict, path, title: str = "") -> None:
    """ROC, precision-recall and reliability curves for one or several models.

    ``predictions`` maps a model name to its probability array (same sample order as ``labels``).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, probabilities in predictions.items():
        fpr, tpr, _ = skm.roc_curve(labels, probabilities)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {skm.roc_auc_score(labels, probabilities):.3f})")
        precision, recall, _ = skm.precision_recall_curve(labels, probabilities)
        axes[1].plot(recall, precision, label=f"{name} (AP {skm.average_precision_score(labels, probabilities):.3f})")
        table = calibration_table(labels, probabilities)
        axes[2].plot(table["mean_probability"], table["positive_rate"], marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].axhline(np.mean(labels), color="black", linestyle="--", linewidth=0.8)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall", ylim=(0, 1.02))
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title="Calibration")
    for ax in axes:
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(labels, probabilities, threshold: float, path, title: str = "") -> None:
    counts = confusion_counts(labels, probabilities, threshold)
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"],
           title=title or f"Confusion matrix (threshold {threshold:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
