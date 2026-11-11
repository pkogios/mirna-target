"""Uncertainty of metrics and of differences between models (bootstrap over samples)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import METRIC_NAMES, compute_metrics


def _resample_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample with replacement inside each class, so both classes are always present."""
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    return np.concatenate([
        rng.choice(positives, size=len(positives), replace=True),
        rng.choice(negatives, size=len(negatives), replace=True),
    ])


def bootstrap_ci(labels, probabilities, threshold: float = 0.5, n_boot: int = 1000,
                 confidence: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Point estimate and percentile confidence interval for every metric."""
    labels, probabilities = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    point = compute_metrics(labels, probabilities, threshold)
    samples = np.empty((n_boot, len(METRIC_NAMES)))
    for b in range(n_boot):
        idx = _resample_indices(labels, rng)
        values = compute_metrics(labels[idx], probabilities[idx], threshold)
        samples[b] = [values[name] for name in METRIC_NAMES]
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha], axis=0)
    return pd.DataFrame({
        "metric": METRIC_NAMES,
        "value": [point[name] for name in METRIC_NAMES],
        "ci_low": low, "ci_high": high,
    })


def paired_bootstrap(labels, probabilities_a, probabilities_b, threshold_a: float = 0.5,
                     threshold_b: float = 0.5, metrics=("roc_auc", "average_precision", "balanced_accuracy", "mcc"),
                     n_boot: int = 2000, confidence: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Is model A different from model B on the *same* samples?

    Both models are scored on identical resamples, so the difference (A - B) is
    not blurred by which examples happened to be drawn. The two-sided p-value is
    the bootstrap probability that the difference has the opposite sign.
    """
    labels = np.asarray(labels, dtype=int)
    a, b = np.asarray(probabilities_a, dtype=float), np.asarray(probabilities_b, dtype=float)
    rng = np.random.default_rng(seed)
    point_a, point_b = compute_metrics(labels, a, threshold_a), compute_metrics(labels, b, threshold_b)
    deltas = np.empty((n_boot, len(metrics)))
    for i in range(n_boot):
        idx = _resample_indices(labels, rng)
        va, vb = compute_metrics(labels[idx], a[idx], threshold_a), compute_metrics(labels[idx], b[idx], threshold_b)
        deltas[i] = [va[m] - vb[m] for m in metrics]
    alpha = (1 - confidence) / 2
    low, high = np.quantile(deltas, [alpha, 1 - alpha], axis=0)
    below = ((deltas <= 0).sum(axis=0) + 1) / (n_boot + 1)
    above = ((deltas >= 0).sum(axis=0) + 1) / (n_boot + 1)
    return pd.DataFrame({
        "metric": list(metrics),
        "a": [point_a[m] for m in metrics], "b": [point_b[m] for m in metrics],
        "difference": [point_a[m] - point_b[m] for m in metrics],
        "ci_low": low, "ci_high": high,
        "p_value": np.minimum(1.0, 2 * np.minimum(below, above)),
    })
