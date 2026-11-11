"""Model-independent evaluation of miRNA-target binding predictions.

Takes a predictions file and an answers file; knows nothing about how the
predictions were produced.
"""
from .bootstrap import bootstrap_ci, paired_bootstrap
from .io import EvaluationInputError, align, load_answers, load_metadata, load_predictions
from .metrics import (
    HIGHER_IS_BETTER, METRIC_NAMES, calibration_table, compute_metrics, confusion_counts,
    expected_calibration_error, metrics_by_group, select_threshold,
)

__all__ = [
    "EvaluationInputError", "align", "load_answers", "load_metadata", "load_predictions",
    "HIGHER_IS_BETTER", "METRIC_NAMES", "calibration_table", "compute_metrics", "confusion_counts",
    "expected_calibration_error", "metrics_by_group", "select_threshold",
    "bootstrap_ci", "paired_bootstrap",
]
