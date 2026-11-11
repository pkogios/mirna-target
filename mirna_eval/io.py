"""Reading and validating prediction / answer files.

File formats (same as the course example notebook):

    predictions.csv :  sample_id,probability
    answers.csv     :  sample_id,label

The evaluator never looks at sequences or at any model code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class EvaluationInputError(ValueError):
    """Raised when a predictions/answers file cannot be evaluated safely."""


def _read(path, value_column: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise EvaluationInputError(f"File not found: {path}")
    table = pd.read_csv(path, dtype={"sample_id": str})
    missing = {"sample_id", value_column} - set(table.columns)
    if missing:
        raise EvaluationInputError(f"{path}: missing column(s) {sorted(missing)}")
    if table.empty:
        raise EvaluationInputError(f"{path}: file has no rows")
    if table["sample_id"].isna().any():
        raise EvaluationInputError(f"{path}: empty sample_id values")
    duplicated = table["sample_id"][table["sample_id"].duplicated()].unique()
    if len(duplicated):
        raise EvaluationInputError(
            f"{path}: {len(duplicated)} duplicated sample_id(s), e.g. {list(duplicated[:3])}"
        )
    return table


def load_predictions(path) -> pd.DataFrame:
    table = _read(path, "probability")
    values = pd.to_numeric(table["probability"], errors="coerce")
    if values.isna().any():
        raise EvaluationInputError(f"{path}: {int(values.isna().sum())} missing or non-numeric probabilities")
    if ((values < 0) | (values > 1)).any():
        raise EvaluationInputError(f"{path}: probabilities must lie in [0, 1]")
    return pd.DataFrame({"sample_id": table["sample_id"], "probability": values.astype(float)})


def load_answers(path) -> pd.DataFrame:
    table = _read(path, "label")
    values = pd.to_numeric(table["label"], errors="coerce")
    if values.isna().any() or not set(values.unique()) <= {0, 1}:
        raise EvaluationInputError(f"{path}: labels must be 0 or 1")
    if values.nunique() < 2:
        raise EvaluationInputError(f"{path}: both classes are needed for evaluation")
    return pd.DataFrame({"sample_id": table["sample_id"], "label": values.astype(int)})


def align(predictions: pd.DataFrame, answers: pd.DataFrame):
    """Match predictions to answers by sample_id (order in the files does not matter).

    Every reference sample must have exactly one prediction and there must be no
    predictions for unknown samples; otherwise a model could skip hard examples.
    Returns (sample_ids, labels, probabilities) in the order of the answers file.
    """
    predicted, reference = set(predictions["sample_id"]), set(answers["sample_id"])
    missing, extra = reference - predicted, predicted - reference
    if missing:
        raise EvaluationInputError(
            f"{len(missing)} reference sample(s) have no prediction, e.g. {sorted(missing)[:3]}"
        )
    if extra:
        raise EvaluationInputError(
            f"{len(extra)} prediction(s) have no reference label, e.g. {sorted(extra)[:3]}"
        )
    merged = answers.merge(predictions, on="sample_id", how="left", validate="one_to_one")
    return (
        merged["sample_id"].to_numpy(),
        merged["label"].to_numpy(dtype=int),
        merged["probability"].to_numpy(dtype=float),
    )


def load_metadata(path, sample_ids: np.ndarray) -> pd.DataFrame:
    """Optional per-sample grouping columns (e.g. miRNA, seed type) for breakdowns."""
    table = pd.read_csv(path, dtype={"sample_id": str})
    if "sample_id" not in table.columns:
        raise EvaluationInputError(f"{path}: missing column 'sample_id'")
    table = table.drop_duplicates("sample_id").set_index("sample_id")
    absent = set(sample_ids) - set(table.index)
    if absent:
        raise EvaluationInputError(f"{path}: no metadata for {len(absent)} sample(s)")
    return table.loc[sample_ids].reset_index()
