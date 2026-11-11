"""Train/validation splits of the official training file.

``stratified``  the split required by the brief: 20% per label, identical to the
                example notebook (same seed, same rows), so results are comparable.
``group``       a stricter split: all pairs whose miRNA shares the same seed
                (nucleotides 2-8, a proxy for the miRNA family computed from the
                sequence alone) are kept on the same side. Validation then
                contains only miRNA families never seen in training.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SPLITS = ("stratified", "group")


def seed_family(mirna: pd.Series) -> pd.Series:
    return mirna.str[1:8]


def split_train_validation(table: pd.DataFrame, split: str = "stratified", seed: int = 42):
    """Return (train, validation) as views of ``table`` with the original index."""
    if split == "stratified":
        validation = table.groupby("label").sample(frac=0.20, random_state=seed)
        return table.drop(validation.index), validation.sort_index()
    if split == "group":
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_rows, validation_rows = next(folds.split(table, table["label"], seed_family(table["mirna"])))
        return table.iloc[train_rows], table.iloc[validation_rows]
    raise ValueError(f"Unknown split '{split}', expected one of {SPLITS}")


def cross_validation_folds(table: pd.DataFrame, split: str = "stratified", n_folds: int = 5, seed: int = 42):
    """Yield (fit_rows, held_out_rows) positions for out-of-fold predictions."""
    if split == "group":
        folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        yield from folds.split(table, table["label"], seed_family(table["mirna"]))
    else:
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        yield from folds.split(table, table["label"])
