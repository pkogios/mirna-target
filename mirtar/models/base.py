"""Common interface of every predictor.

    model.fit(train, validation=None)   -> self
    model.predict_proba(table)          -> probabilities of label 1, one per row

``train`` / ``validation`` / ``table`` are DataFrames with columns mirna, target
(and label for fitting). ``validation`` is used only for early stopping. After
such a fit ``model.best_rounds`` holds the selected number of epochs / trees, so
that the model can be refitted on other data for a fixed number of rounds
(``rounds=...``) without needing a validation set, e.g. for out-of-fold predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PairModel:
    best_rounds: int | None = None

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame | None = None) -> "PairModel":
        raise NotImplementedError

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
