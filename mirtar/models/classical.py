"""Models without neural networks: M1 (features + gradient boosting) and simple references."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features import build_features, pair_features
from .base import PairModel


class FeatureGBM(PairModel):
    """M1: biologically motivated pair features + LightGBM."""

    def __init__(self, seed: int = 0, rounds: int | None = None, drop_groups=(), max_rounds: int = 2000,
                 patience: int = 100, **params):
        self.seed, self.rounds, self.drop_groups = seed, rounds, tuple(drop_groups)
        self.max_rounds, self.patience, self.params = max_rounds, patience, params

    def features(self, table: pd.DataFrame) -> pd.DataFrame:
        return build_features(table, self.drop_groups)

    def fit(self, train, validation=None):
        early_stopping = validation is not None and self.rounds is None
        self.model = lgb.LGBMClassifier(
            n_estimators=self.max_rounds if early_stopping else (self.rounds or 300),
            random_state=self.seed, verbose=-1, n_jobs=4, **self.params,
        )
        if early_stopping:
            self.model.fit(
                self.features(train), train["label"],
                eval_X=self.features(validation), eval_y=validation["label"], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(self.patience, verbose=False)],
            )
            self.best_rounds = int(self.model.best_iteration_)
        else:
            self.model.fit(self.features(train), train["label"])
            self.best_rounds = int(self.model.n_estimators)
        return self

    def predict_proba(self, table):
        return self.model.predict_proba(self.features(table))[:, 1]


class SeedRule(PairModel):
    """No learning at all: the score is the best seed complementarity (the textbook rule)."""

    def __init__(self, seed: int = 0, rounds=None):
        self.best_rounds = 0

    def fit(self, train, validation=None):
        return self

    def predict_proba(self, table):
        scores = [pair_features(m, t)["seed_score_max"] for m, t in zip(table["mirna"], table["target"])]
        return np.asarray(scores) / 7.0


class KmerLogistic(PairModel):
    """Logistic regression on k-mer counts of one or both sequences.

    With ``use='mirna'`` or ``use='target'`` the model sees a single sequence and
    cannot know anything about the interaction. Any performance above chance
    then measures a shortcut in how the negative pairs were constructed.
    With ``use='both'`` the counts are concatenated (still no interaction terms).
    """

    def __init__(self, seed: int = 0, rounds=None, use: str = "both", k: int = 3, C: float = 0.1):
        self.seed, self.use, self.k, self.C = seed, use, k, C
        self.best_rounds = 0

    def _text(self, table):
        if self.use == "mirna":
            return table["mirna"]
        if self.use == "target":
            return table["target"]
        return table["mirna"].str.lower() + " " + table["target"]  # separate vocabularies

    def fit(self, train, validation=None):
        self.model = make_pipeline(
            CountVectorizer(analyzer="char", ngram_range=(1, self.k), lowercase=False),
            StandardScaler(with_mean=False),
            LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed),
        )
        self.model.fit(self._text(train), train["label"])
        return self

    def predict_proba(self, table):
        return self.model.predict_proba(self._text(table))[:, 1]
