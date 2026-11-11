"""M4: combine the predictions of several models.

    python -m mirtar.ensemble --models m1_features_gbm m2_cross_attention m3_complementarity_cnn

Two combinations are written, each as an ordinary predictions file:

    ensemble_mean   average of the base probabilities (nothing is fitted)
    ensemble_stack  logistic regression on the base models' logits. It is fitted on
                    out-of-fold predictions for the training part, so it never sees
                    a prediction that a base model made for a pair it was trained on,
                    and never sees a validation label.

The fitted weights are stored in stack.json and reused unchanged for other data
(``combine``), e.g. the final test predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .splits import SPLITS

ROOT = Path(__file__).resolve().parents[1]


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def read_aligned(paths, sample_ids) -> np.ndarray:
    """Probabilities of several prediction files as columns, rows in the order of ``sample_ids``."""
    columns = []
    for path in paths:
        table = pd.read_csv(path, dtype={"sample_id": str}).set_index("sample_id")
        columns.append(table.loc[sample_ids, "probability"].to_numpy())
    return np.column_stack(columns)


def fit_stack(oof: np.ndarray, labels: np.ndarray, models) -> dict:
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(logit(oof), labels)
    return {"models": list(models), "weights": meta.coef_[0].tolist(), "intercept": float(meta.intercept_[0])}


def combine(stack: dict, probabilities: np.ndarray) -> np.ndarray:
    score = logit(probabilities) @ np.asarray(stack["weights"]) + stack["intercept"]
    return 1 / (1 + np.exp(-score))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--split", default="stratified", choices=SPLITS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default=ROOT / "results")
    parser.add_argument("--suffix", default="", help="appended to the output names, e.g. _all")
    args = parser.parse_args(argv)

    base = Path(args.results_dir) / args.split
    run = lambda model, name: base / "predictions" / model / f"seed{args.seed}" / name
    oof_answers = pd.read_csv(base / "oof_answers.csv", dtype={"sample_id": str})
    validation_ids = pd.read_csv(base / "validation_answers.csv", dtype={"sample_id": str})["sample_id"].to_numpy()

    oof = read_aligned([run(m, "oof_predictions.csv") for m in args.models], oof_answers["sample_id"].to_numpy())
    validation = read_aligned([run(m, "validation_predictions.csv") for m in args.models], validation_ids)
    stack = fit_stack(oof, oof_answers["label"].to_numpy(), args.models)
    print("stack weights:", {m: round(w, 3) for m, w in zip(args.models, stack["weights"])}, "intercept", round(stack["intercept"], 3))

    for name, probabilities in (("ensemble_mean", validation.mean(axis=1)), ("ensemble_stack", combine(stack, validation))):
        out = run(name + args.suffix, "validation_predictions.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample_id": validation_ids, "probability": probabilities}).to_csv(out, index=False)
        print("wrote", out.relative_to(Path(args.results_dir).parent) if Path(args.results_dir).is_relative_to(ROOT) else out)
    run("ensemble_stack" + args.suffix, "stack.json").write_text(json.dumps(stack, indent=2))
    (run("ensemble_mean" + args.suffix, "stack.json")).write_text(json.dumps({"models": args.models}, indent=2))


if __name__ == "__main__":
    main()
