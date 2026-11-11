"""Train a model and write its validation predictions.

    python -m mirtar.train --model m1_features_gbm                 # required 20% stratified split
    python -m mirtar.train --model m3_complementarity_cnn --split group --seeds 0 1 2
    python -m mirtar.train --model m1_features_gbm --oof 5 --save  # also out-of-fold predictions + saved model

Outputs (under results/<split>/):
    validation_answers.csv, validation_metadata.csv        reference labels and grouping columns
    oof_answers.csv                                        labels of the training part (for stacking)
    predictions/<model>/seed<k>/validation_predictions.csv
    predictions/<model>/seed<k>/oof_predictions.csv        with --oof
    predictions/<model>/seed<k>/run.json                   settings, selected rounds, time
Saved models go to artifacts/<split>/ (not in git).

This script never reads the test file.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import DATA_DIR, load_train
from .features import site_type
from .models import build_model, load_config
from .splits import SPLITS, cross_validation_folds, split_train_validation

ROOT = Path(__file__).resolve().parents[1]


def write_predictions(path: Path, table: pd.DataFrame, probabilities: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": table["sample_id"].to_numpy(), "probability": probabilities}).to_csv(path, index=False)


def write_references(out: Path, train: pd.DataFrame, validation: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    validation[["sample_id", "label"]].to_csv(out / "validation_answers.csv", index=False)
    train[["sample_id", "label"]].to_csv(out / "oof_answers.csv", index=False)
    pd.DataFrame({
        "sample_id": validation["sample_id"],
        "mirna": validation["mirna"],
        "mirna_length": validation["mirna"].str.len(),
        "site_type": [site_type(m, t) for m, t in zip(validation["mirna"], validation["target"])],
    }).to_csv(out / "validation_metadata.csv", index=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="name in configs/models.yaml")
    parser.add_argument("--split", default="stratified", choices=SPLITS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="model seeds (the data split is fixed)")
    parser.add_argument("--oof", type=int, default=0, metavar="K", help="also write K-fold out-of-fold predictions for the training part")
    parser.add_argument("--save", action="store_true", help="save the fitted model to artifacts/")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--results-dir", default=ROOT / "results")
    parser.add_argument("--limit", type=int, help="use only the first N training pairs (smoke test)")
    parser.add_argument("--threads", type=int, help="CPU threads for neural models")
    args = parser.parse_args(argv)

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    config = load_config()
    table = load_train(args.data_dir)
    if args.limit:
        table = table.groupby("label").head(args.limit // 2)
    train, validation = split_train_validation(table, args.split)
    out = Path(args.results_dir) / args.split
    write_references(out, train, validation)
    print(f"{args.model} | split={args.split} | train={len(train)} validation={len(validation)}")

    for seed in args.seeds:
        run_dir = out / "predictions" / args.model / f"seed{seed}"
        started = time.time()
        model = build_model(args.model, seed=seed, config=config).fit(train, validation)
        probabilities = model.predict_proba(validation)
        write_predictions(run_dir / "validation_predictions.csv", validation, probabilities)
        auc = roc_auc_score(validation["label"], probabilities)
        print(f"seed {seed}: rounds={model.best_rounds} validation ROC-AUC={auc:.4f} ({time.time() - started:.0f}s)")
        run = {"model": args.model, "split": args.split, "seed": seed, "settings": config[args.model],
               "best_rounds": model.best_rounds, "n_train": len(train), "n_validation": len(validation),
               "fit_seconds": round(time.time() - started, 1), "history": getattr(model, "history", [])}

        if args.oof:
            oof = np.full(len(train), np.nan)
            for fold, (fit_rows, held_rows) in enumerate(cross_validation_folds(train, args.split, args.oof)):
                fold_model = build_model(args.model, seed=seed, rounds=model.best_rounds, config=config, **(
                    {"verbose": False} if config[args.model]["kind"] == "neural" else {}))
                fold_model.fit(train.iloc[fit_rows])
                oof[held_rows] = fold_model.predict_proba(train.iloc[held_rows])
                print(f"  fold {fold}: ROC-AUC={roc_auc_score(train['label'].iloc[held_rows], oof[held_rows]):.4f}", flush=True)
            write_predictions(run_dir / "oof_predictions.csv", train, oof)
            run["oof_folds"] = args.oof

        if args.save:
            artifact = ROOT / "artifacts" / args.split / f"{args.model}_seed{seed}.pkl"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            with open(artifact, "wb") as handle:
                pickle.dump(model, handle)
            print(f"  saved {artifact.relative_to(ROOT)}")
        (run_dir / "run.json").write_text(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
