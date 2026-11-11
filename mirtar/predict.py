"""Predict with a saved model on any file of miRNA-target pairs.

    python -m mirtar.predict --model artifacts/stratified/m1_features_gbm_seed0.pkl \
        --data data/AGO2_CLASH_Hejret2023_train.tsv.gz --prefix train --out my_predictions.csv

The input needs the columns noncodingRNA and gene (tab separated, optionally
gzipped). A label column, if present, is ignored: predictions never depend on it.
Output: sample_id,probability with sample_id = <prefix>_<row number>.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from .data import validate_pairs


def read_inputs(path, prefix: str) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", compression="infer", usecols=["noncodingRNA", "gene"])
    table = table.rename(columns={"noncodingRNA": "mirna", "gene": "target"})[["mirna", "target"]]
    validate_pairs(table)
    table.insert(0, "sample_id", [f"{prefix}_{i}" for i in table.index])
    return table


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="file written by `python -m mirtar.train --save`")
    parser.add_argument("--data", required=True)
    parser.add_argument("--prefix", required=True, help="sample_id prefix, e.g. train or test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    with open(args.model, "rb") as handle:  # only load model files you created yourself
        model = pickle.load(handle)
    table = read_inputs(args.data, args.prefix)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": table["sample_id"], "probability": model.predict_proba(table)}).to_csv(args.out, index=False)
    print(f"wrote {len(table)} predictions to {args.out}")


if __name__ == "__main__":
    main()
