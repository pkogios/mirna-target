"""Loading the miRBench AGO2_CLASH_Hejret2023 files.

Only the two sequences and the label are ever read. The other columns
(``Nunique`` in particular) reveal the label and must not reach a model.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ZENODO_URL = "https://zenodo.org/records/20540907/files/"
TRAIN_FILE = "AGO2_CLASH_Hejret2023_train.tsv.gz"
TEST_FILE = "AGO2_CLASH_Hejret2023_test.tsv.gz"
INPUT_COLUMNS = ["noncodingRNA", "gene", "label"]
MAX_MIRNA_LENGTH = 28
TARGET_LENGTH = 50
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_pairs(path, prefix: str) -> pd.DataFrame:
    """Return columns sample_id, mirna, target, label.

    ``sample_id`` is ``<prefix>_<row number in the original file>`` (``train_17``,
    ``test_3``), the same convention as the course example notebook.
    """
    table = pd.read_csv(path, sep="\t", compression="infer", usecols=INPUT_COLUMNS)
    table = table.rename(columns={"noncodingRNA": "mirna", "gene": "target"})[["mirna", "target", "label"]]
    validate_pairs(table)
    table.insert(0, "sample_id", [f"{prefix}_{i}" for i in table.index])
    return table


def validate_pairs(table: pd.DataFrame) -> None:
    if not table["mirna"].str.fullmatch(f"[ACGT]{{1,{MAX_MIRNA_LENGTH}}}").all():
        raise ValueError(f"miRNA sequences must be 1-{MAX_MIRNA_LENGTH} nt of A/C/G/T")
    if not table["target"].str.fullmatch(f"[ACGT]{{{TARGET_LENGTH}}}").all():
        raise ValueError(f"target sequences must be exactly {TARGET_LENGTH} nt of A/C/G/T")
    if "label" in table and not set(table["label"].unique()) <= {0, 1}:
        raise ValueError("labels must be 0 or 1")
    if table.duplicated(["mirna", "target"]).any():
        raise ValueError("duplicated miRNA-target pairs")


def load_train(data_dir=DATA_DIR) -> pd.DataFrame:
    return load_pairs(Path(data_dir) / TRAIN_FILE, "train")
