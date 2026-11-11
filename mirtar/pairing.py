"""Shared sequence encoding and base-pairing rules.

Sequences are stored 5'->3' with T instead of U. A miRNA binds its target
antiparallel: miRNA position i pairs with a target position, i+1 with the
position *before* it. We therefore reverse the target, so that a stretch of
consecutive base pairs lies on a diagonal of the (miRNA x reversed target) grid.
"""
from __future__ import annotations

import numpy as np

from .data import MAX_MIRNA_LENGTH, TARGET_LENGTH

PAD, A, C, G, T = 0, 1, 2, 3, 4
TOKEN = {"A": A, "C": C, "G": G, "T": T}

# Pair classes of (miRNA base, target base)
PAIR_PAD, PAIR_GC, PAIR_AU, PAIR_GU, PAIR_MISMATCH = 0, 1, 2, 3, 4
PAIR_NAMES = ["pad", "G:C", "A:U", "G:U wobble", "mismatch"]

PAIR_TABLE = np.full((5, 5), PAIR_MISMATCH, dtype=np.int64)
PAIR_TABLE[PAD, :] = PAIR_TABLE[:, PAD] = PAIR_PAD
PAIR_TABLE[G, C] = PAIR_TABLE[C, G] = PAIR_GC
PAIR_TABLE[A, T] = PAIR_TABLE[T, A] = PAIR_AU
PAIR_TABLE[G, T] = PAIR_TABLE[T, G] = PAIR_GU


def encode_sequence(sequence: str) -> np.ndarray:
    return np.fromiter((TOKEN[base] for base in sequence), dtype=np.int64, count=len(sequence))


def encode_table(table) -> tuple[np.ndarray, np.ndarray]:
    """Token arrays for a table of pairs.

    Returns (mirna, target_reversed): miRNAs right-padded with 0 to 28 columns,
    targets reversed (3'->5') in 50 columns.
    """
    mirna = np.zeros((len(table), MAX_MIRNA_LENGTH), dtype=np.int64)
    target = np.zeros((len(table), TARGET_LENGTH), dtype=np.int64)
    for row, (m, t) in enumerate(zip(table["mirna"], table["target"])):
        if not 1 <= len(m) <= MAX_MIRNA_LENGTH or len(t) != TARGET_LENGTH:
            raise ValueError("Unexpected sequence length; inputs are never truncated silently.")
        mirna[row, :len(m)] = encode_sequence(m)
        target[row] = encode_sequence(t)[::-1]
    return mirna, target


def pair_grid(mirna_tokens: np.ndarray, target_reversed_tokens: np.ndarray) -> np.ndarray:
    """Pair class of every (miRNA position, reversed target position)."""
    return PAIR_TABLE[mirna_tokens[..., :, None], target_reversed_tokens[..., None, :]]
