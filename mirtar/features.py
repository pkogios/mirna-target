"""Hand-made, biologically motivated features of a miRNA-target pair (model M1).

Everything is computed from the two sequences only. Feature name prefixes
define groups that can be switched off for ablations:

    seed_  complementarity of the miRNA seed (positions 2-8) to the target
    site_  canonical site types (6mer, 7mer-m8, 7mer-A1, 8mer, offset 6mer) and site context
    ext_   pairing beyond the seed along the same ungapped register (central, 3' supplementary, tail)
    glob_  best pairing found anywhere, without assuming that the seed is involved
    comp_  composition of the two sequences (no interaction information)

Limitation: alignments are ungapped, so bulges and loops are not modelled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pairing import A, C, G, PAIR_AU, PAIR_GC, PAIR_GU, PAIR_TABLE, T, encode_sequence

SEED_OFFSETS = 43  # registers in which miRNA positions 2-8 fit inside the 50-nt target
FEATURE_GROUPS = ("seed", "site", "ext", "glob", "comp")
SITE_TYPES = ("8mer", "7mer-m8", "7mer-A1", "6mer", "offset-6mer", "none")


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _diagonal_runs(mask: np.ndarray) -> np.ndarray:
    """runs[i, j] = number of consecutive True cells ending at (i, j) along the diagonal."""
    runs = np.zeros(mask.shape, dtype=np.int64)
    runs[0] = mask[0]
    runs[:, 0] = mask[:, 0]
    for i in range(1, mask.shape[0]):
        runs[i, 1:] = (runs[i - 1, :-1] + 1) * mask[i, 1:]
    return runs


def _seed_profile(wc: np.ndarray):
    """Per register: Watson-Crick matches in positions 2-8, 2-7 and 3-8."""
    columns = np.arange(SEED_OFFSETS)
    per_position = np.stack([wc[1 + k, k + columns] for k in range(7)])  # 7 x registers
    return per_position.sum(0), per_position[:6].sum(0), per_position[1:].sum(0)


def site_type(mirna: str, target: str) -> str:
    """Strongest canonical seed site of the miRNA in the target (TargetScan hierarchy)."""
    m, t = encode_sequence(mirna), encode_sequence(target)[::-1]
    pairs = PAIR_TABLE[m[:, None], t[None, :]]
    wc = (pairs == PAIR_GC) | (pairs == PAIR_AU)
    seed7, seed6, offset6 = _seed_profile(wc)
    a1 = np.concatenate([[False], t[:SEED_OFFSETS - 1] == A])
    if ((seed7 == 7) & a1).any():
        return "8mer"
    if (seed7 == 7).any():
        return "7mer-m8"
    if ((seed6 == 6) & a1).any():
        return "7mer-A1"
    if (seed6 == 6).any():
        return "6mer"
    if (offset6 == 6).any():
        return "offset-6mer"
    return "none"


def pair_features(mirna: str, target: str) -> dict:
    m, t = encode_sequence(mirna), encode_sequence(target)[::-1]
    length = len(m)
    pairs = PAIR_TABLE[m[:, None], t[None, :]]
    wc = (pairs == PAIR_GC) | (pairs == PAIR_AU)
    gu = pairs == PAIR_GU
    paired = wc | gu
    f = {}

    # --- seed complementarity over all registers -------------------------------------------
    seed7, seed6, offset6 = _seed_profile(wc)
    columns = np.arange(SEED_OFFSETS)
    seed_gu = np.stack([gu[1 + k, k + columns] for k in range(7)]).sum(0)
    score = seed7 + 0.5 * seed_gu
    best = int(np.argmax(score))
    f["seed_wc_max"] = seed7.max()
    f["seed_wc_gu_max"] = (seed7 + seed_gu).max()
    f["seed_score_max"] = score.max()
    f["seed_score_second"] = np.sort(score)[-2]
    f["seed_registers_ge5"] = (seed7 >= 5).sum()
    f["seed_registers_ge6"] = (seed7 >= 6).sum()
    f["seed_gu_at_best"] = seed_gu[best]
    in_seed = pairs[np.arange(1, 8), np.arange(1, 8) + best - 1]
    f["seed_gc_pairs_at_best"] = (in_seed == PAIR_GC).sum()
    f["seed_longest_run_at_best"] = _longest_run((in_seed == PAIR_GC) | (in_seed == PAIR_AU))
    for k in range(7):  # which seed positions are paired in the best register
        f[f"seed_pos{k + 2}_paired"] = int(in_seed[k] in (PAIR_GC, PAIR_AU))

    # --- canonical site types and site context ---------------------------------------------
    a1 = np.concatenate([[False], t[:SEED_OFFSETS - 1] == A])
    f["site_6mer"] = (seed6 == 6).sum()
    f["site_7mer_m8"] = (seed7 == 7).sum()
    f["site_7mer_a1"] = ((seed6 == 6) & a1).sum()
    f["site_8mer"] = ((seed7 == 7) & a1).sum()
    f["site_offset_6mer"] = (offset6 == 6).sum()
    f["site_a1_at_best"] = int(a1[best])
    f["site_position"] = (49 - best) / 49  # where the seed match ends in the 5'->3' target
    flank = t[max(0, best - 10): best + 18]
    f["site_flank_au"] = np.isin(flank, (A, T)).mean()

    # --- pairing beyond the seed, same ungapped register -----------------------------------
    shift = best - 1
    rows = np.arange(max(0, -shift), min(length, 50 - shift))
    diagonal = pairs[rows, rows + shift]
    diagonal_wc = (diagonal == PAIR_GC) | (diagonal == PAIR_AU)
    for name, low, high in (("central", 8, 12), ("supp", 12, 16), ("tail", 16, 99)):
        region = (rows >= low) & (rows < high)
        f[f"ext_{name}_wc"] = diagonal_wc[region].sum()
        f[f"ext_{name}_gu"] = (diagonal[region] == PAIR_GU).sum()
    f["ext_total_wc"] = diagonal_wc.sum()
    f["ext_total_fraction"] = diagonal_wc.mean()
    f["ext_longest_run"] = _longest_run(diagonal_wc)
    f["ext_3p_longest_run"] = _longest_run(diagonal_wc[rows >= 8])

    # --- best pairing anywhere --------------------------------------------------------------
    runs_wc, runs_any = _diagonal_runs(wc), _diagonal_runs(paired)
    end_i, end_j = np.unravel_index(int(np.argmax(runs_wc)), runs_wc.shape)
    longest = runs_wc[end_i, end_j]
    f["glob_longest_wc_run"] = longest
    f["glob_longest_pair_run"] = runs_any.max()
    f["glob_run_mirna_start"] = end_i - longest + 1
    f["glob_run_in_seed"] = int(end_i - longest + 1 <= 7 and end_i >= 1)
    f["glob_run_target_position"] = (49 - end_j) / 49
    f["glob_runs_ge6"] = (runs_wc == 6).sum()
    traces = np.array([np.trace(wc, offset=d) for d in range(-(length - 8), 43)])
    f["glob_best_diagonal_wc"] = traces.max()
    f["glob_best_diagonal_is_seed_register"] = int(np.argmax(traces) - (length - 8) == shift)
    f["glob_wc_density"] = wc.mean()

    # --- composition ------------------------------------------------------------------------
    f["comp_mirna_length"] = length
    for name, base in (("a", A), ("c", C), ("g", G), ("u", T)):
        f[f"comp_mirna_{name}"] = (m == base).mean()
        f[f"comp_target_{name}"] = (t == base).mean()
        f[f"comp_mirna_pos1_{name}"] = int(m[0] == base)
    forward = t[::-1]
    dinucleotides = (forward[:-1] - 1) * 4 + (forward[1:] - 1)
    counts = np.bincount(dinucleotides, minlength=16) / len(dinucleotides)
    for index, value in enumerate(counts):
        f[f"comp_target_{'ACGU'[index // 4]}{'ACGU'[index % 4]}"] = value
    return f


def build_features(table: pd.DataFrame, drop_groups=()) -> pd.DataFrame:
    """Feature matrix for a table of pairs; ``drop_groups`` removes whole feature groups."""
    unknown = set(drop_groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature group(s): {sorted(unknown)}")
    rows = [pair_features(m, t) for m, t in zip(table["mirna"], table["target"])]
    features = pd.DataFrame(rows, index=table.index, dtype=np.float32)
    keep = [c for c in features.columns if c.split("_", 1)[0] not in drop_groups]
    return features[keep]
