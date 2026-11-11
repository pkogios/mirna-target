"""Tests of the data handling, representations and models on small synthetic pairs."""
import numpy as np
import pandas as pd
import pytest
import torch

from mirtar import data, splits
from mirtar.ensemble import combine, fit_stack
from mirtar.features import build_features, pair_features, site_type
from mirtar.models import build_model, load_config
from mirtar.models.neural import NETWORKS
from mirtar.pairing import PAIR_AU, PAIR_GC, PAIR_GU, PAIR_MISMATCH, PAIR_PAD, encode_table, pair_grid

COMPLEMENT = str.maketrans("ACGT", "TGCA")
MIRNA = "TGAGGTAGTAGGTTGTATAGTT"  # let-7a


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


def target_with(site, position=20, filler="C"):
    return (filler * position + site + filler * 50)[:50]


def synthetic_pairs(n=240, seed=0):
    """Positives contain the reverse complement of miRNA positions 1-8; negatives are random."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        mirna = "".join(rng.choice(list("ACGT"), rng.integers(18, 25)))
        target = "".join(rng.choice(list("ACGT"), 50))
        if i % 2:
            start = rng.integers(0, 42)
            target = target[:start] + reverse_complement(mirna[:8]) + target[start + 8:]
        rows.append({"sample_id": f"train_{i}", "mirna": mirna, "target": target, "label": i % 2})
    return pd.DataFrame(rows)


def test_only_sequences_and_label_are_loaded(tmp_path):
    path = tmp_path / "pairs.tsv"
    pd.DataFrame({"gene": ["A" * 50, "C" * 50], "noncodingRNA": ["ACGT" * 5, "TTGCA" * 4], "Nunique": [9, 0],
                  "label": [1, 0]}).to_csv(path, sep="\t", index=False)
    table = data.load_pairs(path, "train")
    assert list(table.columns) == ["sample_id", "mirna", "target", "label"]
    assert list(table["sample_id"]) == ["train_0", "train_1"]


@pytest.mark.parametrize("mirna, target", [("ACGN", "A" * 50), ("ACGT", "A" * 49), ("A" * 29, "A" * 50)])
def test_invalid_sequences_are_rejected(mirna, target):
    with pytest.raises(ValueError):
        data.validate_pairs(pd.DataFrame({"mirna": [mirna], "target": [target], "label": [1]}))


def test_splits_are_reproducible_and_disjoint():
    table = synthetic_pairs(400)
    for name in splits.SPLITS:
        train, validation = splits.split_train_validation(table, name)
        again = splits.split_train_validation(table, name)[1]
        assert list(validation.index) == list(again.index)
        assert not set(train.index) & set(validation.index) and len(train) + len(validation) == len(table)
        assert 0.15 < len(validation) / len(table) < 0.25
        assert abs(validation["label"].mean() - 0.5) < 0.1
    train, validation = splits.split_train_validation(table, "group")
    assert not set(splits.seed_family(train["mirna"])) & set(splits.seed_family(validation["mirna"]))


def test_stratified_split_equals_the_example_notebook():
    table = synthetic_pairs(400)
    expected = table.groupby("label").sample(frac=0.20, random_state=42)
    _, validation = splits.split_train_validation(table, "stratified")
    assert set(validation.index) == set(expected.index)


def test_pair_grid_is_antiparallel_and_padded():
    table = pd.DataFrame({"mirna": ["GATG"], "target": [target_with("", 0, "C")[:46] + "TATC"]})
    mirna, target = encode_table(table)
    grid = pair_grid(mirna, target)[0]
    assert grid.shape == (28, 50)
    # target read 3'->5' is C,T,A,T,...: G:C, A:U, U:A, then G:U wobble on the diagonal
    assert [grid[i, i] for i in range(4)] == [PAIR_GC, PAIR_AU, PAIR_AU, PAIR_GU]
    assert grid[0, 1] == PAIR_GU and grid[1, 0] == PAIR_MISMATCH
    assert (grid[4:] == PAIR_PAD).all() and (grid[:4] != PAIR_PAD).all()


@pytest.mark.parametrize("site, flank_a, expected", [
    (reverse_complement(MIRNA[1:8]), True, "8mer"),
    (reverse_complement(MIRNA[1:8]), False, "7mer-m8"),
    (reverse_complement(MIRNA[1:7]), True, "7mer-A1"),
    (reverse_complement(MIRNA[1:7]), False, "6mer"),
    (reverse_complement(MIRNA[2:8]), False, "offset-6mer"),
    ("", False, "none"),
])
def test_canonical_site_types(site, flank_a, expected):
    target = target_with(site + ("A" if flank_a else "G"), 20, "G")
    assert site_type(MIRNA, target) == expected


def test_features_of_a_perfect_site():
    f = pair_features(MIRNA, target_with(reverse_complement(MIRNA), 10, "G"))
    assert f["seed_wc_max"] == 7 and f["site_7mer_m8"] == 1 and f["glob_longest_wc_run"] == len(MIRNA)
    assert f["ext_total_fraction"] == 1 and f["glob_run_mirna_start"] == 0 and f["glob_run_in_seed"] == 1
    none = pair_features(MIRNA, "G" * 50)
    assert none["site_6mer"] == 0 and none["seed_wc_max"] <= 2


def test_feature_groups_can_be_dropped():
    table = synthetic_pairs(6)
    everything, reduced = build_features(table), build_features(table, drop_groups=("seed", "site"))
    assert not everything.isna().any().any()
    assert not [c for c in reduced.columns if c.startswith(("seed_", "site_"))]
    assert set(reduced.columns) < set(everything.columns)
    with pytest.raises(ValueError):
        build_features(table, drop_groups=("typo",))


@pytest.mark.parametrize("name", list(NETWORKS))
def test_networks_ignore_padding_and_batch_order(name):
    torch.manual_seed(0)
    net = NETWORKS[name]().eval()
    mirna, target = map(torch.from_numpy, encode_table(synthetic_pairs(8)))
    with torch.no_grad():
        full, reordered, single = net(mirna, target), net(mirna.flip(0), target.flip(0)).flip(0), net(mirna[:1], target[:1])
    assert full.shape == (8,)
    assert torch.allclose(full, reordered, atol=1e-5) and torch.allclose(full[:1], single, atol=1e-5)


def test_attention_map_is_a_distribution_over_the_target():
    net = NETWORKS["cross_attention"]()
    mirna, target = map(torch.from_numpy, encode_table(synthetic_pairs(4)))
    attention = net.attention_map(mirna, target)
    assert attention.shape == (4, 28, 50) and torch.allclose(attention.sum(-1), torch.ones(4, 28), atol=1e-5)


@pytest.mark.parametrize("name, minimum", [
    ("m1_features_gbm", 0.95), ("seed_rule", 0.95),
    # two epochs on 480 pairs only check that training, early stopping and refitting run
    ("m2_cross_attention", 0.0), ("m3_complementarity_cnn", 0.0), ("example_cnn", 0.0),
])
def test_models_fit_predict_and_refit(name, minimum):
    from sklearn.metrics import roc_auc_score
    table = synthetic_pairs(600)
    train, validation = table.iloc[:480], table.iloc[480:]
    overrides = {"max_epochs": 2, "verbose": False} if load_config()[name]["kind"] == "neural" else {}
    model = build_model(name, seed=0, **overrides).fit(train, validation)
    probabilities = model.predict_proba(validation)
    assert probabilities.shape == (120,) and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert roc_auc_score(validation["label"], probabilities) > minimum
    refit = build_model(name, seed=0, rounds=max(model.best_rounds, 1), **overrides).fit(train)
    assert refit.predict_proba(validation).shape == (120,)


def test_single_sequence_models_cannot_see_the_interaction():
    from sklearn.metrics import roc_auc_score
    table = synthetic_pairs(600)
    train, validation = table.iloc[:480], table.iloc[480:]
    for name in ("mirna_only", "target_only"):
        probabilities = build_model(name).fit(train).predict_proba(validation)
        assert roc_auc_score(validation["label"], probabilities) < 0.7


def test_every_configured_model_can_be_built():
    for name in load_config():
        assert build_model(name) is not None


def test_stacking_weights_the_informative_model():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 2000)
    good = np.clip(0.5 + 0.3 * (labels - 0.5) + rng.normal(0, 0.15, 2000), 0.01, 0.99)
    noise = rng.uniform(0.01, 0.99, 2000)
    stack = fit_stack(np.column_stack([good, noise]), labels, ["good", "noise"])
    assert stack["weights"][0] > 5 * abs(stack["weights"][1])
    combined = combine(stack, np.column_stack([good, noise]))
    assert ((combined > 0) & (combined < 1)).all()
