"""The evaluator is checked on cases with known answers and against scikit-learn."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

import mirna_eval as me
from mirna_eval.__main__ import main


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    probabilities = np.clip(0.5 + 0.25 * (labels - 0.5) + rng.normal(0, 0.2, 400), 0, 1)
    return labels, probabilities


def write(tmp_path, labels, probabilities, name="p"):
    ids = [f"s_{i}" for i in range(len(labels))]
    answers, predictions = tmp_path / "answers.csv", tmp_path / f"{name}.csv"
    pd.DataFrame({"sample_id": ids, "label": labels}).to_csv(answers, index=False)
    pd.DataFrame({"sample_id": ids, "probability": probabilities}).to_csv(predictions, index=False)
    return predictions, answers


def test_perfect_inverted_and_constant_predictions():
    labels = np.array([0, 0, 1, 1, 0, 1])
    perfect = me.compute_metrics(labels, labels.astype(float))
    assert perfect["roc_auc"] == 1 and perfect["mcc"] == 1 and perfect["balanced_accuracy"] == 1
    inverted = me.compute_metrics(labels, 1.0 - labels)
    assert inverted["roc_auc"] == 0 and inverted["mcc"] == -1 and inverted["accuracy"] == 0
    constant = me.compute_metrics(labels, np.full(6, 0.5))
    assert constant["roc_auc"] == 0.5 and constant["mcc"] == 0 and constant["balanced_accuracy"] == 0.5


def test_random_predictions_are_at_chance():
    rng = np.random.default_rng(1)
    labels, probabilities = rng.integers(0, 2, 20000), rng.random(20000)
    values = me.compute_metrics(labels, probabilities)
    assert abs(values["roc_auc"] - 0.5) < 0.02 and abs(values["mcc"]) < 0.03


def test_matches_scikit_learn(data):
    labels, probabilities = data
    ours, predicted = me.compute_metrics(labels, probabilities, 0.4), (probabilities >= 0.4).astype(int)
    expected = {
        "roc_auc": skm.roc_auc_score(labels, probabilities),
        "average_precision": skm.average_precision_score(labels, probabilities),
        "log_loss": skm.log_loss(labels, np.clip(probabilities, 1e-7, 1 - 1e-7)),
        "brier": skm.brier_score_loss(labels, probabilities),
        "accuracy": skm.accuracy_score(labels, predicted),
        "balanced_accuracy": skm.balanced_accuracy_score(labels, predicted),
        "precision": skm.precision_score(labels, predicted),
        "recall": skm.recall_score(labels, predicted),
        "specificity": skm.recall_score(labels, predicted, pos_label=0),
        "f1": skm.f1_score(labels, predicted),
        "mcc": skm.matthews_corrcoef(labels, predicted),
    }
    for name, value in expected.items():
        assert ours[name] == pytest.approx(value, abs=1e-9), name


def test_calibration_error():
    labels = np.array([1] * 80 + [0] * 20)
    assert me.expected_calibration_error(labels, np.full(100, 0.8)) == pytest.approx(0)
    assert me.expected_calibration_error(labels, np.full(100, 0.3)) == pytest.approx(0.5)


def test_selected_threshold_is_optimal(data):
    labels, probabilities = data
    threshold = me.select_threshold(labels, probabilities)
    best = me.compute_metrics(labels, probabilities, threshold)["balanced_accuracy"]
    for other in np.linspace(0.05, 0.95, 91):
        assert best >= me.compute_metrics(labels, probabilities, other)["balanced_accuracy"] - 1e-12


def test_bootstrap_interval_contains_estimate(data):
    labels, probabilities = data
    table = me.bootstrap_ci(labels, probabilities, n_boot=200)
    assert ((table["ci_low"] <= table["value"]) & (table["value"] <= table["ci_high"])).all()
    again = me.bootstrap_ci(labels, probabilities, n_boot=200)
    pd.testing.assert_frame_equal(table, again)  # deterministic for a fixed seed


def test_paired_bootstrap_detects_a_real_difference(data):
    labels, probabilities = data
    noise = np.random.default_rng(2).random(len(labels))
    table = me.paired_bootstrap(labels, probabilities, noise, n_boot=300).set_index("metric")
    assert table.loc["roc_auc", "difference"] > 0 and table.loc["roc_auc", "p_value"] < 0.05
    same = me.paired_bootstrap(labels, probabilities, probabilities, n_boot=100).set_index("metric")
    assert same.loc["roc_auc", "difference"] == 0 and same.loc["roc_auc", "p_value"] == 1


def test_group_breakdown_skips_small_and_single_class_groups(data):
    labels, probabilities = data
    groups = np.array(["big"] * 340 + ["one_class"] * 40 + ["tiny"] * 20, dtype=object)
    labels = labels.copy()
    labels[340:380] = 1
    table = me.metrics_by_group(labels, probabilities, groups, min_size=30)
    assert list(table["group"]) == ["big"] and table["n"].iloc[0] == 340


def test_order_of_rows_does_not_matter(tmp_path, data):
    labels, probabilities = data
    predictions, answers = write(tmp_path, labels, probabilities)
    pd.read_csv(predictions).sample(frac=1, random_state=3).to_csv(predictions, index=False)
    _, y, p = me.align(me.load_predictions(predictions), me.load_answers(answers))
    assert me.compute_metrics(y, p) == me.compute_metrics(labels, probabilities)


@pytest.mark.parametrize("damage, message", [
    (lambda t: t.iloc[:-5], "no prediction"),
    (lambda t: pd.concat([t, t.iloc[:1].assign(sample_id="unknown")]), "no reference label"),
    (lambda t: pd.concat([t, t.iloc[:1]]), "duplicated"),
    (lambda t: t.assign(probability=t["probability"] + 1), r"\[0, 1\]"),
    (lambda t: t.assign(probability=[np.nan] + list(t["probability"][1:])), "non-numeric"),
    (lambda t: t.rename(columns={"probability": "score"}), "missing column"),
])
def test_bad_prediction_files_are_rejected(tmp_path, data, damage, message):
    labels, probabilities = data
    predictions, answers = write(tmp_path, labels, probabilities)
    damage(pd.read_csv(predictions)).to_csv(predictions, index=False)
    with pytest.raises(me.EvaluationInputError, match=message):
        me.align(me.load_predictions(predictions), me.load_answers(answers))
    assert main(["evaluate", "--predictions", str(predictions), "--answers", str(answers), "--n-boot", "10"]) == 2


def test_command_line_end_to_end(tmp_path, data):
    labels, probabilities = data
    predictions, answers = write(tmp_path, labels, probabilities, "good")
    worse, _ = write(tmp_path, labels, np.random.default_rng(5).random(len(labels)), "worse")
    threshold_file = tmp_path / "threshold.json"
    assert main(["select-threshold", "--predictions", str(predictions), "--answers", str(answers), "--out", str(threshold_file)]) == 0
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"sample_id": [f"s_{i}" for i in range(len(labels))], "half": ["a", "b"] * (len(labels) // 2)}).to_csv(metadata, index=False)
    out = tmp_path / "report"
    assert main(["evaluate", "--predictions", str(predictions), "--answers", str(answers), "--out", str(out),
                 "--threshold-file", str(threshold_file), "--metadata", str(metadata), "--n-boot", "50"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["threshold"] == json.loads(threshold_file.read_text())["threshold"]
    assert sum(summary["confusion"].values()) == len(labels)
    assert {"metrics.csv", "curves.png", "confusion.png", "by_half.csv"} <= {p.name for p in out.iterdir()}
    assert main(["compare", "--answers", str(answers), "--predictions", f"good={predictions}", f"worse={worse}",
                 "--out", str(tmp_path / "cmp"), "--n-boot", "50"]) == 0
    comparison = pd.read_csv(tmp_path / "cmp" / "comparison.csv")
    assert list(comparison["model"]) == ["good", "worse"]
