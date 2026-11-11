"""Command line interface.

    python -m mirna_eval evaluate         --predictions P.csv --answers A.csv --out DIR
    python -m mirna_eval select-threshold --predictions P.csv --answers A.csv --out threshold.json
    python -m mirna_eval compare          --answers A.csv --predictions name=P1.csv name2=P2.csv --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import (
    EvaluationInputError, align, bootstrap_ci, confusion_counts, load_answers, load_metadata,
    load_predictions, metrics_by_group, paired_bootstrap, select_threshold,
)
from .metrics import HIGHER_IS_BETTER, calibration_table


def _threshold(args) -> float:
    if getattr(args, "threshold_file", None):
        return float(json.loads(Path(args.threshold_file).read_text())["threshold"])
    return args.threshold


def _load(predictions_path, answers):
    return align(load_predictions(predictions_path), answers)


def evaluate(args) -> None:
    ids, labels, probabilities = _load(args.predictions, load_answers(args.answers))
    threshold = _threshold(args)
    table = bootstrap_ci(labels, probabilities, threshold, n_boot=args.n_boot, seed=args.seed)
    print(f"{args.name}: {len(labels)} samples, {int(labels.sum())} positive, threshold {threshold:.4f}")
    print(table.round(4).to_string(index=False))
    if not args.out:
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics.csv", index=False)
    calibration_table(labels, probabilities).to_csv(out / "calibration.csv", index=False)
    summary = {
        "name": args.name, "n": int(len(labels)), "n_positive": int(labels.sum()), "threshold": threshold,
        "confusion": confusion_counts(labels, probabilities, threshold),
        "metrics": dict(zip(table["metric"], table["value"])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.metadata:
        metadata = load_metadata(args.metadata, ids)
        for column in args.by or [c for c in metadata.columns if c != "sample_id"]:
            breakdown = metrics_by_group(labels, probabilities, metadata[column].to_numpy(), threshold, args.min_group)
            breakdown.to_csv(out / f"by_{column}.csv", index=False)
            print(f"\nBy {column}: {len(breakdown)} group(s) with >= {args.min_group} samples and both classes")
            if len(breakdown):
                print(breakdown[["group", "n", "positive_rate", "roc_auc", "balanced_accuracy"]].head(12).round(3).to_string(index=False))
    if not args.no_plots:
        from .plots import plot_confusion, plot_curves
        plot_curves(labels, {args.name: probabilities}, out / "curves.png", args.name)
        plot_confusion(labels, probabilities, threshold, out / "confusion.png")
    print(f"\nSaved to {out}")


def choose_threshold(args) -> None:
    _, labels, probabilities = _load(args.predictions, load_answers(args.answers))
    threshold = select_threshold(labels, probabilities, args.criterion)
    payload = {"threshold": threshold, "criterion": args.criterion, "selected_on": str(args.answers)}
    print(json.dumps(payload))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))


def compare(args) -> None:
    answers = load_answers(args.answers)
    thresholds = dict(item.split("=", 1) for item in args.thresholds or [])
    models, labels = {}, None
    for item in args.predictions:
        if "=" not in item:
            raise EvaluationInputError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        _, labels, models[name] = _load(path, answers)

    def threshold_of(name):
        value = thresholds.get(name, args.threshold)
        return float(json.loads(Path(value).read_text())["threshold"]) if str(value).endswith(".json") else float(value)

    rows = []
    for name, probabilities in models.items():
        table = bootstrap_ci(labels, probabilities, threshold_of(name), n_boot=args.n_boot, seed=args.seed)
        row = {"model": name, "threshold": threshold_of(name)}
        for record in table.itertuples():
            row[record.metric] = record.value
            row[f"{record.metric}_ci"] = f"[{record.ci_low:.3f}, {record.ci_high:.3f}]"
        rows.append(row)
    ascending = not HIGHER_IS_BETTER[args.sort_by]
    summary = pd.DataFrame(rows).sort_values(args.sort_by, ascending=ascending).reset_index(drop=True)
    shown = ["model", "roc_auc", "roc_auc_ci", "average_precision", "balanced_accuracy", "mcc", "f1", "log_loss", "brier", "ece"]
    print(summary[shown].round(4).to_string(index=False))

    reference = args.reference or summary["model"].iloc[0]
    if reference not in models:
        raise EvaluationInputError(f"Reference model '{reference}' is not among the predictions")
    tests = []
    for name, probabilities in models.items():
        if name == reference:
            continue
        table = paired_bootstrap(labels, models[reference], probabilities, threshold_of(reference),
                                 threshold_of(name), n_boot=args.n_boot, seed=args.seed)
        table.insert(0, "b_model", name)
        table.insert(0, "a_model", reference)
        tests.append(table)
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    if len(tests):
        print(f"\nPaired bootstrap: {reference} minus each other model")
        print(tests.round(4).to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "comparison.csv", index=False)
        tests.to_csv(out / "paired_tests.csv", index=False)
        if not args.no_plots:
            from .plots import plot_curves
            ordered = {name: models[name] for name in summary["model"]}
            plot_curves(labels, ordered, out / "comparison_curves.png", args.title)
        print(f"\nSaved to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mirna_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub):
        sub.add_argument("--answers", required=True, help="CSV with sample_id,label")
        sub.add_argument("--out", help="output directory (or file for select-threshold)")
        sub.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples")
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--no-plots", action="store_true")

    sub = commands.add_parser("evaluate", help="metrics with confidence intervals for one model")
    common(sub)
    sub.add_argument("--predictions", required=True, help="CSV with sample_id,probability")
    sub.add_argument("--name", default="model")
    sub.add_argument("--threshold", type=float, default=0.5)
    sub.add_argument("--threshold-file", help="JSON written by select-threshold (overrides --threshold)")
    sub.add_argument("--metadata", help="CSV with sample_id and grouping columns for breakdowns")
    sub.add_argument("--by", nargs="*", help="metadata columns to break down by (default: all)")
    sub.add_argument("--min-group", type=int, default=20)
    sub.set_defaults(run=evaluate)

    sub = commands.add_parser("select-threshold", help="choose a decision threshold on validation data")
    sub.add_argument("--predictions", required=True)
    sub.add_argument("--answers", required=True)
    sub.add_argument("--criterion", default="balanced_accuracy", choices=["balanced_accuracy", "f1"])
    sub.add_argument("--out", help="JSON file to write")
    sub.set_defaults(run=choose_threshold)

    sub = commands.add_parser("compare", help="several models on the same answers, with paired tests")
    common(sub)
    sub.add_argument("--predictions", nargs="+", required=True, metavar="NAME=PATH")
    sub.add_argument("--thresholds", nargs="*", metavar="NAME=VALUE_OR_JSON", help="per-model thresholds")
    sub.add_argument("--threshold", default=0.5, help="threshold for models not listed in --thresholds")
    sub.add_argument("--reference", help="model to test the others against (default: best by --sort-by)")
    sub.add_argument("--sort-by", default="roc_auc", choices=list(HIGHER_IS_BETTER))
    sub.add_argument("--title", default="")
    sub.set_defaults(run=compare)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.run(args)
    except EvaluationInputError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
