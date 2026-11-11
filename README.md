# miRNA–target binding prediction (Hejret 2023 AGO2 CLASH)

Group project for Exercise 10 of the *AI in Bioinformatics* course
([brief](https://github.com/palexiou/ML_Course_2026/blob/main/exercise_10.md)).
Given a miRNA sequence and a 50-nt target site, predict whether they form an
observed AGO2-CLASH binding pair (label 1) or a synthetic negative pair (label 0).

The course example is a small 2D CNN on a grid of the 16 ordered nucleotide pairs.
We built three predictors that work differently from it and from each other, a
combination of them, and an evaluation package that is independent of all of them.

| | Model | Idea |
|---|---|---|
| M1 | `m1_features_gbm` | 72 hand-made biological features (seed matches, canonical site types, pairing beyond the seed, composition) + gradient boosting. Interpretable. |
| M2 | `m2_cross_attention` | miRNA and target attend to each other; attention is biased by a learned score per base-pair type. |
| M3 | `m3_complementarity_cnn` | Residual CNN on a grid of pair types (G:C, A:U, G:U wobble, mismatch) in the antiparallel orientation, plus a miRNA-position channel. |
| M4 | `ensemble_stack` | Logistic regression on the out-of-fold predictions of M1–M3. |

References and checks: the course example network (`example_cnn`), a rule with no
learning (`seed_rule`), and models that see only one of the two sequences
(`mirna_only`, `target_only`) to detect shortcuts in the synthetic negatives.

Results, decisions and the experiment log are in [`docs/`](docs).

## Setup

Python 3.11+.

```bash
git clone https://github.com/pkogios/mirna-target.git && cd mirna-target
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
python scripts/download_data.py          # training file only, into data/
python -m pytest tests -q                # about 1.5 minutes
```

## Train and predict

```bash
# validation predictions on the required split (20% of the training file, stratified by label)
python -m mirtar.train --model m1_features_gbm

# several seeds, out-of-fold predictions for stacking, and a saved model
python -m mirtar.train --model m3_complementarity_cnn --seeds 0 1 2 --oof 5 --save

# stricter split: validation miRNA families (seed 2-8) never appear in training
python -m mirtar.train --model m1_features_gbm --split group

# combine models (needs --oof predictions of each)
python -m mirtar.ensemble --models m1_features_gbm m2_cross_attention m3_complementarity_cnn

# predict with a saved model on any TSV with the columns noncodingRNA and gene
python -m mirtar.predict --model artifacts/stratified/m1_features_gbm_seed0.pkl \
    --data data/AGO2_CLASH_Hejret2023_train.tsv.gz --prefix train --out predictions.csv
```

Model names and hyperparameters live in [`configs/models.yaml`](configs/models.yaml).
Training writes `results/<split>/predictions/<model>/seed<k>/validation_predictions.csv`
and the reference files `results/<split>/validation_answers.csv` and `validation_metadata.csv`.
On a CPU, M1 takes seconds and M2/M3 take about one minute per epoch.

## Evaluate

`mirna_eval` knows nothing about the models. It takes a predictions file
(`sample_id,probability`) and an answers file (`sample_id,label`), from any source.

```bash
R=results/stratified
P=$R/predictions/m1_features_gbm/seed0

# 1. choose the decision threshold on validation data
python -m mirna_eval select-threshold --predictions $P/validation_predictions.csv \
    --answers $R/validation_answers.csv --out $P/threshold.json

# 2. metrics with 95% bootstrap confidence intervals, breakdowns and figures
python -m mirna_eval evaluate --predictions $P/validation_predictions.csv \
    --answers $R/validation_answers.csv --threshold-file $P/threshold.json \
    --metadata $R/validation_metadata.csv --by site_type mirna_length --name M1 --out $P/report

# 3. several models on the same samples, with paired bootstrap tests
python -m mirna_eval compare --answers $R/validation_answers.csv \
    --predictions M1=$P/validation_predictions.csv example=$R/predictions/example_cnn/seed0/validation_predictions.csv \
    --out $R/comparison
```

Files with missing, duplicated, unknown or out-of-range predictions are rejected
with an explanation (exit code 2) rather than scored.

Metrics:

| Metric | What it tells us |
|---|---|
| ROC-AUC | Probability that a random positive is ranked above a random negative. Threshold-free; our main ranking metric. |
| Average precision | Area under the precision–recall curve; focuses on how clean the top-ranked predictions are. |
| Log loss, Brier score | Quality of the probabilities themselves, not only of their order. |
| ECE + calibration curve | Whether "0.8" really means about 80% positives. |
| Balanced accuracy, MCC | Decision quality at the chosen threshold; both account for the two classes equally. |
| Precision, recall, specificity, F1, accuracy | The usual thresholded views, reported for completeness. |

The threshold is selected on validation data only and reused unchanged on the test set.

## Test set policy

The official test file is used once, after all models, hyperparameters, thresholds
and ensemble weights are fixed and committed. Until then nothing in this
repository downloads or reads it (`scripts/download_data.py --test` is the only way to get it).

## Repository layout

```
mirtar/        data loading, splits, pairing rules, features, models, training, ensemble
mirna_eval/    the independent evaluator (library + command line)
configs/       model settings
scripts/       data download
hpc/           SLURM template for ARIS and a cleanup script
tests/         unit tests for the evaluator and the pipeline
results/       predictions, metrics and figures (small files, committed)
docs/          decisions.md, experiments.md
```

## Working together

Branch from `main`, open a pull request, and have another member review it.
Record every experiment in `docs/experiments.md` and every design decision in
`docs/decisions.md` as part of the same pull request.
