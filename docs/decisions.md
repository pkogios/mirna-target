# Design decisions

Newest at the bottom. Each entry: what we decided, why, and what we gave up.

## D1. Three different kinds of model instead of one tuned model
The example is a 2D CNN on a grid of 16 ordered nucleotide pairs. We wanted
approaches that differ from it in *what knowledge they are given*:
M1 is told the biology explicitly (features), M3 is told only the pairing rules
(grid of pair types), M2 is told the least (two token sequences, plus a learned
attention bias per pair type). Comparing them shows how much of the performance
comes from prior knowledge and how much is learned.

## D2. Reverse the target so that duplexes lie on a diagonal
miRNA and target pair antiparallel. In the example both are 5'→3', so a stretch
of consecutive base pairs is an anti-diagonal. We reverse the target once, in
`mirtar/pairing.py`; every model and feature then uses the same convention.

## D3. Pair types instead of 16 pair channels (M3)
(A,U) and (U,A) are the same physical interaction for our purpose. Collapsing
the 16 ordered pairs to G:C / A:U / G:U / mismatch gives the network the
pairing rules instead of making it rediscover them from 6,512 examples.
Given up: any effect that depends on which strand carries which base.

## D4. The evaluator is a separate package with a file interface
`mirna_eval` imports nothing from `mirtar`. It reads `sample_id,probability` and
`sample_id,label` (the format of the example notebook), so it can score any
model, including ones from other groups. It refuses incomplete or malformed
prediction files instead of scoring a subset.

## D5. Two validation protocols
The brief requires 20% of the training file, and we reproduce the example's
split exactly (same rows) so numbers are comparable with it. In that split 188
miRNA seed families occur on both sides. We therefore also report a `group`
split in which validation families are unseen. The seed (nt 2–8) is used as the
family, computed from the sequence; the `noncodingRNA_fam` column is not read.

## D6. Single-sequence models as a shortcut check
The negatives are synthetic. If a model that sees only the miRNA or only the
target scored well, the labels would be predictable without any interaction and
all results would be suspect. See `docs/experiments.md` E1.

## D7. Threshold chosen on validation, by balanced accuracy
Classes are balanced (50.2% positive), so balanced accuracy and Youden's J are
natural. The threshold is stored in `threshold.json` next to the predictions
and must be reused unchanged on the test set. Thresholded validation metrics
are slightly optimistic because the threshold was selected on the same data.

## D8. Stacking on out-of-fold predictions
The ensemble weights are fitted on 5-fold out-of-fold predictions of the
training part. Fold models are trained for the number of rounds selected by
early stopping in the main run, so no validation label is used by the ensemble.

## D9. Test set is read once
Nothing in the repository downloads the test file except
`scripts/download_data.py --test`. It will be run after models, thresholds and
ensemble weights are committed; the commit hash will be recorded here.

## Open questions for the team
- Early stopping uses the validation set, as in the example. Validation scores of
  the neural models are therefore mildly optimistic. Accept, or carve an inner
  early-stopping set out of the training part?
- Add RNA duplex energy (ViennaRNA) to M1? Needs an extra dependency.
- Pretrained RNA language model embeddings as a fourth model (stretch goal).
