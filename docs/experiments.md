# Experiment log

All numbers are **validation** results (no test data has been downloaded).
ROC-AUC with 95% bootstrap interval from `mirna_eval`, model seed 0, one run each
unless stated. Reproduce any row with `python -m mirtar.train --model <name> [--split group]`
followed by `bash scripts/evaluate_all.sh <split>`.

Data: 8,140 training pairs (4,084 positive), 435 distinct miRNAs, 3,933 distinct
targets. `stratified` split: 6,512 / 1,628. `group` split: 6,561 / 1,579, no
miRNA seed family shared between the two sides.

## E1. Is there a shortcut in the synthetic negatives? (no)

| Model | sees | stratified | group |
|---|---|---|---|
| `mirna_only` | miRNA k-mers | 0.438 | 0.500 |
| `target_only` | target k-mers | 0.476 | 0.351 |
| `kmer_logistic` | both, no interaction | 0.451 | – |

No single sequence predicts the label. `target_only` is *below* chance because
of how negatives were built: 82% (stratified) / 86% (group) of validation
negatives use a target that is a positive in training, paired with another
miRNA. A model that memorises "this target binds" is therefore wrong on purpose.
Consequence: a model can only succeed by looking at the pair.

## E2. Reference points

| Model | stratified | group |
|---|---|---|
| `seed_rule` (best seed complementarity, nothing learned) | 0.738 [0.715, 0.761] | 0.759 |
| `example_cnn` (course example network, our re-implementation) | 0.805 [0.783, 0.825] | not run |

Only 22% of training positives contain a canonical 6-mer seed match (1.4% of
negatives), which is why the seed rule is precise but misses most positives.

## E3. M1: features + gradient boosting

| Variant | stratified | group |
|---|---|---|
| `m1_features_gbm` (all 72 features) | 0.867 [0.849, 0.884] | 0.850 |
| `m1_no_seed` (without seed_ and site_ features) | 0.869 | 0.847 |
| `m1_composition_only` | 0.720 | not run |

- M1 beats the example CNN by 0.064 ROC-AUC [0.046, 0.082], paired bootstrap p = 0.002
  (computed for `m1_no_seed`; the two M1 variants do not differ, p = 0.38).
- Removing the explicit seed features changes nothing. The `glob_` features
  (longest Watson–Crick stretch anywhere, best diagonal) carry the signal and
  cover seed matches as a special case. SHAP per group: glob 1.82, comp 0.89,
  seed 0.42, ext 0.23, site 0.19; top feature `glob_longest_wc_run`.
- Composition alone reaches 0.72. Trees can combine miRNA and target composition
  (a G-rich miRNA with a C-rich target), so this is a weak interaction signal,
  not a single-sequence shortcut (see E1).
- By site type (stratified): pairs with no canonical site are 85% of validation
  and are the hard part (ROC-AUC 0.828).

## E4. M3: complementarity CNN

| Variant | stratified | group |
|---|---|---|
| `m3_complementarity_cnn` | 0.871 [0.852, 0.887] | not run |

Early stopping selected epoch 8 of 16 (validation loss 0.445; it overfits
quickly afterwards: training loss 0.33, validation 0.51 at epoch 16).
5-fold out-of-fold ROC-AUC on the training part: 0.848–0.864.
Input-gradient saliency follows diagonals of the pair grid, and averaged over
validation positives it peaks at miRNA positions 4–7 (1.3–1.45 times uniform),
although the network was never told where the seed is.

## E5. M2: cross-attention (weaker than expected)

| Variant | pairing information | stratified |
|---|---|---|
| `m2_scalar_bias` | learned bias per pair type on attention logits | 0.766 [0.745, 0.786] |
| `m2_bias_only` | bias from a 3×3 convolution over the pair grid | 0.752 |
| `m2_cross_attention` | convolutional bias + pairing added to attended values | see below |

All variants *underfit* (training loss still about 0.53 after 27 epochs, where
M3 reaches 0.45 in 8), so the problem is optimisation of this architecture on
6,512 examples, not overfitting. Changing how pairing information enters the
attention did not help. We stopped after two redesigns; see the open question in
`decisions.md`.

<!-- M2_FINAL -->

## E6. Ensembles

| Combination | stratified |
|---|---|
| mean of M1 and M3 | 0.880 [0.863, 0.896] |
| stack of M1 and M3 (weights 0.69 / 0.33) | 0.879 [0.863, 0.895] |

The mean of M1 and M3 beats M3 by 0.0095 [0.004, 0.015] (p = 0.002) and M1 by
0.013 [0.007, 0.019] (p = 0.002). Stacking is not better than the plain mean
(p = 0.21).

<!-- ENSEMBLE_FINAL -->

## Not done yet
- Neural models on the `group` split, and several seeds per model.
- Hyperparameter tuning (all settings are first guesses).
- The final test evaluation.
