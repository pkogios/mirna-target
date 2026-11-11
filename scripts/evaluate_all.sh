#!/bin/bash
# Evaluate every model that has validation predictions for a split, then compare them.
#
#   bash scripts/evaluate_all.sh [stratified|group] [seed]
#
# Uses only the independent evaluator (mirna_eval) on files in results/.
set -euo pipefail
SPLIT=${1:-stratified}
SEED=${2:-0}
R=results/$SPLIT
PYTHON=${PYTHON:-python}

predictions=()
thresholds=()
for dir in "$R"/predictions/*/seed"$SEED"; do
    [[ -f $dir/validation_predictions.csv ]] || continue
    name=$(basename "$(dirname "$dir")")
    $PYTHON -m mirna_eval select-threshold --predictions "$dir/validation_predictions.csv" \
        --answers "$R/validation_answers.csv" --out "$dir/threshold.json" > /dev/null
    $PYTHON -m mirna_eval evaluate --predictions "$dir/validation_predictions.csv" \
        --answers "$R/validation_answers.csv" --threshold-file "$dir/threshold.json" \
        --metadata "$R/validation_metadata.csv" --by site_type mirna_length \
        --name "$name" --out "$dir/report" > "$dir/report.txt"
    predictions+=("$name=$dir/validation_predictions.csv")
    thresholds+=("$name=$dir/threshold.json")
done

$PYTHON -m mirna_eval compare --answers "$R/validation_answers.csv" \
    --predictions "${predictions[@]}" --thresholds "${thresholds[@]}" \
    --title "Validation ($SPLIT split)" --out "$R/comparison" ${REFERENCE:+--reference "$REFERENCE"}
