#!/bin/bash
# Template for running the neural models on ARIS (GRNET). NOT yet tested on ARIS:
# fill in the account, check the partition/module names with `sinfo` and `module avail`.
#
#   sbatch hpc/train_aris.slurm m3_complementarity_cnn stratified "0 1 2"
#
#SBATCH --job-name=mirtar
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
MODEL=${1:?model name from configs/models.yaml}
SPLIT=${2:-stratified}
SEEDS=${3:-0}

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# module load python/3.x cuda/x.y      # adjust to what ARIS provides
source .venv/bin/activate

python -m mirtar.train --model "$MODEL" --split "$SPLIT" --seeds $SEEDS --oof 5 --save
