#!/bin/bash
# Milestone 5: build the control sets and score each with the full panel (GPU). Submit from the repo root:
#   sbatch slurm/eval_controls.sh                               (Stanford Cars, make scheme)
#   BENCH=_maskcon_new sbatch slurm/eval_controls.sh            (configs/controls_maskcon_new.yaml, configs/eval_generation_maskcon_new.yaml)
#   BENCH=_cifar100 sbatch slurm/eval_controls.sh
# Needs slurm/extract_features.sh (all evaluators, incl. inception), the episodes and slurm/train_eval_classifier.sh first.
# Then (CPU): python scripts/metric_validation.py --config configs/eval_generation$BENCH.yaml
#SBATCH --job-name=lsgen-controls
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        # = n_jobs in configs/eval_generation.yaml
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

BENCH=${BENCH:-}

DIRS=$(python scripts/make_controls.py --config configs/controls$BENCH.yaml)   # builds the controls, prints their directories
for dir in $DIRS; do
    python scripts/eval_generation.py --config configs/eval_generation$BENCH.yaml --samples-dir "$dir" --use-case "${dir: -1}"
done
