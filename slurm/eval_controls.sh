#!/bin/bash
# Milestone 5: build the control sets and score each with the full panel (GPU). Submit from the repo root:
#   sbatch slurm/eval_controls.sh
# Needs slurm/extract_features.sh (all evaluators, incl. inception) and slurm/train_eval_classifier.sh first.
# Then: python scripts/metric_validation.py (CPU) writes reports/metric_validation.{csv,md,png}.
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

python scripts/make_controls.py --config configs/controls.yaml
CONTROLS=/nvme/h/eioannou/data_p315/Stanford_Cars/lsgen_derived/controls   # out_dir in configs/controls.yaml
for dir in "$CONTROLS"/*_A "$CONTROLS"/*_B; do
    python scripts/eval_generation.py --config configs/eval_generation.yaml --samples-dir "$dir" --use-case "${dir: -1}"
done
