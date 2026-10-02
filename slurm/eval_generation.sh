#!/bin/bash
# Milestone 4: use case A (G1-G7) or B (B1-B6) metrics for one generated set (GPU). Submit from the repo root:
#   sbatch slurm/eval_generation.sh
# Settings are in configs/eval_generation.yaml. Run after slurm/extract_features.sh and train_eval_classifier.sh.
#SBATCH --job-name=lsgen-gen-eval
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        # = n_jobs in configs/eval_generation.yaml
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/eval_generation.py --config configs/eval_generation.yaml
