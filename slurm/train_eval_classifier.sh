#!/bin/bash
# Milestone 3: train the evaluator classifiers (subclass make-model 189, make 49; evaluation only). Submit from the repo root:
#   sbatch slurm/train_eval_classifier.sh
# Settings are in configs/classifier.yaml. Run after slurm/prepare_data.sh.
#SBATCH --job-name=lsgen-classifier
#SBATCH --partition=rtx           # any GPU partition works (a100, rtx, a5000, ...)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = num_workers in configs/classifier.yaml
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/train_eval_classifier.py --config configs/classifier.yaml
