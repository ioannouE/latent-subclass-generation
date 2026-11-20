#!/bin/bash
# Milestone 3: train the evaluator classifiers (hidden subclass and coarse class; evaluation only). Submit from the repo root:
#   sbatch slurm/train_eval_classifier.sh                                          (Stanford Cars, make scheme)
#   CONFIG=configs/classifier_maskcon_new.yaml sbatch slurm/train_eval_classifier.sh
#   CONFIG=configs/classifier_cifar100.yaml sbatch slurm/train_eval_classifier.sh
# Settings are in the config. Run after slurm/prepare_data.sh.
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

CONFIG=${CONFIG:-configs/classifier.yaml}

python scripts/train_eval_classifier.py --config $CONFIG
