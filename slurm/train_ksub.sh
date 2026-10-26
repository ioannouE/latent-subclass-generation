#!/bin/bash
# ksub: train every method x seed of the config on the cached frozen features (k-means init, then the head). Submit from the repo root:
#   CONFIG=configs/ksub_cifar100.yaml sbatch slurm/train_ksub.sh
#   CONFIG=configs/ksub_cars.yaml METHODS="frozen coarse" sbatch slurm/train_ksub.sh      (a subset; the variables are passed to the job)
# Settings are in configs/ksub_*.yaml. Run after slurm/extract_features.sh (needs the clean frozen features of the backbone).
#SBATCH --job-name=lsgen-ksub-train
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

CONFIG=${CONFIG:-configs/ksub_cifar100.yaml}
METHODS=${METHODS:-}               # subset of the methods in the config, e.g. "frozen coarse"; empty = all

python scripts/train_ksub.py --config $CONFIG --methods $METHODS
