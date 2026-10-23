#!/bin/bash
# CIFAR-100 benchmark, Milestone 1 (docs/DATA_CIFAR100.md): PNGs, hierarchy, splits, SSCD duplicate audit, metadata, report. Submit from the
# repo root after downloading the raw data (docs/DATA_CIFAR100.md):
#   sbatch slurm/prepare_cifar100.sh
# Settings are in configs/data_cifar100.yaml.
#SBATCH --job-name=lsgen-prepare-cifar100
#SBATCH --partition=rtx           # the duplicates stage runs SSCD on the GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = num_workers in configs/data_cifar100.yaml
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

STAGES="raw splits duplicates finalize report"

python scripts/prepare_cifar100.py --config configs/data_cifar100.yaml --stages ${STAGES}
