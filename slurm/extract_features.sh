#!/bin/bash
# Milestone 3: cache embeddings (CSV: filename, z1..zD) for DINOv3, CLIP-L, SSCD. Submit from the repo root:
#   sbatch slurm/extract_features.sh
# Settings are in configs/features.yaml. Run after slurm/prepare_data.sh.
#SBATCH --job-name=lsgen-features
#SBATCH --partition=rtx           # any GPU partition works (a100, rtx, a5000, ...)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = num_workers in configs/features.yaml
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/extract_features.py --config configs/features.yaml
