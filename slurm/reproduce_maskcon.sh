#!/bin/bash
# Reproduce MaskCon on Stanford Cars196 (8 body-type coarse classes, ResNet-18, 200 epochs) on a GPU. Submit from the repo root:
#   sbatch slurm/reproduce_maskcon.sh
# Settings are in configs/maskcon_repro.yaml. Needs only the raw images and slurm/prepare_data.sh (metadata, splits).
#SBATCH --job-name=lsgen-maskcon-repro
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = num_workers in configs/maskcon_repro.yaml
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/reproduce_maskcon.py --config configs/maskcon_repro.yaml
