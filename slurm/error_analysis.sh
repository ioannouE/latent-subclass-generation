#!/bin/bash
# Error analysis of frozen DINOv3 subclass neighbours (scripts/error_analysis_neighbours.py). CPU only. Submit from the repo root:
#   sbatch slurm/error_analysis.sh
#   CONFIG=configs/error_analysis_maskcon.yaml sbatch slurm/error_analysis.sh   # MaskCon label scheme
# Settings are in configs/error_analysis*.yaml. Needs the cached dinov3_base, dreamsim_ensemble and sscd embeddings (256 px crops).
#SBATCH --job-name=lsgen-error-analysis
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G                  # MaskCon body types: up to 1,849 images, 4.3M candidate pairs
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

CONFIG=${CONFIG:-configs/error_analysis.yaml}

python scripts/error_analysis_neighbours.py --config $CONFIG
