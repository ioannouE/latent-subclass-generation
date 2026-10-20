#!/bin/bash
# DreamSim checks, see scripts/check_dreamsim.py. Submit from the repo root, after
# `git clone https://github.com/ssundaram21/dreamsim third_party/dreamsim` (demo images):
#   STAGE=demo sbatch slurm/check_dreamsim.sh
#   STAGE=features sbatch slurm/check_dreamsim.sh      # after extracting configs/features_dreamsim.yaml
#SBATCH --job-name=lsgen-dreamsim-check
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation

source activate diffusion

STAGE=${STAGE:-demo}

python scripts/check_dreamsim.py $STAGE
