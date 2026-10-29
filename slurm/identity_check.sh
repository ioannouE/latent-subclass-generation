#!/bin/bash
# Pre-check of the DINO-minus-DreamSim identity score (docs/EXPERIMENT_IDSCORE.md). CPU only. Submit from the repo root:
#   sbatch slurm/identity_check.sh
#   CONFIG=configs/identity_check_maskcon.yaml sbatch slurm/identity_check.sh      (MaskCon label scheme: within body types)
#   CONFIG=configs/identity_check_maskcon_new.yaml sbatch slurm/identity_check.sh  (maskcon_new label scheme)
# Settings are in configs/identity_check.yaml. Run after slurm/extract_features.sh (dinov3_base and dreamsim_ensemble embeddings).
#SBATCH --job-name=lsgen-identity-check
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = n_jobs in configs/identity_check.yaml
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

CONFIG=${CONFIG:-configs/identity_check.yaml}

python scripts/identity_check.py --config $CONFIG
