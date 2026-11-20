#!/bin/bash
# geosub diagnostics: trained model vs its own initialisation, train and val (CPU only). Submit from the repo root, after slurm/train_geosub.sh:
#   sbatch slurm/diagnose_geosub.sh
# Settings are in configs/diagnose_geosub_maskcon.yaml.
#SBATCH --job-name=lsgen-geosub-diag
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = n_jobs in configs/diagnose_geosub_maskcon.yaml
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

CONFIG=${CONFIG:-configs/diagnose_geosub_maskcon.yaml}

python scripts/diagnose_geosub.py --config $CONFIG
