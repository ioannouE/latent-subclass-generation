#!/bin/bash
# Step 2: GEORGE baseline (ERM -> UMAP + GMM clusters per make -> group DRO) on a GPU. Submit from the repo root:
#   sbatch slurm/train_george.sh
# Settings are in configs/repr_george.yaml. Needs the clone in third_party/hidden-stratification.
# Then score it: CONFIG=configs/eval_repr_george.yaml sbatch slurm/eval_representation.sh
#SBATCH --job-name=lsgen-george
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = num_workers in configs/repr_george.yaml
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

CONFIG=${CONFIG:-configs/repr_george.yaml}

python scripts/train_george.py --config $CONFIG
