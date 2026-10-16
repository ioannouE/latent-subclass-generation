#!/bin/bash
# GEORGE clusters vs the hidden subclasses (CPU only). Submit from the repo root, after slurm/train_george.sh:
#   sbatch slurm/eval_george_clusters.sh
# Settings are in configs/eval_george_clusters.yaml.
#SBATCH --job-name=lsgen-george-clusters
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12        # = n_jobs in configs/eval_george_clusters.yaml
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

CONFIG=${CONFIG:-configs/eval_george_clusters.yaml}

python scripts/eval_george_clusters.py --config $CONFIG
