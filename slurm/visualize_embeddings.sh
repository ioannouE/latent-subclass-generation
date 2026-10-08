#!/bin/bash
# UMAP / t-SNE figures for every encoder and representation method, both label schemes (CPU only). Submit from the repo root:
#   sbatch slurm/visualize_embeddings.sh
# Settings are in configs/visualize.yaml. Run after slurm/extract_features.sh and slurm/train_repr.sh (methods without
# embeddings are skipped with a warning; projections are cached, so re-running after adding methods only computes the new ones).
#SBATCH --job-name=lsgen-viz
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        # = n_jobs in configs/visualize.yaml
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/visualize_embeddings.py --config configs/visualize.yaml
