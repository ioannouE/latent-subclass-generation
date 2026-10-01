#!/bin/bash
# Milestone 3: metrics, UMAP / t-SNE and plots for the cached embeddings (CPU only). Submit from the repo root:
#   sbatch slurm/analyze_features.sh
# Settings are in configs/analysis.yaml. Run after slurm/extract_features.sh and slurm/train_eval_classifier.sh.
#SBATCH --job-name=lsgen-analysis
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        # = n_jobs in configs/analysis.yaml
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/analyze_features.py --config configs/analysis.yaml
