#!/bin/bash
# Milestone 4: representation metrics R1-R5 for the cached embeddings (CPU only). Submit from the repo root:
#   sbatch slurm/eval_representation.sh
# Settings are in configs/eval_representation.yaml. Run after slurm/extract_features.sh.
#SBATCH --job-name=lsgen-repr
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        # = n_jobs in configs/eval_representation.yaml
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/eval_representation.py --config configs/eval_representation.yaml
