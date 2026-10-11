#!/bin/bash
# Gate 0 of the NHP experiment plan (docs/EXPERIMENT_NHP.md): GPU for the mirror-image embeddings, then CPU regressions. Submit from the repo root:
#   sbatch slurm/gate0.sh
# Settings are in configs/gate0.yaml. Run after slurm/extract_features.sh (reads the frozen train embeddings).
#SBATCH --job-name=lsgen-gate0
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = num_workers in configs/gate0.yaml
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/gate0_nuisance.py --config configs/gate0.yaml
