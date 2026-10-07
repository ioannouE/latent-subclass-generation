#!/bin/bash
# Side experiment: do Gram statistics carry subclass signal beyond the CLS embedding? CPU only. Submit from the repo root,
# after `sbatch slurm/extract_features.sh` has cached the gram_* embeddings:
#   sbatch slurm/gram_check.sh
# Settings are in configs/gram_check.yaml.
#SBATCH --job-name=lsgen-gram
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = n_jobs in configs/gram_check.yaml
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/gram_check.py --config configs/gram_check.yaml
