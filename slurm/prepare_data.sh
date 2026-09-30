#!/bin/bash
# Milestone 1 data preparation + Milestone 2 episodes on a GPU node. Submit from the repo root:
#   sbatch slurm/prepare_data.sh
# Pipeline settings are in configs/data.yaml and configs/episodes.yaml.
#SBATCH --job-name=lsgen-prepare
#SBATCH --partition=rtx           # any GPU partition works (a100, rtx, a5000, ...)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = num_workers in configs/data.yaml
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

STAGES="raw hierarchy crops splits duplicates finalize reports"
EXTRA_ARGS="--recompute-embeddings"   # set to "" once the GPU SSCD embeddings exist
BUILD_EPISODES=1                      # 0 to skip; episodes are frozen (rebuild needs --overwrite)

python scripts/prepare_data.py --config configs/data.yaml --stages ${STAGES} ${EXTRA_ARGS}

if [ "${BUILD_EPISODES}" = "1" ]; then
  python scripts/build_episodes.py --config configs/episodes.yaml
fi
