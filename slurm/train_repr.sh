#!/bin/bash
# Step 2: fine-tune the contrastive representation baselines (SupCon, SimCLR, SupCon+SimCLR) on a GPU. Submit from the repo root:
#   sbatch slurm/train_repr.sh
# Settings are in configs/repr_baselines.yaml. Run after slurm/prepare_data.sh (needs only the stored crops).
#SBATCH --job-name=lsgen-repr-train
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = num_workers in configs/repr_baselines.yaml
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/train_repr.py --config configs/repr_baselines.yaml
