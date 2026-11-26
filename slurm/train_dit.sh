#!/bin/bash
# Train the official DiT (external/DiT) on the train split, every seed of the config (scripts/train_dit.py). Submit from the repo root:
#   CONFIG=configs/gen_dit_pilot_maskcon_new.yaml sbatch slurm/train_dit.sh
# then sample and score: CONFIG=... sbatch --dependency=afterok:<this job> slurm/sample_dit.sh
#SBATCH --job-name=lsgen-train-dit
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:4             # = dit.gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32        # = dit.gpus x dit.num_workers
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

CONFIG=${CONFIG:-configs/gen_dit_pilot_maskcon_new.yaml}

python scripts/train_dit.py --config $CONFIG
