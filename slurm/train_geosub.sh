#!/bin/bash
# geosub: cache the frozen backbone's augmented views and mirror images (once), then train every method in the config on them. Submit from the repo root:
#   sbatch slurm/train_geosub.sh
#   METHODS="geo geo_sub" sbatch slurm/train_geosub.sh      (a subset; the variables are passed to the job)
# Settings are in configs/geosub_maskcon.yaml. Run after slurm/extract_features.sh (needs the clean frozen features of the backbone).
#SBATCH --job-name=lsgen-geosub-train
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # = num_workers in configs/geosub_maskcon.yaml
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

CONFIG=${CONFIG:-configs/geosub_maskcon.yaml}
METHODS=${METHODS:-}               # subset of the methods in the config, e.g. "geo geo_sub"; empty = all

python scripts/train_geosub.py --config $CONFIG --methods $METHODS
