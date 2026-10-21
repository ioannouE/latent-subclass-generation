#!/bin/bash
# geosub prototype clusters vs the hidden subclasses: ACC / NMI / ARI with geosub's own K-hat, K-hat error, flip-ARI (CPU only). Submit from the repo root,
# after slurm/train_geosub.sh:
#   sbatch slurm/eval_geosub.sh
# Settings are in configs/eval_geosub_maskcon.yaml (split: val for choosing weights, test only for the final report).
#SBATCH --job-name=lsgen-geosub-eval
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12        # = n_jobs in configs/eval_geosub_maskcon.yaml
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

CONFIG=${CONFIG:-configs/eval_geosub_maskcon.yaml}

python scripts/eval_geosub.py --config $CONFIG
