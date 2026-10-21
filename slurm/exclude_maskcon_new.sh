#!/bin/bash
# Exclusion list of the maskcon_new label scheme (configs/labels.yaml) from the cached SSCD embeddings; CPU only. Submit from the repo root:
#   sbatch slurm/exclude_maskcon_new.sh
# Writes data/splits/exclude_maskcon_new.txt, data/duplicates_all.csv, data/manifests/exclude_maskcon_new.json.
#SBATCH --job-name=lsgen-exclude-maskcon-new
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

python scripts/prepare_data.py --config configs/data.yaml --stages exclude_maskcon_new
