#!/bin/bash
# G-0c autoencoder choice: train the own AutoencoderKL, reconstruct val with it and with the SD-VAE, score both against val. Submit from the
# repo root:
#   BENCH=_maskcon_new sbatch slurm/autoencoders.sh      (configs/gen_pilot_maskcon_new.yaml, configs/eval_generation_maskcon_new.yaml)
#   BENCH=_cifar100 sbatch slurm/autoencoders.sh
# Then (CPU): python scripts/summarize_pilot.py --config configs/gen_pilot$BENCH.yaml
#SBATCH --job-name=lsgen-autoencoders
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        # = n_jobs in configs/eval_generation*.yaml
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

BENCH=${BENCH:-_maskcon_new}
CONFIG=configs/gen_pilot$BENCH.yaml

python scripts/train_autoencoder.py --config $CONFIG
DIRS=$(python scripts/reconstruct_val.py --config $CONFIG)   # prints the reconstruction directories
for dir in $DIRS; do
    python scripts/eval_generation.py --config configs/eval_generation$BENCH.yaml --samples-dir "$dir" --use-case A
done
