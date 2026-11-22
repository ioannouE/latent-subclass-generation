#!/bin/bash
# G-0c pilot: train the coarse-conditioned latent diffusion model, sample every checkpoint with each number of DDIM steps, score each set
# against val, summarise. Run after slurm/autoencoders.sh (set `ldm.autoencoder` in the config first). Submit from the repo root:
#   BENCH=_maskcon_new sbatch slurm/gen_pilot.sh         (configs/gen_pilot_maskcon_new.yaml, configs/eval_generation_maskcon_new.yaml)
#   BENCH=_cifar100 sbatch slurm/gen_pilot.sh
#SBATCH --job-name=lsgen-gen-pilot
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

CKPTS=$(python scripts/train_ldm.py --config $CONFIG)   # prints the checkpoint paths
for ckpt in $CKPTS; do
    for dir in $(python scripts/sample_ldm.py --config $CONFIG --ckpt "$ckpt"); do
        python scripts/eval_generation.py --config configs/eval_generation$BENCH.yaml --samples-dir "$dir" --use-case A
    done
done
python scripts/summarize_pilot.py --config $CONFIG
