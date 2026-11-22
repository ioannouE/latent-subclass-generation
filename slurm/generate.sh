#!/bin/bash
# Generation ladder: train one rung of the controlled latent diffusion generator (every seed in the config), sample it, score each set
# (reference split from the config's `sampling`), summarise over seeds. Submit from the repo root:
#   BENCH=_maskcon_new sbatch slurm/generate.sh                     (configs/gen_coarse_maskcon_new.yaml, configs/eval_generation_maskcon_new.yaml)
#   BENCH=_cifar100 sbatch slurm/generate.sh
#   RUNG=<name> BENCH=... sbatch slurm/generate.sh                  (configs/gen_<name>$BENCH.yaml; default coarse)
# Cluster rungs need their cluster files first (configs/gen_cluster_*: scripts/train_ksub.py frozen runs).
#SBATCH --job-name=lsgen-generate
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        # = n_jobs in configs/eval_generation*.yaml
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -A p315                     # accounting project

set -euo pipefail
cd /nvme/h/eioannou/code/latent-subclass-generation
source activate diffusion

BENCH=${BENCH:-_maskcon_new}
RUNG=${RUNG:-coarse}
CONFIG=configs/gen_$RUNG$BENCH.yaml

if [ "$RUNG" = oracle ]; then python scripts/oracle_labels.py --config $CONFIG; fi   # the oracle's label file (fine classes)
CKPTS=$(python scripts/train_ldm.py --config $CONFIG)   # prints the checkpoint paths
for ckpt in $CKPTS; do
    for dir in $(python scripts/sample_ldm.py --config $CONFIG --ckpt "$ckpt"); do
        python scripts/eval_generation.py --config configs/eval_generation$BENCH.yaml --samples-dir "$dir" --use-case A
    done
done
python scripts/summarize_generation.py --config $CONFIG
