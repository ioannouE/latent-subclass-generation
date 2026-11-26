#!/bin/bash
# Sample every checkpoint of the official DiT runs of a config (scripts/sample_dit.py), score each set (use case A, reference split from
# the config's `sampling`) and summarise: the pilot curves (config without `seeds`) or the rung over seeds. Submit from the repo root:
#   CONFIG=configs/gen_dit_pilot_maskcon_new.yaml sbatch slurm/sample_dit.sh
#SBATCH --job-name=lsgen-sample-dit
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

CONFIG=${CONFIG:-configs/gen_dit_pilot_maskcon_new.yaml}
EVAL=configs/eval_generation_maskcon_new.yaml

for dir in $(python scripts/sample_dit.py --config $CONFIG); do
    python scripts/eval_generation.py --config $EVAL --samples-dir "$dir" --use-case A
done
if grep -q '^seeds:' $CONFIG; then python scripts/summarize_generation.py --config $CONFIG; else python scripts/summarize_pilot.py --config $CONFIG; fi
