#!/bin/bash
# Re-score the generated sets of the finished ladder rungs with the current metric panel (CAS added 26 Nov 2026), then re-summarise each
# rung. The sets are not regenerated; G1-G7 are recomputed with the same seeds. Submit from the repo root:
#   BENCH=_maskcon_new sbatch slurm/rescore_generation.sh
#   BENCH=_cifar100 sbatch slurm/rescore_generation.sh
# The controls are rebuilt and re-scored by slurm/eval_controls.sh (then scripts/metric_validation.py).
#SBATCH --job-name=lsgen-rescore
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
RUNGS=${RUNGS:-"coarse cluster_dinov3_base cluster_dinov3_base_khat cluster_dinov3_large cluster_dinov3_large_khat oracle"}
EVAL=configs/eval_generation$BENCH.yaml
GEN=$(python -c "from lsgen.utils import load_config; from lsgen.data.labels import suffix; c = load_config('$EVAL'); d = load_config(c['data_config']); print(d['derived_root'] + '/generation' + suffix(c['labels'], d))")

for rung in $RUNGS; do
    for dir in "$GEN"/${rung}_s*/step*_ddim25; do
        python scripts/eval_generation.py --config $EVAL --samples-dir "$dir" --use-case A
    done
    python scripts/summarize_generation.py --config configs/gen_$rung$BENCH.yaml
done
