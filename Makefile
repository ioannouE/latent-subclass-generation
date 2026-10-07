CONFIG ?= configs/data.yaml

.PHONY: prepare episodes features eval-representation eval-generation controls metric-validation report test
# heavy: submit from a GPU node with `sbatch slurm/prepare_data.sh` (runs prepare + episodes)
prepare:
	python scripts/prepare_data.py --config $(CONFIG)

# CPU, seconds; needs `prepare` up to the finalize stage
episodes:
	python scripts/build_episodes.py --config configs/episodes.yaml

# GPU, minutes: cached embeddings are skipped; submit with `sbatch slurm/extract_features.sh`
features:
	python scripts/extract_features.py --config configs/features.yaml

# CPU, hours at n_boot=1000: submit with `sbatch slurm/eval_representation.sh`
eval-representation:
	python scripts/eval_representation.py --config configs/eval_representation.yaml

# GPU, one generated set (configs/eval_generation.yaml): submit with `sbatch slurm/eval_generation.sh`
eval-generation:
	python scripts/eval_generation.py --config configs/eval_generation.yaml

# GPU, hours: builds the control sets and scores each one; submit with `sbatch slurm/eval_controls.sh`
controls:
	python scripts/make_controls.py --config configs/controls.yaml

# CPU, seconds: controls x metrics table and heat-map from the control results
metric-validation:
	python scripts/metric_validation.py --config configs/metric_validation.yaml

# CPU, seconds: the metric-validation table (exits 1 if a control missed a metric), then reports/report.md
report: metric-validation
	python scripts/make_report.py --config configs/report.yaml

test:
	python -m pytest -q
