CONFIG ?= configs/data.yaml

.PHONY: prepare episodes test
# heavy: submit from a GPU node with `sbatch slurm/prepare_data.sh` (runs prepare + episodes)
prepare:
	python scripts/prepare_data.py --config $(CONFIG)

# CPU, seconds; needs `prepare` up to the finalize stage
episodes:
	python scripts/build_episodes.py --config configs/episodes.yaml

test:
	python -m pytest -q
