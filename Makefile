CONFIG ?= configs/data.yaml

.PHONY: prepare test
prepare:
	python scripts/prepare_data.py --config $(CONFIG)

test:
	python -m pytest -q
