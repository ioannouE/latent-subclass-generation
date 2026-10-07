# latent-subclass-generation

Benchmark for subclass-preserving image generation on Stanford Cars (coarse label = make, hidden subclass = make-model-year).
Step 1 builds the data, a validated evaluation library and the frozen-encoder reference; no generative model is trained here.
Plan: [docs/PLAN.md](docs/PLAN.md). Data protocol: [docs/DATA.md](docs/DATA.md). Every metric: [docs/METRICS.md](docs/METRICS.md).
Results: [reports/report.md](reports/report.md).

## Setup
```bash
conda activate diffusion            # python >= 3.10, PyTorch with CUDA
pip install -e ".[dev]"
make test                           # CPU, ~20 s
```
Paths live in [configs/data.yaml](configs/data.yaml): the read-only raw dataset (`raw_root`), the derived data (crops, features,
weights; outside git) and the frozen artefacts committed under `data/`. Needed once: network access for the SSCD weights, the
DINOv3 / CLIP weights (Hugging Face cache) and clean-fid's Inception weights.

## From raw data to the report
Heavy stages are Slurm jobs (partition and account in each script); run them from the repo root, each after the previous finished.
Times are measured on one GPU node with 8-16 CPUs.

| # | Command | Runs on | Time | Output |
|---|---|---|---|---|
| 1 | `sbatch slurm/prepare_data.sh` | GPU | ~3 min (+ SSCD download) | crops (128/256 px), `metadata.parquet`, splits, episodes, near-duplicate audit, `reports/{data_summary,preprocessing,episodes}.md` |
| 2 | `sbatch slurm/train_eval_classifier.sh` | GPU | ~3 min | fine / make evaluator classifiers, `reports/evaluator_*.json` |
| 3 | `sbatch slurm/extract_features.sh` | GPU | ~10 min first time (cached embeddings are skipped) | embeddings of every encoder, incl. Inception and the oracle classifier features |
| 4 | `sbatch slurm/eval_representation.sh` | CPU | ~5 h at 1000 resamples (8 runs; SSCD is the slowest) | `reports/repr/<encoder>_<crop>/` (R1-R5) |
| 5 | `CONFIG=configs/eval_repr_oracle.yaml sbatch slurm/eval_representation.sh` | CPU | ~20 min | the "supervised oracle features" reference row |
| 6 | `sbatch slurm/eval_controls.sh` | GPU | ~40 min | control sets (C1-C6, use cases A and B) and their G / B scores in `reports/generation/` |
| 7 | `make report` | CPU | seconds | `reports/metric_validation.{csv,md,png}`, then `reports/report.md`, `reports/repr_frozen.csv`, `reports/repr_per_make.csv` |

`make report` first runs `scripts/metric_validation.py`, which exits with an error if a control did not fail a metric it must fail:
stop and report, do not tune. Figures (`*.png`) are regenerated and not committed.

To score a method's generated images: put them (128 px PNG), `samples.csv` and `method.json` in one folder, set `samples_dir` and
`use_case` in [configs/eval_generation.yaml](configs/eval_generation.yaml) and run `sbatch slurm/eval_generation.sh`
(formats: header of that config; the evaluator-overlap check is in `lsgen/eval/evaluators.py`).

## Step 2 baselines (already in the repo)
`sbatch slurm/train_repr.sh` fine-tunes SupCon, SimCLR, SupCon+SimCLR, MaskCon and FALCON on a DINOv3-B backbone with make labels only
(~10-15 min per method); score them with `CONFIG=configs/eval_repr_baselines.yaml sbatch slurm/eval_representation.sh` (~1 h).
Results: [reports/repr/summary.md](reports/repr/summary.md).

## Layout
`lsgen/data` splits, crops, episodes - `lsgen/features` encoders and the embedding cache - `lsgen/eval` metrics, evaluators, controls,
validation - `lsgen/methods` baselines - `scripts/`, `slurm/`, `configs/` one experiment per config - `tests/` synthetic cases with known answers.
Fine labels are only read inside `lsgen/eval/` and in split / episode construction; the `test` split is never used for tuning;
conflicting images (`data/splits/exclude.txt`) are left out of every experiment.
