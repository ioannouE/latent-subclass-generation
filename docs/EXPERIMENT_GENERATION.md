# Generation baselines

One diffusion generator per dataset, trained identically under a ladder of conditioning signals. Only the conditioning changes, so
differences in G1-G7 and B1-B6 can be attributed to the subclass signal (docs/PLAN.md A6, tier 3). Published generative methods run beside it
as reference points, in separate, labelled tables.

## Prerequisites (before any generator)

| ID | Task | Why |
|---|---|---|
| G-0a | Decided 18 Nov 2026: Stanford Cars generation uses `maskcon_new` (8 body types / 196 classes), as the representation workstream | The suite was validated on `make` (189) only |
| G-0b | Retrain the evaluator classifiers (196-way fine, 8-way coarse), rebuild the episodes and re-run controls C1-C6 on `maskcon_new`; the same for CIFAR-100 | Every metric must be re-validated on each benchmark before a method is scored |
| G-0c | Pilot the MONAI latent UNet with coarse-only conditioning: throughput, DDIM steps, copying (G7) and FID against training steps; choose the autoencoder (own AutoencoderKL or frozen SD-VAE) | Sets one fixed training budget, equal for every variant |

## G-0b: re-validating the metrics

The evaluation scripts take the label scheme from their config (`labels:`). Outputs of the data config's own scheme keep their paths
(`make` on Stanford Cars, `cifar100` on CIFAR-100); another scheme on the same images gets a suffix (`lsgen.data.labels.suffix`), so the
validated `make` results are not overwritten:

| Output | Stanford Cars, `maskcon_new` | CIFAR-100 |
|---|---|---|
| evaluators | `<derived>/evaluators_maskcon_new/{fine,make}/`, `reports/evaluator_maskcon_new_{fine,make}.json` | `<derived>/evaluators/{fine,make}/`, `reports/cifar100/evaluator_{fine,make}.json` |
| episodes | `data/episodes_maskcon_new/`, `reports/episodes_maskcon_new.md` | `data/cifar100/episodes/`, `reports/cifar100/episodes.md` |
| controls | `<derived>/controls_maskcon_new/` | `<derived>/controls/` |
| control scores | `reports/generation_maskcon_new/` | `reports/cifar100/generation/` |
| controls x metrics | `reports/metric_validation_maskcon_new.{csv,md,png}` | `reports/cifar100/metric_validation.{csv,md,png}` |

The metric list, expected failures and margin are the ones in `configs/metric_validation.yaml`, fixed for Stanford Cars `make` and not
changed per benchmark. "make" in the metric names means the coarse class (body type, superclass).

Differences from Stanford Cars `make`, all fixed before the first run:
- `maskcon_new`: conflicting images are `data/splits/exclude_maskcon_new.txt` (docs/DATA.md); near-duplicate tables for S x T as for `make`.
- CIFAR-100: evaluators see the 32 px images upsampled to 224 (bicubic, the same for generated images; ConvNeXt-T downsamples 32x) and train
  for 15 epochs; S x T near-duplicates come from `duplicates_all.csv` at SSCD >= 0.75 or exact copies; Inception features are added to the
  cache for G1. Every class has 450 train images minus exclusions, so G4's "rare" quartile is not meaningful on CIFAR-100 (docs/DATA_CIFAR100.md).

Run order (per benchmark; `BENCH=_maskcon_new` or `BENCH=_cifar100`):

| Step | Command | Node |
|---|---|---|
| CIFAR-100 Inception features | `CONFIG=configs/features_cifar100.yaml sbatch slurm/extract_features.sh` | GPU |
| evaluators | `CONFIG=configs/classifier$BENCH.yaml sbatch slurm/train_eval_classifier.sh` | GPU |
| episodes | `python scripts/build_episodes.py --config configs/episodes$BENCH.yaml` | CPU |
| controls + scores | `BENCH=$BENCH sbatch slurm/eval_controls.sh` (after evaluators and episodes) | GPU |
| controls x metrics | `python scripts/metric_validation.py --config configs/eval_generation$BENCH.yaml` | CPU |
