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

## G-0c: generator pilot

Generator (`lsgen/methods/gen/ldm.py`): a frozen autoencoder, a MONAI `DiffusionModelUNet` on its latents with one class-label embedding
(label `n_classes` = null label for classifier-free guidance, dropped with p = 0.1), epsilon prediction on a 1000-step scaled-linear
schedule, EMA weights, DDIM sampling (eta = 0) with guidance w. The pilot conditions on the coarse class only; later variants change only the
label. Training sees images and coarse labels only (the train images and their flips, encoded once). Use case A sets are drawn with coarse
labels in train prevalence. Settings: `configs/gen_pilot_maskcon_new.yaml`, `configs/gen_pilot_cifar100.yaml`.

| | Stanford Cars `maskcon_new` | CIFAR-100 |
|---|---|---|
| images | `bbox15` 128 px | native 32 px |
| SD-VAE (`stabilityai/sd-vae-ft-ema`, frozen) | 128 px, f = 8 -> 16 x 16 x 4 | upsampled to 64 px, f = 8 -> 8 x 8 x 4 |
| own AutoencoderKL (train split; L1 + LPIPS + KL, PatchGAN after 30k steps) | f = 8 -> 16 x 16 x 4, LPIPS-AlexNet | f = 4 -> 8 x 8 x 4, LPIPS-VGG |
| UNet | channels 128 / 256 / 512, attention at the two lower levels, 2 res blocks | same |
| pilot | 200k steps, batch 128; checkpoints at 10k, 25k, 50k, 100k, 150k, 200k; DDIM 25 / 50 / 100 / 250; w = 1.5; 5,000 samples | same |

Everything is scored against **val** (method.json `reference_split: val`; G4/G5/G7 still use train), never test. Cars val has ~800
images, so its FID is biased upwards: rows are compared with each other, and KID (unbiased) is the deciding number.

Decision rules, fixed before the first run:
1. **Autoencoder**: the SD-VAE, unless the own autoencoder is better on both the reconstruction KID against val and the fine evaluator's
   accuracy on reconstructions (E1; how much subclass detail survives the autoencoder). The fine-accuracy drop from real val to
   reconstructions is also the ceiling on what any variant can show in G3 / G4.
2. **Budget** (training steps, the same for every variant and seed): the earliest checkpoint after which val KID at DDIM 250 improves by less
   than 10% at the next checkpoint, provided G7 copies stay within 0.05 of the real val rate (`G7 copies, real val`). If copying rises
   first, the last checkpoint before it rises.
3. **DDIM steps**: the fewest steps whose val KID at the chosen checkpoint is within 10% of DDIM 250.

Run order (per benchmark; `BENCH=_maskcon_new` or `BENCH=_cifar100`):

| Step | Command | Node |
|---|---|---|
| own autoencoder, val reconstructions of both, scores | `BENCH=$BENCH sbatch slurm/autoencoders.sh` | GPU |
| autoencoder table | `python scripts/summarize_pilot.py --config configs/gen_pilot$BENCH.yaml` | CPU |
| set `ldm.autoencoder` (and `ldm.name`) by rule 1, then train, sample every checkpoint, score, summarise | `BENCH=$BENCH sbatch slurm/gen_pilot.sh` | GPU |

Outputs: `<derived>/autoencoders[_<labels>]/own/`, `<derived>/generation[_<labels>]/{recon_<ae>, <run>/step_*.pt, <run>/step*_ddim*}/`;
scores in `<reports_root>/generation[_<labels>]/`, summary `<run>_pilot.{md,csv,png}`.

### G-0c results (21 Nov 2026; val only, seed 0)

Full tables: `reports/generation_maskcon_new/pilot_coarse_sd_vae_pilot.md`, `reports/cifar100/generation/pilot_coarse_sd_vae_pilot.md`.
The rules above were applied unchanged.

| | Stanford Cars `maskcon_new` | CIFAR-100 |
|---|---|---|
| reconstruction KID, own / SD-VAE | 0.0084 / **0.0051** | **0.0072** / 0.0130 |
| fine acc. on reconstructions, own / SD-VAE (real val) | 0.54 / **0.69** (0.85) | 0.61 / **0.72** (0.89) |
| rule 1: autoencoder | **SD-VAE** (better on both) | **SD-VAE** (own wins KID only) |
| val KID at DDIM 250 by checkpoint (10k, 25k, 50k, 100k, 150k, 200k) | .0377 .0217 **.0160** .0173 .0176 .0159 | .0328 .0268 .0229 .0194 **.0174** .0163 |
| G7 copies (real val rate) at those checkpoints | .002 .013 **.033** .097 .234 .391 (0.134) | .60 .59 .60 .60 **.60** .60 (0.71) |
| rule 2: budget | **50k steps** (0.40 GPU-h) | **150k steps** (1.19 GPU-h) |
| rule 3: DDIM steps (KID at the budget, 25 / 50 / 100 / 250) | **25** (.0148 .0146 .0151 .0160) | **25** (.0151 .0159 .0167 .0174) |

Throughput: 64M-parameter UNet, ~4,400 training images/s (batch 128, bf16); 5,000 samples take 42 s (Cars) / 10 s (CIFAR-100)
at DDIM 25. Guidance w = 1.5 throughout (not part of the pilot).

Observations (not used by the rules):
- The autoencoder caps subclass fidelity: the fine evaluator loses 16-17 points on SD-VAE reconstructions of real images on both benchmarks.
  Any generator's E1 / G4 is bounded by this.
- Cars memorises with training: G7 copies cross the real-val rate between 100k and 150k steps (7.3k train images), while G4 coverage keeps
  rising (0.53 at 50k, 0.81 at 200k). Past the budget, coverage and copying grow together.
- The coarse-only generator at the budget is weak relative to the controls: on Cars G3 TV 0.27 (C3 control 0.07 on test) and G5 excess
  KID 3.3 (C4 prototype-collapse control 2.1). CIFAR-100 is still improving at 200k; its budget is set by the 10% rule, not convergence.

## Generation ladder

Each rung is one config `configs/gen_<rung>_<benchmark>.yaml`, a copy of `gen_coarse_*` that changes only the conditioning label: the
G-0c budget (Cars 50k steps, CIFAR-100 150k), SD-VAE, DDIM 25, w = 1.5, seeds 0, 1, 2, a use-case-A set of about the test size, scored once
on test. `RUNG=<rung> BENCH=$BENCH sbatch slurm/generate.sh` trains every seed, samples, scores and runs `scripts/summarize_generation.py`
(`<reports_root>/generation[_<labels>]/<rung>.md`: mean and sd over seeds, per seed, links to figures).

Every use-case-A evaluation also writes figures next to the scores (`lsgen/eval/sample_plots.py`, descriptive only): `samples.png` (samples
by requested coarse class), `nearest_real.png` (random samples next to their nearest train image in CLIP-L space, captioned with the fine
evaluator's prediction and the neighbour's true class), `embedding.png` (UMAP / t-SNE of CLIP-L: the generated images among the real
reference images, and the generated set alone), and `g7_nn_grid.png` (the 64 samples closest to a train image in SSCD space).

| Rung | Conditioning | Status |
|---|---|---|
| coarse | coarse class only (lower bound) | done, 21 Nov 2026 (below) |
| cluster_dinov3_{large,base} | coarse + cluster id: frozen DINOv3-L / -B k-means within each class, true K_c ("uses K") | done 22 Nov 2026 |
| cluster_dinov3_{large,base}_khat | the same with the label-free K-hat per class (coarse labels only) | done 22 Nov 2026 |
| oracle | fine class (upper bound; uses fine labels) | done 22 Nov 2026 |
| next | coarse + cluster id from the ksub encoders; then geosub; support tokens (IP-Adapter-style) for use case B | to do |

Cluster rungs (`conditioning` in the config): one label per (coarse class, cluster) pair (`condition_labels`), read from a per-image
cluster file; the generator and its budget are unchanged. Cluster files are the frozen runs of `scripts/train_ksub.py`
(`<derived_root>/ksub_clusters/ksub_frozen<tag>_s<seed>.csv`: k-means on the frozen train features, nearest-centroid assignment, K-hat by
`estimate_k`, k_max 60); generator seed s uses the clustering of seed s, so the spread over seeds includes the clustering's. DINOv3-L files
exist (ksub runs); DINOv3-B ones come from `configs/ksub_<benchmark>_base_frozen{,_khat}.yaml`. The oracle's label file holds the fine
classes and is written by `scripts/oracle_labels.py` (on the fine-label allow list; training reads only the label file). Use case A
draws labels in train prevalence, so clusters in proportion to their train size within each class; samples.csv carries the requested
coarse class, cluster and `group` (G6). For the oracle, the evaluator also reports E1 (fine accuracy on the samples against the requested
fine class). Train label counts (seed 0): Cars 196 (true K) / 192 (K-hat, DINOv3-L), CIFAR-100 100 / 143.

### Coarse rung results (21 Nov 2026, test, 3 seeds; conflicting images excluded as in every split)

Mean (sd over seeds); full tables in `reports/generation_maskcon_new/coarse.md` and `reports/cifar100/generation/coarse.md`.

| | G1 FID | G1 recall | G2 coarse acc | G3 TV | G4 covered | G5 excess KID | G5 cov. ratio | G7 copies (real) |
|---|---|---|---|---|---|---|---|---|
| Cars maskcon_new | 22.5 (0.4) | 0.42 | 0.72 | 0.265 | 0.68 | 3.09 | 0.009 | 0.031 (0.130) |
| CIFAR-100 | 28.8 (0.3) | 0.57 | 0.71 | 0.134 | 0.98 | 0.94 | 0.47 | 0.54 (0.72) |

Seed sd is small (FID 0.3 to 0.4, G3 TV 0.004), so rung differences of a few points are resolvable. On Cars the generated set forms its
own region of CLIP-L space, apart from the real images (`embedding.png`), and its within-subclass variation is worse than the C4
prototype-collapse control (excess KID 3.09 vs 2.09): subclass metrics on Cars measure conditioning on top of a large realism gap. On
CIFAR-100 the generated set overlaps the real one, and G5 is clearly better than C4 (0.94 vs 2.50).

### Frozen DINOv3 cluster and oracle rungs (22 Nov 2026, test, 3 seeds; conflicting images excluded)

Mean over seeds (sd in `reports/generation_maskcon_new/<rung>.md`, `reports/cifar100/generation/<rung>.md`).

| | rung | labels | FID | recall | G3 TV | G4 covered | G4 rare | G5 excess KID | G6 purity | G7 copies (real) | E1 fine acc (real val) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cars | coarse | 8 | 22.5 | 0.42 | 0.265 | 0.68 | 0.58 | 3.09 | | 0.031 (0.130) | |
| Cars | DINOv3-B K-hat | 228 | 21.2 | 0.48 | 0.143 | 0.95 | 0.93 | 2.81 | 0.62 | 0.165 | |
| Cars | DINOv3-B true K | 196 | 20.9 | 0.48 | 0.151 | 0.94 | 0.93 | 2.83 | 0.61 | 0.154 | |
| Cars | DINOv3-L K-hat | 201 | 21.1 | 0.48 | 0.149 | 0.95 | 0.89 | 2.81 | 0.62 | 0.152 | |
| Cars | DINOv3-L true K | 196 | 21.2 | 0.48 | 0.148 | 0.95 | 0.91 | 2.80 | 0.63 | 0.162 | |
| Cars | oracle | 196 | 21.2 | 0.49 | 0.130 | 0.97 | 0.94 | 2.82 | 0.70 | 0.164 | 0.54 (0.85) |
| CIFAR-100 | coarse | 20 | 28.8 | 0.57 | 0.134 | 0.98 | | 0.94 | | 0.54 (0.72) | |
| CIFAR-100 | DINOv3-B K-hat | 111 | 27.1 | 0.60 | 0.077 | 1.00 | | 0.88 | 0.75 | 0.56 | |
| CIFAR-100 | DINOv3-B true K | 100 | 27.2 | 0.61 | 0.079 | 1.00 | | 0.87 | 0.78 | 0.55 | |
| CIFAR-100 | DINOv3-L K-hat | 143 | 27.2 | 0.60 | 0.077 | 1.00 | | 0.88 | 0.75 | 0.56 | |
| CIFAR-100 | DINOv3-L true K | 100 | 27.3 | 0.60 | 0.080 | 1.00 | | 0.88 | 0.79 | 0.55 | |
| CIFAR-100 | oracle | 100 | 27.4 | 0.60 | 0.076 | 1.00 | | 0.88 | 0.80 | 0.55 | 0.64 (0.89) |

- DINOv3-B and DINOv3-L, true K and K-hat, give the same generation: every metric within ~0.01 (seed sd up to 0.008), on both
  benchmarks, although the two encoders' K-hat differs (labels, mean over seeds: Cars 228 B / 201 L, CIFAR-100 111 B / 143 L). The
  finer Cars split of DINOv3-B leaves fewer real subclasses unmatched in G6 (0.75 vs 2.4 of 196) at slightly higher copying (G7 0.165).
- Any subclass label helps a lot over coarse-only (Cars G3 TV halves, G4 coverage 0.68 -> 0.95), and frozen DINOv3-L clusters, even with
  K-hat, are almost as good as the fine labels: the cluster-to-oracle gap is small (Cars G3 TV 0.148 vs 0.130, G6 purity 0.63 vs 0.70;
  none on CIFAR-100).
- The generator, not the subclass signal, bounds the ladder: with the true fine class requested, the fine evaluator recognises it in
  54% of Cars samples (85% on real val) and 64% on CIFAR-100 (89%); Cars G5 stays at 2.8 (C4 prototype-collapse control 2.09), and the
  oracle's samples still occupy their own region of CLIP-L space (`embedding.png`). Better encoders (ksub, geosub) can only move the
  narrow cluster-to-oracle gap at this generator.
- Finer labels raise copying on Cars (G7 0.03 -> 0.15-0.16, above the real rate 0.13; ~35 train images per label, some clusters of 2-6).

## CAS added to use case A (26 Nov 2026)

Classification Accuracy Score (docs/METRICS.md): a linear probe on CLIP-L fit on the generated set only, tested on real test images;
`cas_coarse` on the requested coarse class, `cas_sub` on the requested group (cluster / fine class / coarse class), its groups matched
one-to-one to the real fine classes within each coarse class. `cas_real_train.csv` is the ceiling (the same probe on real train).
Registered before its first run in `configs/metric_validation.yaml`: CAS subclass must fail for C4 and C5. The use-case-A controls now
record the fine class of each source image as `group` (so G6 and CAS sub are computed for them too).

| Step | Command | Node |
|---|---|---|
| controls rebuilt and re-scored | `BENCH=$BENCH sbatch slurm/eval_controls.sh` | GPU |
| controls x metrics (must show CAS subclass `ok` for C4, C5) | `python scripts/metric_validation.py --config configs/eval_generation$BENCH.yaml` | CPU |
| finished rungs re-scored and re-summarised (sets are not regenerated) | `BENCH=$BENCH sbatch slurm/rescore_generation.sh` | GPU |

## Generator backbones from the literature (official code)

The ladder shows that the generator, not the subclass signal, is the bound (oracle E1 0.54 on Cars vs 0.85 real). Stronger published
generators are run with their **official implementations**, pinned as git submodules in `external/` and not modified: our code only writes
the training data in each repo's format, launches its training script, and samples with its own model and sampler code using our labels
(use case A needs labels in train prevalence; the official sample scripts draw them uniformly). The conditioning labels, the evaluation and
the G-0c budget rules are the ladder's. Stanford Cars `maskcon_new` only (decided 26 Nov 2026): the official scripts take 256 / 512 px only,
so they train on the `bbox15_256` crops and their samples are resized to the scored 128 px; CIFAR-100 (32 px) would need 8x upsampling.
Resolution therefore differs from the 128 px UNet ladder; recorded as `train_image_size` in method.json.

| Backbone | Official code | What it trains | Extra pretrained parts (`uses`) | Status |
|---|---|---|---|---|
| DiT-B/2 (Peebles & Xie 2023) | `external/DiT` (facebookresearch/DiT `ed81ce2`, CC-BY-NC 4.0) | DiT on SD-VAE latents (32 x 32 x 4), epsilon + learned sigma, 1000-step linear schedule, label dropout 0.1, EMA, AdamW 1e-4, batch 256, fp32/TF32 | sd_vae | pilot ready |
| SiT-B/2 + REPA (Yu et al. 2025) | sihyun-yu/REPA (`67f7145`, MIT) | SiT (velocity, linear path) + alignment of block-8 tokens to frozen DINOv2-B; `--proj-coeff 0` is plain SiT from the same code | sd_vae, dinov2_b | next |
| RAE (Zheng et al. 2025) | bytetriper/RAE (`a4d18c4`, MIT) | DiT-DH on frozen DINOv2-B (registers) features, 16 x 16 x 768; released ImageNet ViT-XL decoder and normalisation statistics | dinov2_b_reg, rae_decoder (ImageNet) | after REPA |

None of the extra encoders is an evaluator; DINOv2 is not the DINOv3 of the cluster methods. Per backbone: a pilot on coarse labels against
val sets its own budget and DDIM steps with the G-0c rules 2 and 3, then the coarse and oracle rungs on test (3 seeds), then the cluster
rungs if the oracle's E1 rises.

### DiT

`scripts/train_dit.py` writes the train split as an ImageFolder (`lsgen.methods.gen.ldm.write_imagefolder`: symlinks to the 256 px crops,
conflicting images excluded, one folder per conditioning label) and `labels.json`, then runs DiT's `train.py` unchanged through `torchrun`
for the whole epochs that reach `dit.steps` (checkpoints every `dit.ckpt_every`). `scripts/sample_dit.py` loads the EMA weights into DiT's
model and samples with DiT's `ddim_sample_loop` and `forward_with_cfg` (guidance 1.5, applied to the first 3 latent channels as published),
decodes with the SD-VAE and resizes to 128 px. Pilot: `configs/gen_dit_pilot_maskcon_new.yaml` (150k steps, checkpoints every 25k, DDIM
25 / 50 / 100 / 250, 5,000 samples, val).

| Step | Command | Node |
|---|---|---|
| train (one 24 h job; DiT has no resume: check `Train Steps/Sec` in `<run>/results/000-DiT-B-2/log.txt`, 150k steps need >= 1.8) | `CONFIG=configs/gen_dit_pilot_maskcon_new.yaml sbatch slurm/train_dit.sh` | GPU |
| sample every checkpoint, score against val, pilot summary | `CONFIG=configs/gen_dit_pilot_maskcon_new.yaml sbatch --dependency=afterok:<train job> slurm/sample_dit.sh` | GPU |
