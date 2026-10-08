# Subclass-Preserving Generation — Benchmark Plan & Coding-Agent Prompts

Version 1 · 29 Sep 2026 · for the problem statement "Learning Latent Subclasses for Distribution-Preserving Image Generation" (v0.5)

Decisions fixed for this plan: Stanford Cars, **coarse = make (49), hidden subclass = make-model-year (196)** (superseded 7 Oct 2026: hidden subclass = **make-model, 189**, see docs/DATA.md); single GPU (24–48 GB); **128 px bounding-box crops** as the main resolution; fresh repository.

---

## Part A — The design (what you are building and why)

### A1. The principle

The statement says: *quantify how well existing methods meet the goal before inventing anything.* So the first paper-grade asset is a **benchmark whose metrics are validated before any method is scored**. Every metric must (i) map to one of the three objectives or one of the two use cases, (ii) have a real-vs-real reference score, and (iii) be shown to *fail* on a control built to break it. That last property is what makes reviewers trust the tables, and it is cheap to build.

### A2. Data protocol

| Item | Choice | Reason |
|---|---|---|
| Raw data | Already downloaded: `/nvme/h/eioannou/data_p315/Stanford_Cars/kaggle/` (`car_devkit/`, `cars_test/`, `cars_train/`, `devkit/`), read-only | No re-download; derived data lives in the repo |
| Coarse label | Make (49) | Given at training time |
| Hidden subclass | Make-model-year (196); secondary level make-model (years merged) | Natural, uneven K_c (1 to 20+), rare subclasses exist without engineering |
| Makes with K_c = 1 | Kept | Tests over-fragmentation: a good method should *not* split them |
| Image (main) | `bbox15`: square crop of side 1.5 × the bbox's longer side (FineGAN/StackGAN convention), shifted to stay inside the image, never stretched; 128 px (256 px stored too). Full spec in A2b | Car fills the frame at full detail; matches the most common convention on Cars; feasible on 1 GPU |
| Image (ablation) | `full_cc`: whole image, short side → 128/256, centre crop (ADM / C3-GAN convention) | No bounding boxes needed, which matters for the medical transfer later |
| `train` | Official train minus val (≈90%) | All method training, coarse labels only |
| `val` | ≈10% of official train, stratified by fine label | Tuning. Fine labels used *only* to build the split (documented) |
| `test` | Official test (8,041) | Held-out real reference. Never used for tuning |
| `test_A` / `test_B` | Stratified halves of `test` | Real-vs-real noise floor at matched n |
| Episodes (use case B) | Frozen JSON files built from `test` before any method runs | Support sets are unseen images, targets are the *rest* of that subclass |

Fine labels are allowed in exactly one place: the evaluation package. Training datasets must not return them.

### A2b. Image preprocessing

**What the closest methods do on Stanford Cars** (read from their released code; OneGAN from its paper)

| Method | Bbox? | Pipeline | Effect on a wide car |
|---|---|---|---|
| FineGAN | Yes | Square of side 1.5·max(w,h) centred on the bbox, **clamped** at image borders (so often not square) → short side to 152 → random 128 crop | No stretching; ends can be cut |
| OneGAN | Yes | Bbox-based, 128 px; bboxes also cut background patches | — |
| C3-GAN | No | Short side → 128 → random crop (train) / centre crop (test) | Ends cut |
| MaskCon | No | Train: RandomResizedCrop 224; test: `Resize([224,224])` | Test images stretched |

**Our main variant, `bbox15`** — FineGAN's crop size and centre, with two fixes: the window is *shifted* instead of clamped, and it is never resized to a non-square shape.

1. Load with PIL and convert to RGB (Stanford Cars contains some greyscale images). Do **not** apply EXIF rotation: the bboxes refer to the stored pixels.
2. Devkit bboxes are MATLAB 1-indexed and inclusive: `x1 -= 1; y1 -= 1`, keep `x2, y2`, clip to the image. `bw = x2 - x1`, `bh = y2 - y1`, `m = max(bw, bh)`, centre `(cx, cy)`.
3. Target side `s = 1.5 · m` (FineGAN/StackGAN). If `s > min(W, H)`, reduce it to `s = max(min(W, H), 1.02 · m)`: the context margin shrinks before any padding is added, and the car always stays fully inside.
4. Place the square at the bbox centre, then **shift** it (don't shrink it) until it lies inside the image on every axis where it fits.
5. Only if `s` still exceeds an image dimension (a car wider than the photo is tall), keep that whole dimension and split the padding equally on both sides, filled with a constant ImageNet-mean colour `(124, 116, 104)`. No edge replication, no reflection.
6. Resize the square crop straight to 256 and to 128 with PIL `LANCZOS` (antialiased). Don't make the 128 from the 256.
7. Save as lossless PNG (uint8), named `{official_split}_{fname_stem}.png`. Record per image: the crop box in original coordinates, the scale, the pad on each side, `pad_fraction`, and the margin actually used (`s / m`).
8. Training augmentation is applied at load time on top of the stored PNGs: horizontal flip only by default. An optional `finegan_jitter` (resize to 152, random 128 crop) is labelled wherever it is used. No augmentation when evaluating.

**Ablation variant, `full_cc`**: the whole image with the ADM `center_crop_arr` recipe: repeated BOX halving while the short side is ≥ 2× the target, then BICUBIC resize of the short side to 128/256, then a centre crop. This is the standard for class-conditional generation and matches C3-GAN at evaluation time.

**Native pipelines are used only for reproduction.** Each external baseline is first run with its own preprocessing, to check we can reach its published numbers. Every number in a comparison table comes from our stored `bbox15` PNGs, evaluated through our pipeline.

**Evaluator input.** Real and generated images go through exactly the same path: 128 px PNG → each evaluator's standard resize (clean-fid for Inception, the model's own processor for CLIP/DINOv2/SSCD). Generated images are saved as PNG, never JPEG.

### A3. Evaluators (circularity rule)

Evaluators must be disjoint from anything a method uses for conditioning or training.

- **Primary**: (1) a fine-grained classifier (e.g. ConvNeXt-T, ImageNet init) trained on `train` with fine labels at 128 px, temperature-calibrated on `val` — evaluation only; (2) Inception-v3 (FID/KID, clean-fid); (3) a non-DINO SSL/VLM encoder (CLIP ViT-L/14 or SigLIP2).
- **Secondary**: DINOv2 ViT-L (FD_DINOv2). Flag as "same family" whenever a method uses DINOv3.
- **Copy detection**: SSCD.
- A runtime check refuses to evaluate a method with an evaluator it declares it used.

### A4. Metric panel

**Representation level** (test embeddings; fine labels for evaluation only; per-make tables + macro means + 95% bootstrap CIs)

| ID | Objective | Metric |
|---|---|---|
| R1 | O1 coarse | kNN@20 and linear-probe make accuracy |
| R2 | O2 subclasses | Per make with K_c ≥ 2: k-means with oracle K (labelled "uses K") and label-free K̂; ACC (Hungarian), NMI, ARI; within-make fine Recall@1 (MaskCon protocol); within-make fine linear probe (is the information present at all?) |
| R3 | O1+O2 geometry | On L2-normalised embeddings: Δ0, Δ1, Δ2 per make; ratios Δ0/Δ1, Δ1/Δ2; % of makes satisfying Eq. (1) and (2) |
| R4 | O3 variation | Within-subclass trace-cov ÷ within-make trace-cov; participation-ratio effective rank; "variation retention" vs frozen reference |
| R5 | Stability | ARI between clusterings across seeds and across bootstrap resamples; K̂ error for K_c = 1 makes |

**Use case A — distribution-preserving dataset synthesis** (N = |test|, 3 seeds, all vs `test`)

| ID | Target | Metric |
|---|---|---|
| G1 | Realism | FID, KID, FD_DINOv2; precision/recall; density/coverage |
| G2 | O1 | Evaluator make accuracy; make-prevalence TV distance |
| G3 | O2 prevalence | Per make: TV and smoothed KL between evaluator fine-label histogram of samples and of `test` |
| G4 | Rare subclasses | Rare = bottom quartile by train frequency. % of fine classes with ≥ m confident samples; per-class coverage, rare vs common |
| G5 | O3 within-subclass | Per fine class: KID and coverage (samples assigned to g vs test_g), reported as a ratio to the real ceiling (train_g vs test_g, same n) |
| G6 | Group alignment (methods with explicit groups) | Per make: KID cost matrix between generated groups and real subclasses → Hungarian; group purity / histogram entropy (TreeDiffusion-style); # unmatched real subclasses |
| G7 | Copying | SSCD nearest-train similarity: distribution, % > 0.5, against the null of test→train similarities; NN grids saved |

**Use case B — example-guided synthesis** (k ∈ {1, 5, 10}; ≥ 5 episodes per eligible subclass)

| ID | Metric |
|---|---|
| B1 | Hit rate: evaluator predicts the support's subclass |
| B2 | **Beyond-support fidelity**: KID and coverage vs T = test_g \ S, and separately vs S |
| B3 | Exemplar dependence: mean max-SSCD to S vs to T, relative to the real reference (T-to-S similarity) |
| B4 | Diversity: intra-episode pairwise distance vs pairwise distance within T |
| B5 | Mixed supports (two subclasses of one make, ratio r): error between the predicted proportion and r |
| B6 | Scaling of B1–B4 with k (does collective support help, per the statement) |

### A5. Controls (these validate the metrics)

| Control | Built from | Must fail |
|---|---|---|
| C1 Copy-train | Train images returned as "samples" | G7 (and B3 for copy-support) |
| C1b Copy-support / Retrieve-NN | Support S (augmented) / nearest train images to S | B2-vs-T, B3 |
| C2 Oracle split | test_A as samples, test_B as reference | Nothing. Defines the noise floor |
| C3 Coarse-only | Correct make, subclass uniform within make | G3, G4 |
| C4 Prototype collapse | Medoid of each subclass repeated | G5, B4, R4-style variation |
| C5 Majority-only | Largest subclass per make only | G3, G4 |
| C6 Degraded | Blur/JPEG of real images | G1 |

The metric-validation table (controls × metrics) is itself a figure for the paper.

### A6. Baselines, in tiers

1. **Representation**: frozen DINOv2 / DINOv3 / CLIP / SigLIP2 / supervised ResNet-50; SupCon (coarse); SupCon + SimCLR; MaskCon; FALCON; TFB; SCGM (if code exists); fine-label SupCon oracle.
2. **Published generative clustering, run as published**: C3-GAN (official Stanford Cars checkpoint: verify ACC/NMI/FID against the paper first), FineGAN, TreeDiffusion / TreeVAE, OneGAN if code runs. These use *no* coarse labels, so report them in a separately labelled "less supervision" block.
3. **Controlled shared generator** (the fair comparison): one latent diffusion backbone (SD-VAE latents, 128 px → 16×16×4, DiT-B/2 or a small UNet), trained identically with different conditioning:
   - coarse only (lower bound);
   - coarse + cluster id from [frozen encoder → per-make clustering with K̂] (the "obvious two-stage" baseline);
   - coarse + cluster id from the MaskCon / FALCON encoders;
   - representation-conditioned (RCG-style): use case A samples embeddings per cluster, use case B conditions on support tokens (multi-token cross-attention, no pooling);
   - fine-label conditional (oracle upper bound).
4. **Best-available** (separate table, external pretraining disclosed): DataDream-style LoRA on Stable Diffusion for use case B.

### A7. Roadmap

| Step | Deliverable | Gate to next step |
|---|---|---|
| 1 | Data, splits, episodes, feature cache, full evaluation library, controls, frozen-encoder representation table | Every control fails the metrics it should; the real-vs-real floor is reported |
| 2 | Representation baselines (tier 1) | One published number reproduced per ported method (on the paper's own dataset) |
| 3 | Generative baselines (tiers 2–4) at 3 seeds | C3-GAN checkpoint numbers match the paper within noise |
| 4 | Objective-by-method failure matrix and a diagnosis of *why* the best baseline fails | A concrete, measured limitation → only then design the method |

---

## Part B — Prompt for the coding agent: Step 1

> Copy everything inside the block into a new session in an empty repository.

```text
You are setting up a new research repository for a CVPR submission. Work step by step,
commit after each numbered milestone, and stop at the end of each milestone to report
what exists, what was verified, and anything that deviates from this spec. Do not
implement any generative model or new method in this step.

# Repository
Remote: https://github.com/ioannouE/latent-subclass-generation.git
Clone it (or, if it is empty, initialise locally and set it as `origin`), work on a
feature branch per milestone (e.g. `step1/m1-data`), and push after each milestone commit.
Python package name: `lsgen` (do not use `scgen`, which collides with an existing PyPI
package). Never commit raw data, cached features, checkpoints, or generated images: add
them to .gitignore and keep them under the configured data/output roots.

# Context
Research question: given images with coarse labels only, can a model discover the hidden
subclasses inside each coarse class and generate images that preserve (O1) coarse-class
distinctions, (O2) the subclasses and their prevalence, and (O3) the variation within each
subclass? Two use cases: (A) distribution-preserving dataset synthesis; (B) example-guided
synthesis from a support set S of k images of one subclass, where generation should draw
from that subclass's learned distribution, not merely vary S.
Dataset: Stanford Cars. Coarse label = make (49). Hidden subclass = the 196 official
make-model-year classes. Secondary hidden level: make-model (years merged).
Compute: one GPU with 24-48 GB. Main resolution 128 px; also store 256 px.
STEP 1 GOAL: data preparation, frozen splits and episodes, feature caching, a validated
evaluation library, control "generators", and a representation table for frozen encoders.

# Hard rules
1. Fine labels are used ONLY inside `lsgen/eval/` and in split/episode construction. Training
   dataset classes must not return fine labels (write a test that asserts this).
2. The `test` split is never used for tuning or model selection.
3. Evaluator/pipeline disjointness: every method declares the encoders it uses; the eval
   entry point refuses to score it with an overlapping evaluator (and flags same-family pairs,
   e.g. DINOv2 vs DINOv3).
4. Everything seeded and deterministic where possible; every output file records git hash,
   config, seed, and dataset manifest hash.
5. Every metric is reported per coarse class and as a macro mean, with 95% bootstrap CIs
   (1000 resamples, resampling images within class).
6. Never silently swallow errors or skip classes; log skipped items with the reason.

# Repository layout (Python >= 3.10, PyTorch, uv or conda, pinned lockfile)
lsgen/
  data/        stanford_cars.py, hierarchy.py, crops.py, splits.py, episodes.py
  features/    encoders.py, extract.py        (cache to CSV: filename, z1..zD; + JSON manifest)
  eval/        representation.py, synthesis.py (use case A), support.py (use case B),
               memorization.py, evaluators.py, stats.py (bootstrap, CIs), report.py
  controls/    controls.py
configs/       YAML (one file per experiment; no hard-coded paths)
scripts/       prepare_data.py, build_episodes.py, extract_features.py,
               train_eval_classifier.py, eval_representation.py, eval_generation.py,
               run_controls.py, make_report.py
tests/         pytest; synthetic Gaussian-mixture fixtures with known answers
docs/          DATA.md, METRICS.md (exact definition of each metric), PROTOCOL.md
Makefile       prepare, features, controls, report, test

# Milestone 1: data
- The raw data is ALREADY DOWNLOADED (Kaggle mirror). Do not download it again. Root:
    /nvme/h/eioannou/data_p315/Stanford_Cars/kaggle/
  containing the folders: car_devkit/  cars_test/  cars_train/  devkit/
  Treat this directory as read-only; write all derived data (crops, parquet, splits) into
  the repo's data directory, configured via a single `raw_root` / `data_root` setting in the
  config (no hard-coded paths elsewhere).
- First inspect it and report: the contents of car_devkit/ and devkit/ (which one holds
  cars_meta.mat, cars_train_annos.mat, and test annotations WITH class labels, e.g.
  cars_test_annos_withlabels.mat; whether the two devkits differ), and whether cars_train/
  and cars_test/ hold images directly or in a nested subfolder. If the test labels are
  missing, stop and tell me; do not proceed with an unlabeled test split.
- Verify: 16,185 images; 8,144 train / 8,041 test; 196 classes; one bbox per image; every
  annotation points to an existing, readable image. Store SHA256 checksums in a manifest.
- hierarchy.py: parse the 196 class names into make / model / year. Handle multi-word makes
  with an explicit, reviewed list (e.g. "AM General", "Aston Martin", "Land Rover", and check
  all others by hand). Write data/hierarchy.csv. Assert 49 makes. Print K_c per make, and list
  makes with K_c = 1. Also produce the make-model level.
- crops.py: implement exactly the two variants in Part A2b of docs/PLAN.md, as pure functions
  `crop_bbox15(img, bbox) -> (square_img, meta)` and `crop_full_cc(img, size)`:
  * bbox15: convert to RGB, no EXIF transpose; convert the 1-indexed inclusive devkit bbox
    (x1-=1, y1-=1); m = max(bw, bh); s = 1.5*m; if s > min(W, H) then
    s = max(min(W, H), 1.02*m); centre on the bbox, SHIFT the window inside the image (never
    shrink or clamp it into a non-square); on an axis where s > image size keep the whole axis and split the padding equally, constant
    (124, 116, 104); resize the square directly to 256 and to 128 with PIL LANCZOS.
    NEVER resize a non-square region to a square.
  * full_cc: ADM center_crop_arr (BOX halving while short side >= 2*size, BICUBIC short
    side -> size, centre crop), at 128 and 256.
  Write derived/{bbox15,full_cc}_{128,256}/{official_split}_{stem}.png plus a metadata
  parquet (id, path, official_split, make_id, fine_id, model_id, bbox, crop_box, scale,
  pad_l/t/r/b, pad_fraction, margin_used = s/m, was_grayscale, orig_W, orig_H) and a manifest
  with SHA256 of every output file. Deterministic: two runs must give byte-identical files.
- Preprocessing checks (write reports/preprocessing.md):
  * Unit tests: the car's aspect ratio (bbox w/h) is preserved within 1%; the transformed
    bbox lies fully inside the crop; the 1-indexed conversion is correct on a synthetic image.
  * Statistics: % of images needing any padding; histogram of pad_fraction; histogram of
    margin_used (how often we fall below 1.5); number of greyscale images; original size
    distribution.
  * Grids: 64 random crops, the 32 most-padded crops, the 32 smallest margin_used, and 16
    bbox15 vs full_cc pairs.
  * If more than 10% of images have pad_fraction > 0.10, STOP and report before moving on.
- Native-preprocessing adapters (used only in Step 3 for reproducing published numbers):
  `native_finegan` (1.5x bbox clamped, short side 152, random 128 crop) and `native_c3gan`
  (short side 128, centre crop). Keep them in lsgen/data/native.py and never use them for
  comparison tables.
- Near-duplicate audit: pHash + SSCD between train and test; write the list of near-duplicate
  pairs to data/duplicates.csv and report the count. Do not delete them; mark them.
- splits.py: val = 10% of official train, stratified by fine_id (seed 0); train = the rest;
  test = official test; test_A / test_B = stratified halves of test. Save as ID lists.
- Report: images per make, per fine class, and a K_c histogram. Save a figure.

# Milestone 2: episodes (use case B), frozen before any method runs
- build_episodes.py: for k in {1, 5, 10} and for every fine class with >= k + 10 test images,
  create 5 episodes: support S (k test images) and target T = remaining test images of that
  class. Also "mixed" episodes: S drawn from two subclasses of the same make in ratios
  {0.2, 0.5, 0.8} (k = 10), and a support_source = "train" variant (S from train images).
  Save as JSON with seeds. Report how many classes are eligible per k.

# Milestone 3: features and evaluators
- features/encoders.py: a uniform interface (preprocess, embed, dim, family) for: DINOv2
  ViT-B/14 and ViT-L/14; DINOv3 ViT-B/16 and ViT-L/16 (loaded through timm, e.g.
  `vit_base_patch16_dinov3.lvd1689m`, as in med-img-gen/zero_shot/encoders.py; the weights are in the HF cache);
  CLIP ViT-L/14; SigLIP2; torchvision ResNet-50 (supervised ImageNet);
  Inception-v3 pool3 (use clean-fid's implementation); SSCD (disc_mixup or disc_large).
  Embedding = CLS/pooled output, as in zero_shot/extract_embeddings.py. Cache embeddings for train/val/test at 128
  and 256 px (native model resolution, same resize path for real and generated images).
- Embedding file format (all encoders, real and generated): CSV, first column `filename`, then one column per latent
  dimension `z1, z2, ..., zD`; one row per image; a JSON manifest next to it (encoder, input size, split, n, dim, git hash).
- Conflicting images (docs/DATA.md, `data/splits/exclude.txt`) are left out of every experiment, from feature
  extraction onwards, and every report states it.
- train_eval_classifier.py: evaluator-only classifiers trained on `train` at 128 px:
  (a) fine (196), (b) make (49). ConvNeXt-T, ImageNet init, standard augmentation, model
  selection on val, temperature scaling on val. Report test top-1 and ECE. These are
  EVALUATORS; no method may ever load them.

# Milestone 4: evaluation library (see docs/METRICS.md, which you must write)
Implement exactly these, each with a unit test on synthetic data with a known answer.

Representation (inputs: embeddings, make, fine; L2-normalise first):
- R1: make kNN@20 accuracy; logistic-regression linear-probe make accuracy (fit on train
  embeddings, score on test).
- R2: per make with K_c >= 2: k-means with oracle K (label as "uses K") and with a
  label-free K-hat (silhouette sweep over K in [1, 25], with a gap-statistic
  fallback); ACC (Hungarian), NMI, ARI; within-make fine Recall@1; within-make fine linear
  probe.
- R3: Delta0 (same fine), Delta1 (same make, different fine), Delta2 (different make) as mean
  squared distances excluding self-pairs; per-make ratios; % of makes with
  Delta0 < Delta1 and Delta1 < Delta2.
- R4: within-subclass trace covariance / within-make trace covariance; participation-ratio
  effective rank per subclass.
- R5: ARI between clusterings over 3 seeds and 20 bootstrap resamples; K-hat on K_c = 1 makes.

Use case A (inputs: folder of generated images + CSV with requested make and optional
group id; reference = test; n matched):
- G1: FID and KID (clean-fid), FD with DINOv2-L, precision/recall (k=5), density/coverage.
- G2: evaluator make accuracy against the requested make; TV distance of make prevalence.
- G3: per make, TV and add-one-smoothed KL between the evaluator's fine-label histogram of
  samples and the fine histogram of test.
- G4: rare = bottom quartile of fine classes by train count. % of fine classes with >= 5
  samples predicted with calibrated confidence >= tau (tau chosen on val); per-class coverage
  in CLIP space, split into rare and common.
- G5: per fine class g: KID and coverage between samples predicted as g and test_g, also
  expressed as a ratio to the real ceiling KID(train_g subsample, test_g) at the same n.
- G6 (only when group ids are given): per make, KID cost matrix groups x real subclasses,
  Hungarian matching, matched cost, per-group purity and entropy of the evaluator histogram,
  number of unmatched real subclasses.
- G7: SSCD nearest-train cosine similarity per sample; histogram; % > 0.5; compared to the
  null distribution of test->train similarities; save NN grids for the top 64.

Use case B (inputs: per-episode generated images; N = 16 per episode):
- B1: hit rate (evaluator predicts the support's fine class).
- B2: KID and coverage vs T, and vs S.
- B3: mean max-SSCD similarity to S vs to T, relative to the real reference (T vs S).
- B4: mean pairwise CLIP distance among generated vs among T.
- B5: mixed episodes: |predicted proportion of subclass A - r|.
- B6: all of the above as functions of k.

Small-n caution: many test classes have about 40 images. Use KID (unbiased) and coverage,
not FID, at the subclass level, and always report n.

# Milestone 5: controls (metric validation)
Implement as "generators" that write images + CSV in the same format a real method will:
C1 copy-train; C1b copy-support (augmented S) and retrieve-NN (nearest train images to S
in DINOv2 space); C2 oracle split (test_A scored against test_B); C3 coarse-only (right
make, subclass uniform within make, drawn from train); C4 prototype collapse (medoid of each
subclass repeated); C5 majority-subclass-only; C6 degraded (Gaussian blur sigma 2 + JPEG q20
of train images). Run the full panel on all controls and produce
reports/metric_validation.csv plus a heat-map: rows = controls, columns = metrics, cells
coloured by whether the metric moved in the expected direction relative to C2.
If a control does NOT fail a metric it should fail, stop and report; do not tune it away.

# Milestone 6: first results
- eval_representation.py for all frozen encoders at 128 and 256 px -> reports/repr_frozen.csv
  and a per-make table. Include the fine-classifier penultimate features as a reference row
  labelled "supervised oracle features (not a baseline)".
- make_report.py: a single Markdown/HTML report with the data summary, metric validation
  heat-map, frozen-encoder table, real-vs-real floor, and per-make breakdown highlighting the
  10 worst makes on R2/R3.
- README: exact commands from a clean machine to the report; expected runtime per stage.

# Definition of done
`make test` passes; `make prepare features controls report` runs end-to-end on one GPU;
docs/METRICS.md defines every number in the report; the controls table shows every metric
failing where expected; nothing outside lsgen/eval reads fine labels.
Report back with: the K_c histogram, the duplicate count, the evaluator classifier accuracy,
the controls heat-map, the frozen-encoder table, and a list of anything you were unsure about.
```

---

## Part C — Follow-up prompts (use after Step 1 is verified)

### Step 2: representation baselines

```text
Using the Step 1 repo and evaluation library unchanged, add representation baselines under
lsgen/methods/repr/: SupCon (coarse labels), SupCon+SimCLR, MaskCon (port from
github.com/MrChenFeng/MaskCon_CVPR2023), FALCON (github.com/mlbio-epfl/falcon), TFB (AAAI
2025; find official code, else implement from the paper and mark it "reimplemented"),
SCGM (only if code exists), and fine-label SupCon as an oracle.
Two regimes: (a) native: each method's published backbone/recipe; (b) controlled: all on
the same DINOv3 ViT-B backbone with an identical fine-tuning recipe (last-N-blocks,
same epochs/augmentation/batch, tuned only on val with a fixed, equal budget per method).
Before running on Cars, reproduce one published number per ported method on the paper's
own dataset and report the gap. 3 seeds. Output reports/repr_baselines.csv with the full
R1-R5 panel, CIs, compute (GPU-hours), and whether a method needs K.
Stop and report which objectives each method satisfies per make, versus frozen DINOv3.
```

### Step 3: generative baselines

```text
Add generative baselines under lsgen/methods/gen/, all writing outputs in the Step 1 format
and evaluated only through lsgen/eval.
Preprocessing: reproduce each published number with the method's NATIVE pipeline
(lsgen/data/native.py), then retrain and evaluate on our bbox15 PNGs for all comparison
tables. Report both, clearly labelled.
Tier 2 (as published, labelled "no coarse labels"): C3-GAN (github.com/naver-ai/c3-gan):
first evaluate the official Stanford Cars checkpoint and compare ACC/NMI/FID with the paper,
then retrain on our train split. FineGAN (github.com/kkanshul/finegan; port to current
PyTorch). TreeDiffusion (github.com/JoGo175/TreeDiffusion).
Tier 3 (controlled): one latent-diffusion backbone (SD-VAE latents at 128 px, DiT-B/2,
fixed schedule, CFG) trained with identical budgets under these conditionings:
coarse-only; coarse + cluster id from frozen-encoder per-make clustering with K-hat;
coarse + cluster id from the MaskCon and FALCON encoders; representation-conditioned
(multi-token cross-attention on support embeddings for use case B; per-cluster embedding
resampling for use case A); fine-label conditional (oracle). Use case A sampling:
prevalence-matching by default, plus a balanced variant labelled as such.
Tier 4 (best-available, separate table): DataDream-style LoRA on Stable Diffusion for use
case B, with external pretraining disclosed.
3 seeds each. Report G1-G7 and B1-B6 with CIs and GPU-hours.
```

### Step 4: diagnosis

```text
Build the objective x method failure matrix (O1/O2/O3 x use cases A/B) from Steps 2-3.
For the best baseline, localise each failure: encoder geometry (R-metrics on real data)
vs conditioning vs generator, using the oracle rows as upper bounds, and test whether
real-data geometry predicts per-make generation fidelity (rank correlation across makes).
Write docs/DIAGNOSIS.md: measured limitations only, each with its evidence.
```

---

## Part D — Directions and pitfalls

1. **Get the benchmark accepted by yourself before any method.** The controls heat-map and the real-vs-real floor are what make every later table credible.
2. **Keep supervision tiers visibly separate** in every table: no labels (C3-GAN, FineGAN, TreeDiffusion), coarse labels (your setting), uses K, and fine-label oracle.
3. **Subclass-level statistics are small-n.** About 40 test images per class means KID/coverage with CIs; never per-class FID.
4. **Anomalous wins are suspects.** If any generator beats the real ceiling on G5 or B2, check G7/B3 before reporting (this is the VW/Volvo lesson from the previous repo).
5. **Where novelty can come from** (only after Step 4): the statement's own candidates are rare-subclass coverage and within-subclass fidelity at equal realism and exemplar dependence. Use case B with k > 1 (selecting a learned distribution from a multi-image support set without retraining) is the least occupied cell in Table 1.
6. **Medical transfer later:** freeze the protocol on Cars, then port the same metrics to HAM10000/PAPILA with patient-disjoint splits.