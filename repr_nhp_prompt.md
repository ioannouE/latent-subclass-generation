# Nuisance-factored hierarchical prototypes (NHP)


## Role and goal

You are a careful research engineer working in my existing benchmark repo for **subclass-preserving image generation on Stanford Cars** (package `lsgen`, configs in `configs/`, docs in `docs/`, reports in `reports/`). Read `docs/DATA.md`, `docs/METRICS.md`, `configs/eval_representation.yaml`, `configs/repr_baselines.yaml`, `reports/summary_repr.md` and `reports/report.md` first, plus the existing code in `lsgen/methods/repr/` (SupCon, SimCLR, MaskCon, FALCON) and the R1-R5 evaluation code. Match their conventions (config style, seeding, logging, output folder layout, CI computation). Do not rewrite or modify the evaluation code or the metric definitions.

Your task has three parts:

1. **Implement** a new representation-learning method, "nuisance-factored hierarchical prototypes" (NHP), described below, as a new module next to the existing baselines.
2. **Unit-test** it on a small synthetic dataset that runs on CPU in minutes.
3. **Write an experiment plan and the runnable configs and scripts** for the real Stanford Cars run, including ablations and a fair comparison against the existing baselines. I will run the real experiments on my own GPU machine. You may not have a GPU or dataset access, so do not claim any result on real data. If something cannot be run in your environment, say so and prepare it to run on mine.

## Background (what the existing results say)

Setup: 49 makes, 189 hidden make-model subclasses (years merged), 19 makes with a single subclass (K_c = 1), 30 with K_c >= 2. After excluding 154 conflicting images the splits are train 7,270 / val 807 / test 7,954. Fine labels exist **in evaluation only**.

Frozen and fine-tuned baselines (DINOv3-B, 256 px, last 4 blocks, 50 epochs, one untuned recipe, make labels only, one seed):

| Method | kNN make | ACC oracle K | ACC K-hat | d0/d1 | d1/d2 | var ratio | K-hat err (K_c=1) |
|---|---|---|---|---|---|---|---|
| frozen DINOv3-B | 0.949 | 0.717 [0.696, 0.746] | 0.619 | 0.630 | 0.744 | 0.690 | 1.16 |
| SupCon | 0.968 | 0.718 | 0.634 | 0.645 | 0.719 | 0.703 | 1.26 |
| SimCLR | 0.813 | 0.583 | 0.478 | 0.778 | 0.847 | 0.821 | 2.68 |
| MaskCon | 0.953 | 0.705 | 0.582 | 0.648 | 0.744 | 0.707 | 1.47 |
| FALCON | 0.967 | 0.698 | 0.613 | 0.700 | 0.710 | 0.754 | 1.47 |

Reference points: frozen DINOv3-L (256 px) has ACC oracle K 0.782 and ACC K-hat 0.698; the supervised fine-label oracle has 0.928. The within-make fine probe is about 0.97 for DINOv3, so subclass information is present but clustering does not recover it. var_ratio of 0.64-0.70 says most within-make variance is **not** the subclass. K-hat never returns 1 on single-subclass makes for DINOv3-based embeddings.

Hypothesis behind NHP: nuisance factors (viewpoint, colour, background) dominate the within-make geometry, and none of the tested losses says which factors to ignore. This is a hypothesis to test, not a fact.

## The method

Backbone: DINOv3-B, 256 px crops, same preprocessing and the same trainable part as the baselines (last 4 blocks), so the comparison is fair. A small head on the backbone output produces three codes:

- **z_make**: trained with SupCon on make labels (reuse the existing SupCon implementation). Serves R1.
- **z_sub**: a per-make mixture of learnable prototypes.
  - Each make m has up to P prototypes (make P a config value; default the largest K_c in the data plus margin, so 24). Soft assignment of an image to a prototype of its own make: temperature-scaled cosine softmax over that make's prototypes.
  - Assignment loss: DINO/SwAV-style self-distillation between two augmented views (EMA teacher, sharpened teacher targets, cross-entropy with the student). Do **not** impose an equipartition constraint, because subclass prevalence is part of what the benchmark measures (O2). A mild anti-collapse term is fine, but document it.
  - Nuisance invariance: the positive views use colour permutation and strong colour jitter, horizontal flip, random crop and background swap or mask. The invariance comes from assignment consistency across these views. Optionally add a gradient-reversal adversary or an HSIC penalty that removes information about the augmentation parameters (colour, pose proxies) from z_sub. Implement it behind a flag, off by default, so it can be ablated.
  - Hierarchy margin: using the soft assignments, compute differentiable surrogates of the three distances in embedding space. Delta0 is a pair-weighted distance between images that share a prototype assignment, Delta1 between same-make images with different assignments, Delta2 between different-make images. Add hinge terms Delta0 + m1 < Delta1 and Delta1 + m2 < Delta2. No fine labels are used. Pair weights come from assignment agreement.
  - Sparse prototype use: per-make gates on prototypes (hard-concrete / L0 relaxation, or a stick-breaking Dirichlet-process prior on usage) with a penalty, so a make with one real subclass can end up with one active prototype. Behind a flag, so it can be ablated.
- **z_nuis**: a code for within-subclass variation. A VICReg-style variance term (and a covariance term) keeps it from collapsing, so subclasses do not collapse to a point. It is excluded from the embedding used for subclass metrics.

Training signal summary: make labels and augmentations only. **No fine (make-model) labels, no evaluator classifiers (ConvNeXt-T), and no test data may be used in training, model selection or tuning.**

Evaluation embedding (declare it before running, in the config, and do not change it afterwards): primary = `z_sub` L2-normalised; secondary = `z_make ⊕ z_sub`. Use the same dimensionality as the baseline embedding where possible, and report the dimension. Run the existing R1-R5 pipeline unchanged on that embedding, including K-hat with the fixed rule (SIL_MIN = 0.10; do not tune it).

## Part 1: implementation

- New module under `lsgen/methods/repr/` (for example `nhp.py`) with: the head, the three losses as separate, individually testable functions, the training loop (reuse the baselines' data loading, augmentation plumbing, optimiser setup, EMA, checkpointing and logging), and the export of embeddings in the format the evaluation code expects.
- A config `configs/repr_nhp.yaml` with every hyperparameter, loss weight and flag (including each ablation switch), the pre-declared evaluation embedding, and the seed.
- Training-time monitoring that uses only allowed information: loss terms, number of active prototypes per make, assignment entropy, make kNN on val. Do not log or select on fine-label metrics. If you think fine labels on val would be useful for model selection, do not add it; list it as a question for me, because the baselines would need the same treatment.
- Keep the git tree clean and commit in small steps with clear messages.

## Part 2: unit tests on synthetic data (CPU, minutes)

Build `tests/test_nhp_synthetic.py` and a small synthetic generator that **replaces the backbone** with an identity or a small MLP on pre-computed features. Planted structure: several makes; each make with a known number of subclasses (including single-subclass makes); a subclass-specific signal; and independent nuisance factors (colour-like and pose-like) that shift features strongly and are shared across subclasses. Tests to write and run:

1. Shapes, dtype, determinism under a fixed seed, and finite gradients for every loss term separately and together.
2. Each loss term decreases when optimised alone on the toy data.
3. With invariance on, a linear probe from z_sub to the planted nuisance factors is near chance, while without invariance it is clearly above chance. The probe from z_sub to the planted subclass must stay high.
4. With the sparsity flag on, single-subclass makes end up with one active prototype and multi-subclass makes keep about the planted number. Report K-hat error on the toy data with and without it.
5. Hierarchy: after training, Delta0 < Delta1 < Delta2 holds on the toy data, and removing the margin term makes it worse.
6. Recovery: per-make k-means ACC and ARI against the planted subclasses beat the same pipeline on the raw toy features, and beat a make-only SupCon head.
7. Negative controls: shuffle subclass signal so it carries no structure and check the method does not "find" confident subclasses (assignment entropy stays high or prototypes stay inactive).

Report honestly if a test fails or is flaky, and fix the code, not the test. Passing toy tests show the code is correct, not that the idea works on cars. Say so in the summary.

## Part 3: experiment plan and runnable setup for real data

Create `docs/EXPERIMENT_NHP.md`, plus configs and scripts under `configs/` and `scripts/`, containing:

1. **Gate 0, diagnostic (before any NHP training).** A script that, per make, regresses the top within-make principal components of frozen DINOv3-B embeddings on cheap nuisance proxies computed from the images without fine labels: mean colour / HSV histogram of the car region, left-right orientation via flip-consistency, aspect ratio, background brightness. Report variance explained per make and macro-averaged, with CIs. State in advance the decision rule: if nuisance proxies explain little of the top within-make PCs, the premise is weak and NHP is not expected to help; stop and tell me.
2. **Fair baselines.** Re-run SupCon, MaskCon, FALCON and frozen DINOv3-B with the same hyperparameter-search budget and the same selection criterion as NHP (label-free or coarse-label criteria on val only), at least 3 seeds each. Existing untuned numbers must not be reused as the comparison.
3. **NHP runs:** the full method with at least 3 seeds, plus single-component ablations (no invariance, no hierarchy margin, no sparsity prior, no z_nuis variance term), each with 3 seeds. Include one ablation using z_make ⊕ z_sub as the evaluation embedding.
4. **Statistics.** Report macro means over makes. For method-versus-method claims use a **paired bootstrap over makes** (resample makes, same makes for both methods), and also report per-make win rates. State the minimum effect that counts, for example oracle-K ACC gain over frozen DINOv3-B above 0.03 with a paired CI excluding zero. Do not rank methods from single-seed differences of a few hundredths.
5. **Success criteria written before the run:** primary = oracle-K ACC, K-hat ACC and Delta0/Delta1 improve over frozen DINOv3-B and over the best tuned baseline; no regression in make kNN beyond 0.01; K-hat error on K_c = 1 makes does not worsen. Report whether DINOv3-L frozen (ACC 0.782) is also beaten as a secondary result. Also report the supervised fine-label oracle (0.928) as the ceiling.
6. **Metric-gaming checks:** compare at equal embedding dimension; report effective rank; confirm results are not driven by a few makes (per-make table, worst-10 makes for R2 and R3 as in `reports/report.md`); confirm K-hat is not rewarded just because the embedding lost structure (check the fine probe and Recall@1 do not drop).
7. **Open-risk list** for me to read: identifiability from coarse labels alone, background-swap augmentation quality, whether K_c = 1 makes truly lack viewpoint or colour structure (untested), evaluator circularity if NHP is later used to condition a generator.
8. Compute estimate per run and a run order that gets a first real signal cheaply, for example 8-10 makes, 1 seed, 10 epochs, then scale up.

## Working rules

- Stop and ask me before changing any existing evaluation, metric or data code, or before using any fine label in training.
- Never invent or paste results. Every number you report must come from a command you actually ran, and you should state which command.
- Put assumptions you had to make in `docs/EXPERIMENT_NHP.md` under "Assumptions", and list every question for me at the end of your final message.
- if a job requires gpu, give me instruction of what I need to run, I will run it myself for testing and report to you. 
- if you create something that needs training, create the corresponding sh script for me to submit as job in slurm
- base everything according to what is already implemented in the repository. Don't develop a seprate style but be cohesive and congruent with the code and everything that already exists.

## Final message format

A short summary: what was implemented, which tests passed or failed, what is ready to run on my GPU machine and in what order, the decision rule for Gate 0, and the open questions. No claims about performance on Stanford Cars.