# ksub: known-K subclass baseline

Method: `lsgen/methods/repr/ksub/`, `scripts/train_ksub.py`, `configs/ksub_{cifar100,maskcon_new}.yaml`. **Uses K**: the number of subclasses K_c
per coarse class is given (`lsgen.data.labels.coarse_classes`); fine labels are otherwise used only for evaluation. Each dataset's conflicting images
(CIFAR-100: `data/cifar100/splits/exclude.txt`, 90 images; Stanford Cars, `maskcon_new` scheme: `data/splits/exclude_maskcon_new.txt`, 211 images) are left out of every run.
No tuning: the defaults below were fixed before any run; `val` is only monitored (coarse kNN), `test` is scored once.

| | CIFAR-100 | Stanford Cars |
|---|---|---|
| coarse / subclass | 20 superclasses / 100 classes, K_c = 5 | `maskcon_new`: 8 body types / 196 official classes, K_c = 6-50 |
| batch (P classes x m images) | 20 x 16 | 8 x 32 |
| features | frozen DINOv3-L, native 32 px | frozen DINOv3-L, bbox15_256 |

## Method
- **Head**: geosub's residual head z = norm(f + g(f)) (imported from `lsgen/methods/geosub/model.py`), last layer zero-initialised so z = f
  before training; f = cached, L2-normalised frozen features of the clean images (no augmented views).
- **Prototypes**: K_c per class, k-means (n_init 10, seed = run seed) on the frozen train features of the class; not learnable. After each step
  each prototype moves by EMA (momentum 0.99) towards the r-weighted mean of the batch embeddings, then is re-normalised.
- **Soft assignment**: r_i = softmax_g cos(z_i, mu_{c_i,g}) / tau_a over the prototypes of the image's own class; recomputed every step, no gradient.
- **Loss** L = L_coarse + lambda_sub L_sub + lambda_var L_var:
  - L_coarse: SupCon on the coarse labels (`supcon_loss`, temperature 0.1);
  - L_sub: for anchor i, softmax of z_i . z_j / tau over the other images j of its class, cross-entropy against w_ij / sum_j w_ij with
    w_ij = r_i . r_j; other classes are not in this term;
  - L_var: per (class, prototype) with sum_i r_ig >= 2 in the batch, relu(1 - tr Cov_r(z) / tr Cov_r(f)), averaged.
- Defaults: tau 0.1, tau_a 0.05, lambda_sub = lambda_var = 1; AdamW (lr 1e-3, wd 0.05), 5 warm-up epochs then cosine, 100 epochs, g width 2048
  (geosub's). An epoch is N / (P m) class-balanced batches.

| run | terms |
|---|---|
| `frozen` | none: k-means with K_c on f (z = f) |
| `coarse` | L_coarse |
| `sub` | L_coarse + L_sub |
| `full` | L_coarse + L_sub + L_var |

## Evaluation
R1-R5 on test embeddings z (`scripts/eval_representation.py`; `frozen` = the backbone's own row), and the Hungarian ACC of the model's own final
assignment argmax_g r_ig per class with K_c >= 2 ("ACC own"; `frozen`: nearest normalised k-means centroid by cosine). `scripts/summarize_ksub.py` writes
`<reports_root>/ksub/summary.md` (run x {kNN coarse, ACC oracle K, NMI, d0/d1, d1/d2, var ratio, ARI boot, ACC own}, mean ± std over seeds 0-2,
and the per-class table) with provenance.

## Commands (per dataset; `maskcon_new` instead of `cifar100` for Stanford Cars)
```
CONFIG=configs/ksub_cifar100.yaml sbatch slurm/train_ksub.sh                  # GPU: k-means + 3 trained methods x 3 seeds
CONFIG=configs/eval_repr_ksub_cifar100.yaml sbatch slurm/eval_representation.sh  # CPU: R1-R5, ~25 min per encoder (10 encoders)
python scripts/summarize_ksub.py --config configs/ksub_cifar100.yaml          # CPU, seconds: reports/cifar100/ksub/summary.md
```

## Assumptions
- No augmentation: f are the clean cached features, on both datasets (CIFAR-100 has no cached views; the `maskcon_new` DINOv3-L views
  used by geosub exist but are not used, to keep the two benchmarks alike).
- K_c = number of hidden subclasses per coarse class over all splits: `hierarchy.csv` for `make` / `cifar100`, the body-type mapping
  (`data/maskcon_coarse.csv`) for `maskcon_new`.
- Stanford Cars batches are 8 x 32 (every body type, batch 256) rather than 16 x 16: there are only 8 coarse classes.
- (`make` scheme only) For K_c = 1 makes the single prototype gets r = 1: L_sub is then plain within-class SupCon and L_var a class-level floor.

## Results (test, DINOv3-L, seeds 0-2; `reports/ksub/summary.md`, `reports/cifar100/ksub/summary.md`)
Stanford Cars here is the earlier `make` scheme run (`configs/ksub_cars.yaml`, superseded by `maskcon_new`, below).
Conflicting images excluded: Stanford Cars 154 (`make` list), CIFAR-100 90. Mean ± std over seeds; full tables, per-class results and provenance in the summaries.

| | Cars ACC oracle K | Cars ACC own | Cars d1/d2 | CIFAR ACC oracle K | CIFAR ACC own | CIFAR d1/d2 | kNN coarse (Cars / CIFAR) |
|---|---|---|---|---|---|---|---|
| frozen | **0.782** | **0.787 ± 0.003** | 0.709 | **0.892** | **0.902** | 0.921 | 0.973 / 0.964 |
| coarse | 0.410 ± 0.008 | 0.549 ± 0.010 | 0.057 | 0.224 | 0.229 | 0.063 | 0.988 / 0.970 |
| sub | 0.411 ± 0.002 | 0.581 ± 0.012 | 0.052 | 0.224 | 0.230 | 0.062 | 0.988 / 0.970 |
| full | 0.543 ± 0.011 | 0.460 ± 0.007 | 0.489 | 0.269 ± 0.005 | 0.233 | 0.556 | 0.971 / 0.971 |

**Every trained run destroys the subclass structure of the frozen features.** On CIFAR-100 it falls to chance (Hungarian ACC with 5 balanced
classes ~0.2).
- L_coarse collapses each class: SupCon and L_sub both end at log 15 = 2.708 (each anchor has 15 same-class candidates in the batch), the value
  for all same-class images at one point. d1/d2 drops from 0.71 / 0.92 to 0.06, and var ratio (within-subclass / within-class spread) goes to 0.93 / 0.98.
- L_sub does not resist this. r is computed from the collapsing z, so all images of a class get the same r, w_ij becomes uniform and the
  collapsed state is optimal for L_sub too. On CIFAR it starts at 2.21 and rises to 2.708.
- L_var restores spread (d1/d2 0.49 / 0.56) but not subclass structure: var ratio 0.95 / 0.99, so the restored spread is not aligned with
  the subclasses. ACC oracle K recovers only partly on Cars and stays at chance on CIFAR.
- Seed variation is small (std <= 0.012), so these results do not depend on the seed.

## Without L_coarse (test, DINOv3-L, seeds 0-2; `reports/ksub_maskcon_new/summary.md`, `reports/cifar100/ksub/summary.md`)
`sub_only` = L_sub; `sub_var` = L_sub + L_var. Stanford Cars on `maskcon_new` (211 conflicting images excluded), CIFAR-100 (90 excluded).

| | Cars ACC oracle K | Cars ACC own | Cars kNN coarse | CIFAR ACC oracle K | CIFAR ACC own | CIFAR ARI boot | CIFAR kNN coarse |
|---|---|---|---|---|---|---|---|
| frozen | **0.800** | **0.792 ± 0.002** | **0.944** | 0.892 | 0.902 | 0.891 | **0.964** |
| sub_only | 0.605 ± 0.014 | 0.528 ± 0.047 | 0.935 ± 0.006 | 0.575 ± 0.117 | 0.249 ± 0.017 | 0.738 ± 0.064 | 0.865 ± 0.019 |
| sub_var | 0.653 ± 0.004 | 0.610 ± 0.013 | 0.908 ± 0.009 | **0.917 ± 0.002** | **0.917 ± 0.002** | **0.994 ± 0.001** | 0.960 ± 0.001 |

- **CIFAR-100, `sub_var`: the first run above frozen.** ACC oracle K +0.025 and ACC own +0.015, with std 0.002 over seeds. The
  clustering is far more stable (ARI boot 0.994 vs 0.891), and coarse kNN is about level (-0.004). d1/d2 = 1.07: superclasses are no longer
  further apart than their subclasses, which is expected with no between-class term.
- **CIFAR-100, `sub_only`: still unstable.** L_sub drifts towards its collapsed value (training log), ACC own is at chance and the spread across
  seeds is large: L_var is needed.
- **Stanford Cars `maskcon_new`: both runs below frozen** (ACC oracle K -0.15 / -0.19, ACC own -0.18 / -0.26). The loss is in the large
  body types (Sedan K_c 50: 0.73 -> 0.45; SUV 36: 0.82 -> 0.48). The small ones are level or better (Wagon 6, Hatchback 14, Van 13:
  0.70 vs 0.68). Prototypes do not die (`sub_var` uses 49/50 on Sedan), so the clusters exist but do not follow the subclasses. Likely cause
  (not tested): with 32 images per body type in a batch and up to 50 subclasses, most subclasses have 0-1 images per batch, so L_sub
  rarely sees a same-subclass pair. CIFAR-100 has about 3 per subclass (16 images, K_c = 5).

## Weak L_coarse (`sub_var_coarse01` = 0.1 L_coarse + L_sub + L_var; weight fixed in advance, geosub's val choice, not tuned)
Full comparison tables with every representation method: `reports/cifar100/summary.md`, `reports/repr_maskcon_new/summary.md` (ksub section).

| | ACC oracle K | ACC own | ACC K-hat | probe coarse | kNN coarse | d1/d2 |
|---|---|---|---|---|---|---|
| CIFAR-100 frozen DINOv3-L | 0.892 | 0.902 | 0.778 | 0.966 | 0.964 | 0.921 |
| CIFAR-100 `sub_var` | 0.917 ± 0.002 | 0.917 ± 0.002 | 0.917 | 0.338 | 0.960 | 1.074 |
| CIFAR-100 `sub_var_coarse01` | **0.919 ± 0.001** | **0.918 ± 0.001** | **0.918** | 0.951 | **0.968** | 0.998 |
| Cars frozen DINOv3-L | **0.800** | **0.792** | **0.660** | 0.973 | 0.944 | 0.877 |
| Cars `sub_var` | 0.653 ± 0.004 | 0.610 ± 0.013 | 0.580 | 0.445 | 0.908 | 1.029 |
| Cars `sub_var_coarse01` | 0.652 ± 0.011 | 0.588 ± 0.006 | 0.548 | **0.979** | **0.973** | 0.851 |

- **The weak L_coarse restores the coarse structure without collapse,** on both datasets: L_coarse ends at 5.56 / 5.82, far from the collapsed 2.71 / 3.43.
- **CIFAR-100: `sub_var_coarse01` is the best row of the table.** It keeps the subclass gain of `sub_var` and recovers the linear coarse probe (0.338 -> 0.951).
- **Stanford Cars: the subclasses are still worse than frozen.** Body types with K_c <= 14 are level with frozen (Hatchback, Wagon, Van); those
  with K_c 18-50 lose 0.12-0.31 ACC oracle K (Sedan 0.73 -> 0.52, SUV 0.82 -> 0.53). That fits the batch-coverage explanation above.

## Batch size (4x batch, same number of optimiser steps; `configs/ksub_{maskcon_new_b1024,cifar100_b1280}.yaml`)
Diagnostic of the batch-coverage explanation, compared on test; the batch size still has to be chosen on val. Full tables with every method:
`reports/repr_maskcon_new/summary.md` (ksub section) and `reports/cifar100/summary.md`.

| | ACC oracle K | ACC own | ACC K-hat | ARI boot | kNN coarse | probe coarse |
|---|---|---|---|---|---|---|
| Cars frozen DINOv3-L | **0.800** | 0.792 | 0.660 | 0.874 | 0.944 | 0.973 |
| Cars `sub_var`, batch 256 / 1024 | 0.653 / 0.796 | 0.610 / **0.795** | 0.580 / **0.743** | 0.805 / **0.973** | 0.908 / 0.951 | 0.445 / 0.498 |
| Cars `sub_var_coarse01`, batch 256 / 1024 | 0.652 / 0.788 | 0.588 / 0.790 | 0.548 / 0.734 | 0.741 / 0.953 | 0.973 / **0.985** | **0.979** / 0.925 |
| CIFAR frozen DINOv3-L | 0.892 | 0.902 | 0.778 | 0.891 | 0.964 | **0.966** |
| CIFAR `sub_var`, batch 320 / 1280 | 0.917 / 0.912 | 0.917 / 0.912 | 0.917 / 0.912 | **0.994** / 0.992 | 0.960 / 0.954 | 0.338 / 0.478 |
| CIFAR `sub_var_coarse01`, batch 320 / 1280 | **0.919** / 0.912 | **0.918** / 0.911 | **0.918** / 0.911 | 0.990 / 0.988 | **0.968** / 0.967 | 0.951 / 0.909 |

- **The batch-coverage explanation holds.** On Stanford Cars, batch 1024 brings ACC oracle K from 0.65 to 0.79-0.80, level with frozen DINOv3-L.
  The gain is in the body types with many subclasses (Sedan 0.45 -> 0.74, SUV 0.48 -> 0.86).
- **Label-free clustering improves on Cars:** ACC K-hat 0.743 and ARI boot 0.973 are the best of the maskcon_new table.
- **On CIFAR-100 (about 3 images per subclass already at batch 320), a larger batch costs a little:** -0.005 to -0.007 ACC.
- **What sets the batch:** what matters is enough images per subclass in a batch, which depends on K_c. A pseudo-subclass-stratified sampler
  would give that at a fixed batch size.
