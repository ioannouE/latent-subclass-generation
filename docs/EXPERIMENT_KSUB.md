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

## Prototype staleness (train ARI per epoch; `configs/ksub_{maskcon_new,maskcon_new_b1024,cifar100}_ari.yaml`, code `c1a11d1`)
Diagnostic of whether the EMA prototypes keep up with z, i.e. whether re-running k-means each epoch would change anything. Training is
identical to `configs/ksub_{maskcon_new,cifar100}.yaml` (batch 256 / 320), with `sub_var` and `sub_var_coarse01`, seed 0 only and their own run tags (`_ari`).
At the end of every epoch (`ari_every: 1`, `assignment_ari`), on train, per class with K_c >= 2 and then averaged over classes, the
current assignment argmax_g r_ig (z to the EMA prototypes) is compared with (a) a fresh k-means on z (`init_prototypes`, n_init 10, run
seed) and (b) the initial k-means on f. This is monitoring only: nothing feeds back into training. Conflicting images excluded: Stanford Cars 211
(`maskcon_new` list), CIFAR-100 90. Slurm jobs 412808 / 412810; per-epoch values in `reports/ksub_ari/ari_per_epoch.csv`.

| epoch | Cars `sub_var` fresh / initial | Cars `sub_var_coarse01` fresh / initial | CIFAR `sub_var` fresh / initial | CIFAR `sub_var_coarse01` fresh / initial |
|---|---|---|---|---|
| 1 | 0.89 / 0.99 | 0.91 / 0.97 | 0.95 / 0.93 | 0.95 / 0.94 |
| 10 | 0.89 / 0.90 | 0.90 / 0.91 | 0.999 / 0.88 | 0.99 / 0.88 |
| 20 | 0.81 / 0.85 | 0.79 / 0.84 | 1.000 / 0.88 | 0.99 / 0.88 |
| 50 | 0.69 / 0.70 | 0.67 / 0.69 | 1.000 / 0.88 | 0.99 / 0.88 |
| 100 | 0.72 / 0.66 | 0.66 / 0.64 | 1.000 / 0.88 | 0.99 / 0.88 |

- **CIFAR-100: the EMA prototypes track z.** Agreement with a fresh k-means is above 0.99 from epoch 5 on (`sub_var`: exactly 1 from epoch 21).
  The assignments move away from the frozen k-means in the first ~10 epochs (ARI 0.88), then stay fixed. Re-clustering would change nothing.
- **Stanford Cars, batch 256: they do not.** Agreement with a fresh k-means falls from about 0.9 to 0.65-0.72 by epoch 50 and does not
  settle. About 0.9 is k-means' own noise floor here: at epoch 1 z is still almost f, and the frozen ARI boot is 0.874.
- **The existing test reports locate the gap.** "ACC oracle K" is a fresh k-means on z, "ACC own" the EMA assignment. At batch 256
  (`reports/ksub_maskcon_new/per_class.csv`) the gap oracle - own is in the body types with many subclasses: Sedan +0.14 / +0.26, SUV
  +0.09 / +0.16, Coupe +0.07 / +0.05, Cab +0.06 / +0.06 (`sub_var` / `sub_var_coarse01`); about 0 for K_c <= 14. At batch 1024
  (`reports/ksub_maskcon_new_b1024/per_class.csv`) it is at most 0.02 in every body type except Wagon (`sub_var_coarse01` -0.06). This is
  test-split analysis of runs that already existed, not a model choice.
- **The stale readout is only a small part of the Cars loss.** At batch 256 even a fresh k-means on z gives Sedan 0.45 against frozen
  0.73, so most of the damage is in z itself. Likely cause (not tested): the stale prototypes give the wrong r, and L_sub / L_var then train z
  towards the wrong groups. Re-clustering during training tests this (next section).
- **Stanford Cars, batch 1024 (`configs/ksub_maskcon_new_b1024_ari.yaml`, job 421836): they do.** Logged every 4 epochs (`ari_every: 4`)
  = every 28 steps, the same step grid as batch 256. Agreement with a fresh k-means rises from 0.90 / 0.89 (first log) to 0.95 / 0.93
  by step 280 and stays there to the end (0.957 / 0.928 at step 2,800, `sub_var` / `sub_var_coarse01`); agreement with the initial
  k-means stays at 0.93 / 0.92: at batch 1024 the clusters barely move from the frozen ones. Staleness is a batch-256 problem.

## Re-clustering every epoch (`configs/ksub_{maskcon_new,cifar100}_recluster.yaml`; test, DINOv3-L, seeds 0-2)
Variant (`recluster_every: 1`): at the end of every epoch the prototypes are reset to a fresh k-means on z (`init_prototypes`, n_init 10,
run seed); EMA in between, as before. Otherwise identical to `configs/ksub_{maskcon_new,cifar100}.yaml` (batch 256 / 320). Fixed in advance,
not tuned. The last reset follows the last epoch, so ACC own is the assignment of a k-means on train z. R1-R5:
`configs/eval_repr_ksub_{maskcon_new,cifar100}_recluster.yaml`; summaries `reports/ksub_maskcon_new_recluster/summary.md`,
`reports/cifar100/ksub_recluster/summary.md` (per class: `per_class.csv`). Slurm jobs 421857 (Cars), 421859 / 421860 (CIFAR-100).
Conflicting images excluded: Stanford Cars 211 (`maskcon_new` list), CIFAR-100 90. Run from uncommitted code on top of `734259f`
(summaries record `734259f-dirty`); committed unchanged afterwards.

| | ACC oracle K | ACC own | ARI boot | kNN coarse |
|---|---|---|---|---|
| Cars frozen DINOv3-L | 0.800 | 0.792 | 0.874 | 0.944 |
| Cars `sub_var`: batch 256 EMA / batch 1024 EMA / batch 256 re-cluster | 0.653 / 0.796 / **0.810 ± 0.007** | 0.610 / 0.795 / **0.810** | 0.805 / **0.973** / 0.941 | 0.908 / 0.951 / 0.929 |
| Cars `sub_var_coarse01`: same | 0.652 / 0.788 / **0.810 ± 0.008** | 0.588 / 0.790 / **0.811** | 0.741 / 0.953 / 0.927 | 0.973 / **0.985** / 0.978 |
| CIFAR frozen DINOv3-L | 0.892 | 0.902 | 0.891 | 0.964 |
| CIFAR `sub_var`: batch 320 EMA / batch 1280 EMA / batch 320 re-cluster | **0.917** / 0.912 / 0.909 ± 0.001 | **0.917** / 0.912 / 0.910 | **0.994** / 0.992 / 0.992 | 0.960 / 0.954 / 0.959 |
| CIFAR `sub_var_coarse01`: same | **0.919** / 0.912 / 0.911 ± 0.000 | **0.918** / 0.911 / 0.911 | 0.990 / 0.988 / 0.991 | **0.968** / 0.967 / **0.968** |

- **Stanford Cars: re-clustering fixes batch 256.** ACC oracle K 0.65 -> 0.81, slightly above batch 1024 (0.79-0.80) and the first ksub rows
  above frozen DINOv3-L (+0.010, about 1.5 std). The gain is in the body types with many subclasses: Sedan 0.45 -> 0.76 (frozen 0.73), SUV
  0.48 -> 0.85 (frozen 0.82), Cab 0.65 -> 0.78 (frozen 0.72) (`sub_var`). Hatchback (0.96) and Convertible (0.79) stay below frozen (0.99 / 0.84),
  as at batch 1024. Clustering is less stable than at batch 1024 (ARI boot 0.94 vs 0.97).
- **So on Cars the batch-256 loss came from stale prototypes**, not from too few same-subclass pairs per batch: with fresh prototypes, batch
  256 does as well as batch 1024.
- **CIFAR-100: re-clustering costs a little** (-0.008 ACC, std <= 0.002) but stays well above frozen. The EMA prototypes already tracked z
  (ARI 1.0, above), so a fresh k-means each epoch only adds k-means noise.
- **No single setting is best on both datasets** (re-cluster on Cars, EMA at the small batch on CIFAR-100). These are test numbers: the
  choice between EMA / re-cluster and the batch size still has to be made on val.

## Unknown K: plug-in K-hat (`configs/ksub_{maskcon_new,cifar100}_khat.yaml`)
First K-unknown baseline: the re-clustering variant above, unchanged, except that the true K_c is never read (`k: khat`). Per seed, a
label-free K-hat per coarse class is estimated once on the frozen train features (`estimate_k`: R2's own K-hat estimator, silhouette sweep
over K = 2..k_max with gap-statistic fallback, on per-class PCA-50) and used for the whole run, including every re-clustering. `k_max` = 60 on
both datasets, fixed in advance; it lies above the largest K_c (50, Sedan), and R2's eval K-hat keeps its own cap of 25. `frozen` = k-means with
K-hat on f. This measures how much of the known-K result survives a realistic K error. The K used per run is stored in `repr_models/<run>.pt` and
the feature JSONs; `scripts/summarize_ksub.py` reports it per class with K err = mean |K - K_c|. ACC oracle K (R2) still scores z with
the true K_c (evaluation only); ACC own is the run's own K-hat clustering. Conflicting images excluded as above (Cars 211, CIFAR-100 90).
```
CONFIG=configs/ksub_maskcon_new_khat.yaml METHODS="frozen sub_var" sbatch slurm/train_ksub.sh   # and METHODS=sub_var_coarse01; cifar100 alike
CONFIG=configs/eval_repr_ksub_maskcon_new_khat.yaml sbatch slurm/eval_representation.sh
python scripts/summarize_ksub.py --config configs/ksub_maskcon_new_khat.yaml
```

Results (test; git `1862299-dirty`; summaries `reports/ksub_maskcon_new_khat/summary.md`, `reports/cifar100/ksub_khat/summary.md`, per class
`per_class.csv` there). K-hat depends only on the seed's frozen train features, so every method of a seed uses the same K. Conflicting images
excluded (Cars 211, CIFAR-100 90).

| | ACC oracle K: K-hat run / known-K re-cluster | ACC own: K-hat run / known-K re-cluster | ARI boot: K-hat / known K | kNN coarse | K err |
|---|---|---|---|---|---|
| Cars frozen DINOv3-L | 0.800 / 0.800 | 0.719 / 0.792 | 0.874 / 0.874 | 0.944 | 5.4 |
| Cars `sub_var` | 0.764 / **0.810** | 0.717 / **0.810** | 0.870 / 0.941 | 0.931 | 5.4 |
| Cars `sub_var_coarse01` | 0.767 / **0.810** | 0.721 / **0.811** | 0.861 / 0.927 | 0.978 | 5.4 |
| CIFAR frozen DINOv3-L | 0.892 / 0.892 | 0.720 / 0.902 | 0.891 / 0.891 | 0.964 | 3.05 |
| CIFAR `sub_var` | 0.847 / **0.909** | 0.718 / **0.910** | 0.910 / 0.992 | 0.955 | 3.05 |
| CIFAR `sub_var_coarse01` | 0.850 / **0.911** | 0.725 / **0.911** | 0.912 / 0.991 | 0.965 | 3.05 |

- **With an estimated K the gain over frozen is gone on both datasets.** ACC oracle K (z scored with the true K_c) falls below frozen
  DINOv3-L (Cars -0.03, CIFAR-100 -0.04), and the runs' own K-hat clustering is no better than k-means with K-hat on f (ACC own 0.72 throughout).
- **Underestimating K is what hurts; overestimating is roughly harmless.** Training with too few prototypes merges subclasses in z:
  Cab (K-hat 6.7 vs K_c 18) 0.718 -> 0.526 ACC oracle K (`sub_var_coarse01` vs frozen), Wagon (5 vs 6) 0.846 -> 0.798; CIFAR-100
  aquatic_mammals and trees (K-hat 2 vs 5) 0.830 -> 0.421 and 0.603 -> 0.439, small_mammals (3) 0.844 -> 0.609. Classes with K-hat > K_c
  are level or better (Sedan 47 vs 50: 0.732 -> 0.762; CIFAR insects 12.7: 0.920 -> 0.970, large_man-made_outdoor_things 7: 0.910 -> 0.966).
- On Cab the K-hat clusters follow the car make rather than the subclass (`reports/ksub_maskcon_new_compare/summary.md`).
- So an unknown-K ksub needs a K estimate that errs high, or an assignment that can split clusters during training; K-hat on frozen features is not enough.

## Diffusion-map re-clustering and loss-weight grid (`configs/ksub_maskcon_new_recluster_diff.yaml`, `configs/ksub_{maskcon_new,cifar100}_grid.yaml`)
Both start from the known-K re-clustering variant (batch 256 / 320, reset every epoch), are trained on train and scored on test; settings fixed
in advance. No setting is selected on test: every grid setting is reported (sensitivity table).
- **Diffusion maps** (Stanford Cars only): the initial and re-clustered prototypes are the means of diffusion-map spectral clusters (K_c clusters on
  the class's PCA-50, n_neighbors 15, t 1: R2's diffusion row) instead of k-means centroids. `frozen` = diffusion maps with K_c on f.
- **Grid**: lambda_sub in {1, 2} x lambda_coarse in {0.1, 0.3} x var_floor in {0.25, 0.5, 0.75}, lambda_var = 1, k-means re-clustering.
  `var_floor` is the level of L_var's hinge, relu(var_floor - tr Cov_r(z) / tr Cov_r(f)) per soft subclass: 1 (every earlier run) keeps each
  pseudo-subclass at least as spread as in the frozen features; 0.5 lets it shrink to half its frozen variance. var_floor 1 with lambda_sub 1,
  lambda_coarse 0.1 is `sub_var_coarse01` of `configs/ksub_{maskcon_new,cifar100}_recluster.yaml`.
Conflicting images excluded as above (Cars 211, CIFAR-100 90).

Results (test; git `1862299-dirty`; slurm jobs 432307 / 432308 / 432309 (diffusion), 432293-432305 and 432299 / 432306 (grid); summaries
`reports/ksub_maskcon_new_recluster_diff/summary.md`, `reports/ksub_maskcon_new_grid/summary.md`, `reports/cifar100/ksub_grid/summary.md`).

**Diffusion maps vs k-means (Stanford Cars).**

| | ACC oracle K | NMI | d0/d1 | d1/d2 | var ratio | ARI boot | ACC own |
|---|---|---|---|---|---|---|---|
| frozen DINOv3-L (own: k-means / diffusion) | 0.800 | 0.886 | 0.415 | 0.877 | 0.429 | 0.874 | 0.792 / **0.829** |
| `sub_var`, k-means / diffusion | 0.810 / 0.811 | 0.890 / 0.888 | 0.401 / 0.403 | 1.032 / 1.032 | 0.415 / 0.417 | 0.941 / **0.954** | 0.810 / 0.820 |
| `sub_var_coarse01`, k-means / diffusion | 0.810 / **0.819** | 0.886 / **0.892** | 0.480 / 0.479 | 0.927 / 0.925 | 0.495 / 0.494 | 0.927 / 0.950 | 0.811 / 0.825 |

- **Diffusion re-clustering is a small, consistent gain:** +0.009 ACC oracle K for `sub_var_coarse01` (seed std 0.007-0.008), ARI boot
  +0.013 / +0.023; R3 / R4 are unchanged. Per class it helps Hatchback (0.92-0.96 -> 0.995) and SUV (0.85-0.87 -> 0.88-0.91) and hurts Coupe
  (0.79-0.80 -> 0.75-0.76).
- **Diffusion maps on the frozen features alone give the best own assignment** (ACC own 0.829, above every trained run): on Cars the clustering
  algorithm matters about as much as the trained head. Not run on CIFAR-100.

**Loss-weight grid** (mean over seeds 0-2; reference var_floor 1 = `sub_var_coarse01` re-cluster above: Cars 0.810 / d0/d1 0.480 / d1/d2 0.927 / var
ratio 0.495 / ARI boot 0.927, CIFAR-100 0.911 / 0.685 / 0.996 / 0.730 / 0.991; frozen Cars 0.800, CIFAR-100 0.892).

| setting | Cars ACC oracle K | Cars d0/d1 | Cars d1/d2 | Cars var ratio | Cars ARI boot | CIFAR ACC oracle K | CIFAR d0/d1 | CIFAR d1/d2 | CIFAR ARI boot |
|---|---|---|---|---|---|---|---|---|---|
| sub1_c01_f025 | 0.760 | 0.265 | 0.629 | 0.278 | 0.861 | 0.845 | 0.494 | 0.576 | 0.973 |
| sub1_c03_f025 | 0.682 | 0.355 | 0.477 | 0.369 | 0.796 | 0.656 | 0.671 | 0.413 | 0.855 |
| sub2_c01_f025 | 0.787 | 0.229 | 0.742 | 0.239 | 0.890 | 0.851 | 0.449 | 0.637 | 0.975 |
| sub2_c03_f025 | 0.714 | 0.304 | 0.553 | 0.316 | 0.831 | 0.835 | 0.528 | 0.527 | 0.938 |
| sub1_c01_f05 | 0.808 | 0.369 | 0.761 | 0.384 | 0.924 | 0.885 | 0.595 | 0.748 | 0.969 |
| sub1_c03_f05 | 0.743 | 0.460 | 0.600 | 0.476 | 0.844 | 0.837 | 0.677 | 0.619 | 0.917 |
| sub2_c01_f05 | 0.814 | 0.321 | 0.874 | 0.335 | 0.931 | 0.883 | 0.545 | 0.813 | 0.968 |
| sub2_c03_f05 | 0.773 | 0.389 | 0.669 | 0.404 | 0.880 | 0.878 | 0.608 | 0.687 | 0.967 |
| sub1_c01_f075 | 0.817 | 0.430 | 0.853 | 0.444 | 0.932 | 0.906 | 0.647 | 0.888 | 0.989 |
| sub1_c03_f075 | 0.779 | 0.521 | 0.687 | 0.536 | 0.879 | 0.886 | 0.709 | 0.735 | 0.961 |
| sub2_c01_f075 | 0.811 | 0.382 | 0.949 | 0.396 | 0.929 | 0.902 | 0.600 | 0.957 | 0.988 |
| sub2_c03_f075 | 0.797 | 0.449 | 0.753 | 0.464 | 0.904 | 0.900 | 0.654 | 0.788 | 0.980 |

Seed std of ACC oracle K <= 0.013 (Cars) and <= 0.015 (CIFAR-100).
- **No setting improves on var_floor 1.** Cars: the best settings (sub1_c01_f075 0.817, sub2_c01_f05 0.814) are within one seed std of
  0.810. CIFAR-100: every setting is below 0.911 (best 0.906), and all var_floor 0.25 / 0.5 settings are below frozen.
- **A lower var_floor trades R3 for R2 and R4.** d0/d1 falls (Cars 0.48 -> 0.23, CIFAR-100 0.69 -> 0.45): subclasses get tighter relative to their
  neighbours. But the within-subclass variation (var ratio) and ACC oracle K fall with it: the head squeezes images towards prototypes
  that are partly wrong.
- **lambda_coarse 0.3 is worse than 0.1 in every cell** (Cars -0.02 to -0.08, CIFAR-100 up to -0.19), less so at higher floors.
- **lambda_sub 2 helps only at low floors** (Cars f025 0.760 -> 0.787); at f075 it is level or lower but gives the best d1/d2 (0.95 on both datasets).
- Sensitivity only: these are test numbers, and the defaults (var_floor 1, lambda_sub 1, lambda_coarse 0.1) stay as they were.

## Prototype refresh vs EMA momentum (`configs/ksub_maskcon_new_refresh.yaml`; val, DINOv3-L, seeds 0-2)
Stanford Cars maskcon_new, batch 256, loss of `sub_var_coarse01`. Four arms, fixed in advance: `ema099` (EMA 0.99, no refresh: the original),
`refresh1` / `refresh5` (prototypes reset to a fresh k-means on z every 1 / 5 epochs, EMA 0.99 in between), `ema095` (EMA 0.95, no refresh:
can faster updates replace the refresh?). Per epoch and coarse class, the train ARI of the current assignments vs a fresh k-means on z,
measured before that epoch's refresh (`ksub_ari/<run>.csv`; Sedan and SUV are the classes that lose most at batch 256). The arm is chosen on
val subclass accuracy (ACC oracle K, ACC own; `split: val` / `eval_split: val`); test is not used. Conflicting images excluded (Cars 211).
```
CONFIG=configs/ksub_maskcon_new_refresh.yaml METHODS="frozen ema099" sbatch slurm/train_ksub.sh   # and METHODS=refresh1, refresh5, ema095
CONFIG=configs/eval_repr_ksub_maskcon_new_refresh_val.yaml sbatch slurm/eval_representation.sh
python scripts/summarize_ksub.py --config configs/ksub_maskcon_new_refresh.yaml
```

Results (val, slurm jobs 433583-433586 train, 437628 R1-R5; git `1862299-dirty`; per-epoch ARI in
`reports/ksub_maskcon_new_refresh_val/ari_per_epoch.csv`, summary `reports/ksub_maskcon_new_refresh_val/summary.md`). Mean ± std over seeds 0-2.

| | ACC oracle K | ACC own | ARI boot | kNN coarse | Sedan ACC oracle / own | SUV ACC oracle / own | train ARI vs fresh, epoch 100 (Sedan / SUV) |
|---|---|---|---|---|---|---|---|
| frozen | 0.810 | 0.818 ± 0.002 | 0.772 | 0.954 | 0.761 / 0.766 | 0.886 / 0.872 | - |
| ema099 | 0.683 ± 0.010 | 0.630 ± 0.011 | 0.668 ± 0.007 | 0.957 ± 0.002 | 0.496 / 0.363 | 0.528 / 0.452 | 0.38 / 0.43 |
| refresh1 | **0.816 ± 0.003** | **0.831 ± 0.006** | **0.779 ± 0.003** | 0.966 ± 0.005 | 0.748 / 0.772 | 0.868 / 0.884 | 0.99 / 0.97 |
| refresh5 | 0.767 ± 0.009 | 0.771 ± 0.002 | 0.725 ± 0.007 | 0.967 ± 0.002 | 0.633 / 0.693 | 0.707 / 0.767 | 0.57 / 0.72 |
| ema095 | 0.669 ± 0.017 | 0.602 ± 0.009 | 0.650 ± 0.011 | 0.962 ± 0.004 | 0.499 / 0.332 | 0.510 / 0.416 | 0.37 / 0.42 |

- **Refreshing every epoch is the val choice**; it is the only arm at or above frozen DINOv3-L (ACC oracle +0.006, ACC own +0.013). Sedan and SUV
  recover to about frozen (0.75 / 0.87 vs 0.76 / 0.89); the gain is in Cab, Convertible, Coupe. Refreshing every 5 epochs recovers about 2/3 of the gap.
- **Faster EMA does not replace the refresh:** EMA 0.95 is no better than 0.99 (ACC oracle 0.669 vs 0.683; ARI vs fresh k-means as low).
  Without a refresh the assignments drift away from a fresh k-means on z from epoch ~10 (Sedan ARI 0.82 at epoch 10 -> 0.38 at 100).
- **Agreement tracks accuracy:** over (arm, class) with K_c >= 13, Spearman rho between the final train ARI vs fresh k-means and val
  ACC oracle K / ACC own is 0.89 / 0.90 (0.94 / 0.93 on the five classes with K_c >= 18).
- In `refresh5` the ARI drops within the first epoch after a reset (Sedan 0.64 at cycle epoch 1, 0.60 at epoch 5): the drift is fast, so the
  refresh interval has to be short.
- Val is small (Sedan 205, SUV 149 images); differences between `refresh1` and frozen are within ~2 std.

## Soft-assignment temperature tau_a (`configs/ksub_maskcon_new_tau.yaml`; val, DINOv3-L, seeds 0-2)
On the val choice above (`refresh1`: batch 256, `sub_var_coarse01` loss, refresh every epoch, EMA 0.99), tau_a in {0.025, 0.05, 0.10}; tau,
loss weights, momentum, batch and epochs fixed. 0.025: sharper r (fewer blended neighbours, but wrong assignments are reinforced and fewer
positive pairs); 0.10: softer r (borderline images update prototypes, but subclasses overlap more). tau_a 0.05 reuses the `refresh1` runs
(identical training; R1-R5 re-scored in the new report dir, should match). Primary: val ACC oracle K; secondary: ACC own; paired per-seed
differences vs 0.05, per class for Sedan, SUV, Cab, Convertible, Coupe. Logged per epoch and class, before the refresh (an agreement measured after a reset
mostly reflects the reset): train ARI vs a fresh k-means on z and the mean assignment entropy H(r_i) / log K_c (not available for the
reused 0.05 runs). Conflicting images excluded (Cars 211).
```
CONFIG=configs/ksub_maskcon_new_tau.yaml METHODS=tau0025 sbatch slurm/train_ksub.sh   # and METHODS=tau010
CONFIG=configs/eval_repr_ksub_maskcon_new_tau_val.yaml sbatch slurm/eval_representation.sh
python scripts/summarize_ksub.py --config configs/ksub_maskcon_new_tau.yaml
```

Results (val; slurm jobs 438523 / 438527 train, 439522 R1-R5; git `1862299-dirty`; summary `reports/ksub_maskcon_new_tau_val/summary.md`,
per-epoch ARI / entropy `ari_per_epoch.csv` there). Re-scored tau_a 0.05 (`refresh1`) reproduces the earlier val numbers exactly.

| tau_a | ACC oracle K | ACC own | ARI boot | kNN coarse | paired diff vs 0.05, ACC oracle K (s0 / s1 / s2) | paired diff, ACC own |
|---|---|---|---|---|---|---|
| frozen | 0.810 | 0.818 ± 0.002 | 0.772 | 0.954 | | |
| 0.025 | **0.834 ± 0.013** | 0.824 ± 0.010 | **0.792 ± 0.005** | 0.962 ± 0.004 | +0.006 / +0.035 / +0.011 (mean +0.018) | -0.009 / +0.006 / -0.020 (-0.008) |
| 0.05 | 0.816 ± 0.003 | **0.831 ± 0.006** | 0.779 ± 0.003 | 0.966 ± 0.005 | | |
| 0.10 | 0.529 ± 0.009 | 0.527 ± 0.010 | 0.264 ± 0.010 | 0.930 ± 0.001 | -0.300 / -0.279 / -0.282 (-0.287) | -0.300 / -0.292 / -0.320 (-0.304) |

Per class, ACC oracle K / ACC own (mean over seeds): Sedan 0.748 / 0.772 (0.05) -> 0.780 / 0.795 (0.025); SUV 0.868 / 0.884 -> 0.875 / 0.886;
Cab 0.716 / 0.742 -> 0.756 / 0.751; Convertible 0.799 / 0.797 -> 0.805 / 0.759; Coupe 0.809 / 0.838 -> 0.840 / 0.801. ACC oracle K at 0.025 is
above 0.05 in 12 of 15 (class, seed) pairs (SUV s0 -0.074, Convertible s1 / s2 -0.024 / -0.008).
- **tau_a 0.10 collapses the subclass signal.** Assignment entropy / log K_c reaches 0.98 by epoch 5 and ~1.0 after: r is uniform over the
  K_c prototypes, L_sub stays at 3.3 (0.9 at 0.025), and the classes spread out (d0/d1 0.83, d1/d2 0.38). Pre-refresh ARI vs fresh k-means
  falls to 0.44-0.47 (mean of the five classes, epochs 20-80). Every class and seed loses 0.11-0.36 ACC oracle K.
- **tau_a 0.025 is the val choice on the primary metric**, but the margin is small: +0.018 ACC oracle K, positive on all three seeds (+0.006 to +0.035),
  while ACC own is -0.008 (two of three seeds negative); within ~1.5 std of 0.05. Assignments are almost hard (entropy / log K_c 0.01-0.02).
- **Sharper assignments stay closer to a fresh k-means before each refresh:** pre-refresh ARI over epochs 20-100 is higher at 0.025 in all five
  classes (Sedan 0.867 vs 0.832, Cab 0.953 vs 0.904; mean of five 0.926 vs 0.900 at epoch 60). Entropy at 0.05 was not logged (reused runs).

## Coarse-loss weight at tau_a 0.025 (`configs/ksub_maskcon_new_coarse.yaml`; val, DINOv3-L, seeds 0-2)
Can a weak L_coarse improve coarse separation without losing the gain from sharper assignments? lambda_coarse in {0, 0.03, 0.10} at tau_a 0.025,
refresh every epoch, EMA 0.99, lambda_sub 1, lambda_var 1 (var_floor 1), batch 256, tau 0.1, 100 epochs; nothing else changes. lambda_coarse 0.10
reuses the `tau0025` runs (identical; R1-R5 re-scored in the new report dir, should match). Primary: val ACC oracle K; alongside: R1 probe
coarse (linear probe) and kNN coarse, R3 (d0/d1, d1/d2), ACC own; paired per-seed differences vs 0.10; pre-refresh ARI and entropy logged as before.
Conflicting images excluded (Cars 211).
```
CONFIG=configs/ksub_maskcon_new_coarse.yaml METHODS=tau0025_c0 sbatch slurm/train_ksub.sh   # and METHODS=tau0025_c003
CONFIG=configs/eval_repr_ksub_maskcon_new_coarse_val.yaml sbatch slurm/eval_representation.sh
python scripts/summarize_ksub.py --config configs/ksub_maskcon_new_coarse.yaml
```

Choice rule, fixed before the val results: highest mean val ACC oracle K; if two arms are within one seed std of each other, the one with the
higher val probe coarse.

Results (val; slurm job 461649 R1-R5; git `2b82f53-dirty`; summary `reports/ksub_maskcon_new_coarse_val/summary.md`). Re-scored lambda_coarse 0.10
(`tau0025`) reproduces `reports/ksub_maskcon_new_tau_val/` exactly. Mean ± std over seeds 0-2.

| lambda_coarse | ACC oracle K | ACC own | NMI | ARI boot | probe coarse | kNN coarse | d0/d1 | d1/d2 | var ratio |
|---|---|---|---|---|---|---|---|---|---|
| frozen | 0.810 | 0.818 ± 0.002 | 0.910 | 0.772 | **0.974** | 0.954 | 0.409 | 0.880 | 0.421 |
| 0 (`tau0025_c0`) | **0.837 ± 0.011** | **0.837 ± 0.008** | **0.918** | **0.834 ± 0.005** | 0.341 ± 0.007 | 0.923 | 0.405 | 1.032 | 0.417 |
| 0.03 (`tau0025_c003`) | 0.831 ± 0.012 | 0.834 ± 0.006 | 0.916 | 0.824 ± 0.009 | 0.563 ± 0.019 | 0.953 | 0.429 | 1.023 | 0.442 |
| 0.10 (`tau0025`) | 0.834 ± 0.013 | 0.824 ± 0.010 | 0.914 | 0.792 ± 0.005 | 0.958 ± 0.003 | **0.962** | 0.480 | **0.940** | 0.491 |

- **The three arms tie on subclass accuracy:** ACC oracle K 0.831-0.837, all within one seed std (0.011-0.013) of each other; all are
  about +0.02-0.03 above frozen DINOv3-L.
- **By the rule, lambda_coarse 0.10 (`tau0025`) is the val choice:** among the tied arms it alone keeps the coarse structure (probe coarse 0.958 vs
  0.341 / 0.563; d1/d2 0.94 vs 1.03: without L_coarse the body types are no further apart than their subclasses).
- Dropping L_coarse gives slightly more stable clusters (ARI boot 0.834 vs 0.792) and ACC own (+0.013), not a higher ACC oracle K.

## Final Cars configuration on test (`configs/ksub_maskcon_new_final.yaml`; test, DINOv3-L, seeds 0-2)
The configuration chosen on val by the three sweeps above (refresh every epoch, tau_a 0.025, lambda_coarse from the coarse-loss sweep), scored once on test
against frozen DINOv3-L. No new training: the runs above already wrote their test embeddings. The val winner is lambda_coarse 0.10 (`tau0025`):
the final configuration is batch 256, k-means refresh every epoch, EMA 0.99, tau_a 0.025, lambda_coarse 0.1, lambda_sub 1, lambda_var 1 (var_floor 1).
The test reference for the earlier val choice (`refresh1`, tau_a 0.05) is `sub_var_coarse01` of `configs/ksub_maskcon_new_recluster.yaml` (same settings).
Conflicting images excluded (Cars 211).
```
CONFIG=configs/eval_repr_ksub_maskcon_new_final_test.yaml sbatch slurm/eval_representation.sh
python scripts/summarize_ksub.py --config configs/ksub_maskcon_new_final.yaml   # reports/ksub_maskcon_new_final/summary.md
```

Results (test; slurm job 463730; summary `reports/ksub_maskcon_new_final/summary.md`, per class `per_class.csv`). Mean ± std over seeds 0-2.

| | ACC oracle K | ACC own | NMI | ARI boot | kNN coarse | probe coarse | d0/d1 | d1/d2 | var ratio |
|---|---|---|---|---|---|---|---|---|---|
| frozen DINOv3-L | 0.800 | 0.792 ± 0.002 | 0.886 | 0.874 | 0.944 | **0.973** | 0.415 | 0.877 | 0.429 |
| earlier val choice (tau_a 0.05; `sub_var_coarse01` re-cluster) | 0.810 ± 0.008 | 0.811 | 0.886 | 0.927 | **0.978** | | 0.480 | 0.927 | 0.495 |
| **final (tau_a 0.025)** | **0.820 ± 0.009** | **0.813 ± 0.011** | **0.887** | **0.951** | 0.974 | 0.971 | 0.473 | 0.937 | 0.487 |

- **The val choice holds on test:** +0.020 ACC oracle K over frozen DINOv3-L (all three seeds above: 0.809 / 0.822 / 0.827) and +0.021 ACC own;
  clustering is much more stable (ARI boot 0.951 vs 0.874); coarse accuracy is kept (probe 0.971 vs 0.973, kNN +0.03).
- tau_a 0.025 adds +0.010 ACC oracle K and +0.024 ARI boot on test over tau_a 0.05, as on val (+0.018).
- Per class (ACC oracle K, frozen -> final): Cab 0.718 -> 0.787, Sedan 0.732 -> 0.778, SUV 0.821 -> 0.859, Coupe 0.767 -> 0.806, Van 0.683 -> 0.727,
  Wagon 0.846 -> 0.848; still below frozen: Hatchback 0.993 -> 0.960, Convertible 0.838 -> 0.792.
- NMI is level (0.887 vs 0.886) and the subclasses are less compact relative to each other (d0/d1 0.47 vs 0.42): the gain is in how the
  clusters are cut, not in tighter subclasses.

## Final configuration on the make scheme (`configs/ksub_make_final.yaml`; test, DINOv3-L, seeds 0-2)
Stanford Cars with coarse = 49 makes, hidden subclass = 189 make-models (K_c 1-22; 19 makes have K_c = 1). The final `maskcon_new` configuration
is carried over unchanged (tau_a 0.025, lambda_coarse 0.1, lambda_sub 1, lambda_var 1, k-means refresh every epoch, EMA 0.99). Batching is as
`configs/ksub_cars.yaml` (16 makes x 16 images = 256). Nothing was chosen on this scheme, and test is scored once. This supersedes the collapsed
first runs above. Slurm jobs 470780 (train) and 470781 (R1-R5, `configs/eval_repr_ksub_make_final.yaml`); git `da60077-dirty`; summary
`reports/ksub_make_final/summary.md`, per class `per_class.csv`. Conflicting images excluded: 154 (`make` list). Mean ± std over seeds. Clustering
columns are over the 30 makes with K_c >= 2. Baseline rows are from `reports/repr/summary.md` (DINOv3-B, seed 0).

| | kNN / probe make | Recall@1 | fine probe | ACC oracle K | NMI oracle K | ACC K-hat | ARI K-hat | d0/d1 | d1/d2 | var ratio | ARI boot | ACC own |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen DINOv3-L | 0.973 / **0.983** | **0.948** | **0.967** | 0.782 | 0.703 | 0.698 | 0.551 | **0.570** | **0.709** | **0.635** | 0.886 | 0.787 |
| **final (`tau0025`)** | **0.980** / 0.979 | 0.936 | 0.959 | **0.803 ± 0.004** | **0.710** | **0.789 ± 0.011** | **0.639** | 0.656 | 0.794 | 0.707 | **0.981 ± 0.001** | **0.801 ± 0.004** |
| first runs, `full` (collapsed) | 0.971 | | | 0.543 | 0.292 | | | 0.936 | 0.489 | 0.951 | 0.231 | 0.460 |
| GEORGE (DINOv3-B, best baseline) | 0.977 / 0.982 | 0.938 | 0.968 | 0.746 | 0.652 | 0.638 | 0.495 | 0.678 | 0.653 | 0.734 | 0.829 | |
| oracle: fine classifier | 0.901 / 0.860 | 0.922 | 0.929 | 0.928 | 0.827 | 0.849 | 0.764 | 0.526 | 0.824 | 0.601 | 0.921 | |

- **The `maskcon_new` configuration transfers without retuning:** +0.021 ACC oracle K over frozen DINOv3-L (seed std 0.004), the same gain as on
  `maskcon_new` (+0.020). It is the best non-oracle encoder on subclass clustering, but still 0.13 below the oracle.
- **Clustering becomes much more stable** (ARI boot 0.886 -> 0.981, above the oracle's 0.921). ACC K-hat rises 0.698 -> 0.789, but z was trained with
  K: only the K estimate on top of it is label-free.
- **As on `maskcon_new`, the subclasses do not get tighter.** d0/d1 0.570 -> 0.656 and var ratio 0.635 -> 0.707, and makes come closer relative to
  their models (d1/d2 0.709 -> 0.794). Recall@1 and the fine probe drop slightly (-0.012 / -0.008). The gain is in how the clusters are cut.
- Per make (ACC oracle K, frozen -> final): 10 of 30 rise by more than 0.01 (Acura 0.749 -> 0.962, Spyker 0.631 -> 0.837, Lamborghini +0.13,
  Suzuki and Bentley +0.11, Hyundai +0.08). 6 fall (HUMMER 0.963 -> 0.712, FIAT -0.07, Aston Martin -0.05, Dodge, Honda, Land Rover -0.01 to -0.02).
  The large swings are makes with 2-4 models and few test images.
