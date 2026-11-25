# geosub experiment plan (geometry-preserving subprototypes)

Method: `lsgen/methods/geosub/`, `scripts/train_geosub.py`, `configs/geosub_maskcon.yaml`. Label scheme `maskcon_new` (configs/labels.yaml: coarse = 8 body
types, hidden subclass = the 196 official classes); any scheme via `labels`. The scheme's conflicting images (`data/splits/exclude_maskcon_new.txt`,
docs/DATA.md) are left out of every experiment below. Fine labels are used in evaluation only; loss weights are chosen on `val`, `test` is scored once for the chosen runs.

## Method
A frozen backbone (DINOv3-B by default) gives L2-normalised features f. A residual head gives z = norm(f + g(f)); the last layer of g starts at zero,
so z = f before training. Each coarse class has `n_proto` learnable subprototypes (initialised on random features of the class). Loss = weighted sum of
- **L_geo**: per anchor, KL(p_f || p_z) with p = softmax of the batch cosines / tau_geo; the only term that reads the frozen similarities;
- **L_supcon**: SupCon on the coarse labels (`lsgen/methods/repr/losses.py`);
- **L_sub**: augmentation-consistent assignment: each view's soft assignment to its class's prototypes is trained towards the sharpened, no-gradient
  assignment of the other view, plus a weak entropy of each class's mean assignment against collapse (no equipartition, as NHP);
- **L_var**: per class, relu(1 - trace cov(z_c) / trace cov(f_c)): z may not shrink the frozen within-class spread.

**K-hat (label-free)**: a prototype is active if it is the nearest prototype of >= `min_count` = 10 train images (fixed before any run);
K-hat_c = number of active prototypes. Images are assigned to their nearest active prototype.

The backbone is frozen, so its features are cached once: 8 random augmentations of every train crop (the baselines' augmentation) and the mirror
image of every crop. Training then uses only cached features and coarse labels: any dataset with cached features and a coarse label works.

## Order (one step at a time; stop and report after each)
1. Fine probe on frozen DINOv3-B and L: `reports/repr_maskcon_new/dinov3_{base,large}_256/r2_within_make.csv` (configs/eval_repr_maskcon_geosub.yaml).
   With the old exclusion list (`reports/repr_maskcon/`): probe_fine 0.948 / 0.947, Recall@1 0.928 / 0.930; R2 ACC oracle K 0.725 / 0.794, K-hat 0.634 / 0.656.
2. `geo` alone: sanity check, R1-R5 should equal the frozen backbone's.
3. `geo_sub`: ACC with K-hat, K-hat error and flip-ARI (`scripts/eval_geosub.py`), and R1-R5.
4. `geo_sub_supcon{01,03,1,3}`: L_supcon weight against L_geo, chosen on `val`.
5. `geo_sub_supcon*_var`: only if the ablation shows that within-class variance collapses (R4 var_ratio, pr_subclass).

## Commands
```
sbatch slurm/train_geosub.sh                                               # caches views (once), trains all methods (or METHODS="geo")
sbatch slurm/eval_geosub.sh                                                # prototype clusters + frozen k-means reference on val (split: test for the final report)
CONFIG=configs/eval_repr_maskcon_geosub.yaml sbatch slurm/eval_representation.sh   # R1-R5 of z on test
```

## Results (single seed 0, DINOv3-B, 256 px, maskcon_new; conflicting images excluded: 7,231 train / 803 val / 7,940 test)

**Steps 1-2.** Frozen DINOv3-B / L on test: fine probe 0.950 / 0.950, Recall@1 0.934 / 0.935, R2 ACC oracle K 0.727 / 0.800, K-hat 0.659 / 0.660.
`geo` reproduces frozen DINOv3-B on every R1-R5 number (L_geo stays ~0): the head and normalisation lose nothing.

**Steps 3-4, val, geosub's own prototype clusters** (`reports/repr_maskcon_new/<run>_clusters_val/`; macro over the 8 body types, true K_c 24.5):

| Run | ACC (own K-hat) | ARI | flip-ARI | K-hat / class | val kNN coarse |
|---|---|---|---|---|---|
| reference: k-means on frozen DINOv3-B, label-free K-hat (R2 pipeline, same images) | 0.656 | 0.527 | - | 18.3 | - |
| reference: k-means on frozen DINOv3-B, oracle K (uses K) | 0.768 | 0.644 | - | - | - |
| `geo` (prototypes never trained: nearest random train image) | 0.644 | 0.476 | 0.865 | 30.0 | 0.953 |
| `geo_sub` | **0.699** | **0.561** | 0.987 | 22.2 | 0.956 |
| `geo_sub` + supcon 0.1 / 0.3 / 1 / 3 | 0.661 / 0.632 / 0.509 / 0.080 | 0.504 / 0.464 / 0.359 / 0.000 | 0.99-1.00 | 20.6 / 19.5 / 12.5 / 1.0 | 0.960 / 0.965 / 0.981 / 0.984 |

- L_sub helps its own clusters (+0.055 ACC over untrained prototypes) and makes them flip-invariant. `geo_sub` beats k-means with K-hat on the frozen
  features by +0.043 ACC (6 of 8 body types; Sedan +0.09, where the k-means K-hat search stops at 25 for 50 classes), above the +0.03 margin set
  before the comparison. It stays far below oracle-K k-means (0.768). K-hat error -2.2 (abs 5.2): Sedan 35 for 50, small classes over-split by 2-5.
- L_supcon trades subclasses for coarse accuracy at every weight; at weight 3 each body type collapses onto one prototype. Val choice: weight 0 (`geo_sub`).
- Caveats: val has about 4 images per fine class, one seed. The bootstrap CIs of `clusters.csv` lie above the point estimates (resampling repeats images
  and drops rare classes, which inflates Hungarian ACC): quote point estimates only.

**Test, the val choice `geo_sub` scored once** (`reports/repr_maskcon_new/{geosub_geo_sub_bodytype_new_s0,dinov3_base_256}_clusters_test/`):

| Clustering (same 7,940 test images) | ACC | ARI | K-hat / class |
|---|---|---|---|
| k-means on frozen DINOv3-B, label-free K-hat | **0.659** | **0.601** | 20.4 |
| `geo_sub` prototypes (own K-hat) | 0.645 | 0.580 | 22.2 |
| k-means on frozen DINOv3-B, oracle K (uses K) | 0.727 | 0.658 | - |

The val gain does not hold: -0.014 ACC. geosub is ahead only where the k-means K-hat search is capped (Sedan, 50 classes, K <= 25: 0.499 vs 0.405)
and in 3 small classes; without Sedan it is behind (0.666 vs 0.695 macro over the other 7). Flip-ARI 0.984, K-hat error -2.25 (abs 5.25), as on val.
**Verdict (rule fixed before the comparison: > +0.03 over frozen k-means K-hat): geosub does not improve on frozen DINOv3-B; stop.** The +0.043 on val
was within the noise of 803 images (about 4 per fine class) plus the Sedan cap.

**DINOv3-L and the full model** (all four terms, L_supcon 0.1, L_var 1), test, full table with every maskcon_new baseline in
`reports/repr_maskcon_new/summary.md`. On DINOv3-L, `geo_sub` is the best label-free row: K-hat ACC 0.724 with k-means on its z and 0.713 with its own
prototypes, vs 0.660 for frozen DINOv3-L. Most of the gain is the Cab class, where the frozen K-hat rule picks K = 4 for 18 classes; without it, k-means on
`geo_sub` z is +0.02 and the prototypes are level. Oracle-K ACC is unchanged (0.795 vs 0.800). `full` adds nothing over `geo_sub` (0.704 / 0.710).
So the stop verdict holds for DINOv3-B; on DINOv3-L the effect is that label-free K selection fails less often. One seed: to be confirmed with more seeds.

**R1-R5 of z on test** (k-means on the embedding, as for every baseline): `geo_sub` keeps the fine information (probe 0.951, oracle-K ACC 0.734) but
its k-means K-hat ACC is lower than frozen (0.603 vs 0.659); with L_supcon 0.614-0.637. var_ratio 0.512 (`geo_sub`), 0.47-0.48 with L_supcon 0.1-1
(frozen 0.521). So the gain is in the prototype assignment, not in the geometry of z.

## Assumptions
- No EMA teacher in L_sub (the backbone is frozen and the head small); the other view's assignment is the target.
- n_proto = 52 (largest K_c + 2, as NHP), min_count = 10, tau and loss weights other than the L_supcon sweep are my choices, untuned.
- The R2 K-hat rule (k-means, K <= 25) is unchanged; geosub's own K-hat has no such cap (Sedan has 50 classes).

## `geo_sub` with n_proto 50, seeds 0-4, DINOv3-L, Stanford Cars `maskcon_new` and CIFAR-100 (test)
`configs/geosub_{maskcon_large_p50,cifar100}.yaml`: `geo_sub` (L_geo + L_sub), n_proto fixed to 50 on both datasets (not derived from K_c; the earlier
DINOv3-L run used 52 = largest K_c + 2), otherwise as `configs/geosub_maskcon_large.yaml`; on CIFAR-100 the views are cached from the native 32 px
images with the baselines' augmentation. Not tuned: `geo_sub` is the earlier val choice, test scored once. R1-R5: `configs/eval_repr_{maskcon_geosub,cifar100_geosub}.yaml`;
own clusters: `configs/eval_geosub_{maskcon_large,cifar100}_test.yaml`. Reports in `reports/repr_maskcon_new/` and `reports/cifar100/repr/`
(`geosub_geo_sub_*_p50_s<seed>_{256,32,clusters_test}/`). Conflicting images excluded: Stanford Cars 211 (`maskcon_new` list), CIFAR-100 90.
Trained and scored from uncommitted configs on top of `6957f1b` (`6957f1b-dirty`), committed unchanged afterwards. Mean ± std over seeds 0-4.

| | kNN coarse | fine probe | ACC oracle K | ACC K-hat | ARI K-hat | K-hat | d1/d2 | var ratio | ARI boot | own ACC (prototypes) | own K-hat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cars frozen DINOv3-L | 0.944 | 0.950 | **0.800** | 0.660 | 0.641 | 17.0 | 0.877 | 0.429 | 0.874 | - | - |
| Cars `geo_sub` | 0.945 ± 0.002 | 0.949 | 0.797 ± 0.004 | **0.709 ± 0.011** | **0.689 ± 0.008** | 18.0 | 0.886 | 0.416 | **0.894 ± 0.004** | 0.677 ± 0.016 | 21.2 (true 24.5) |
| CIFAR frozen DINOv3-L | 0.964 | 0.961 | 0.892 | 0.778 | 0.706 | 5.8 | 0.921 | 0.773 | 0.891 | - | - |
| CIFAR `geo_sub` | 0.965 ± 0.001 | 0.961 | 0.894 ± 0.002 | 0.782 ± 0.019 | 0.705 ± 0.018 | 5.6 | 0.929 | 0.752 | **0.911 ± 0.004** | 0.685 ± 0.017 | 5.0 (true 5) |

- **n_proto 50 vs 52 changes nothing** (Cars seed 0: every R1-R5 number within 0.004 of the n_proto 52 run).
- **Stanford Cars: k-means K-hat on z is +0.049 over frozen (5 seeds), but it is all one class.** Cab (18 classes): frozen K-hat 4, ACC 0.235;
  `geo_sub` z K-hat 12, ACC 0.622. Over the other 7 body types both are 0.721. The single-seed "+0.02 without Cab" above does not hold.
- **Stanford Cars: the prototypes are only +0.017 over frozen k-means K-hat** (0.677 ± 0.016 vs 0.660; the single-seed 0.713 above was a high seed) and
  under-split (K-hat 21.2 vs 24.5, abs error 5.6).
- **CIFAR-100: `geo_sub` z is level with frozen** (K-hat ACC +0.004 with std 0.019; oracle K +0.002). Only ARI boot rises (0.891 -> 0.911).
- **CIFAR-100: the prototypes are well below frozen k-means K-hat** (0.685 vs 0.778), although their count is right on average (5.0, abs error 1.9).
- Oracle-K ACC, fine probe and coarse kNN do not move on either dataset: geosub does not make the subclasses more separable. ksub (given K) is well
  above both on oracle-K ACC (Cars re-cluster 0.810, CIFAR `sub_var_coarse01` 0.919, docs/EXPERIMENT_KSUB.md), but that is the other supervision tier.

## Prototype initialisation, usage, known K, loss weights and a training diagnostic (DINOv3-L, seeds 0-4)
Configs: `configs/geosub_{maskcon_large,cifar100}_kmeans.yaml` (prototype fixes), `configs/geosub_maskcon_large_weights.yaml` (loss weights, Cars; the last settings and CIFAR-100 in the next section),
`configs/diagnose_geosub_maskcon.yaml` (`scripts/diagnose_geosub.py`). Everything as the n_proto 50 runs above except what is named. `km50` = prototypes
start at k-means centroids (n_init 10, run seed) of each class's frozen train features instead of random train features; `K` = geosub-K, K_c k-means
prototypes per class, all active (**uses K**). Weights `g`/`s`/`v` = L_geo / L_sub / L_var (default 1 / 1 / 0), `u` = usage (default 0.1). Chosen on
val (own-cluster ACC; within one seed std of the incumbent keeps the incumbent); test scored afterwards for every variant as an ablation, not used to
choose. Own clusters: `configs/eval_geosub_{maskcon_large,cifar100}{,_val,_test}.yaml`; R1-R5: `configs/eval_repr_{maskcon,cifar100}_geosub.yaml`.
Reports: `reports/repr_maskcon_new/` and `reports/cifar100/repr/` (`geosub_geo_sub_<variant>_*_s<seed>_{256,32,clusters_val,clusters_test}/`),
`reports/repr_maskcon_new/geosub_diagnostics/`. Conflicting images excluded: Stanford Cars 211, CIFAR-100 90. Code and configs committed in `1bfc312`.

**Own prototype clusters** (Hungarian ACC, macro over coarse classes, mean ± std over seeds; K-hat counted on train, true 24.5 Cars / 5 CIFAR):

| | Cars val | Cars test | Cars K-hat | CIFAR val | CIFAR test | CIFAR K-hat |
|---|---|---|---|---|---|---|
| frozen DINOv3-L k-means, label-free K-hat | 0.686 | 0.660 | 16.1 / 17.0 | 0.679 | **0.778** | 8.5 / 5.8 |
| random init (`geo_sub` p50, above) | 0.705 ± 0.015 | 0.677 ± 0.016 | 21.2 | **0.683 ± 0.017** | 0.685 ± 0.017 | 5.0 |
| **`km50`** | **0.794 ± 0.008** | **0.759 ± 0.009** | 28.7 | 0.602 ± 0.019 | 0.602 ± 0.016 | 5.8 |
| `km50` u 0.3 / u 1 | 0.780 / 0.703 | 0.744 / 0.665 | 29.8 / 34.7 | 0.558 / 0.205 | 0.548 / 0.192 | 22.6 / 49.5 |
| `km50` g 0.3 / g 0.1 / g 0.1 + v 1 | 0.796 / 0.795 / 0.795 | 0.757 / 0.757 / 0.758 | 28.8-29.0 | - | - | - |
| `km50` g 0 + v 1 | 0.762 ± 0.004 | 0.731 ± 0.006 | 28.9 | - | - | - |
| `km50` s 0.3 / s 0.1 | 0.794 / 0.793 | 0.760 / 0.760 | 28.8 | - | - | - |
| `km50` v 0.3 / v 1 / v 3 | 0.794 / 0.795 / 0.794 | 0.759 / 0.759 / 0.759 | 28.6-28.7 | - | - | - |
| `K` (uses K) | 0.813 ± 0.004 | 0.793 ± 0.004 | 24.5 | 0.840 ± 0.004 | 0.841 ± 0.003 | 5 |
| frozen DINOv3-L k-means, oracle K (uses K) | 0.810 | **0.800** | - | 0.893 | **0.892** | - |

**R1-R5 of z on test** (k-means on the embedding, as for every baseline; 5 seeds):

| | kNN / probe coarse | fine probe | ACC oracle K | ACC K-hat | ARI K-hat | K-hat | d0/d1 | var ratio | ARI boot |
|---|---|---|---|---|---|---|---|---|---|
| Cars frozen DINOv3-L | 0.944 / 0.973 | 0.950 | 0.800 | 0.660 | 0.641 | 17.0 | 0.415 | 0.429 | 0.874 |
| Cars `km50` | 0.943 / 0.952 | 0.950 | 0.809 | 0.706 | 0.685 | 18.6 | 0.419 | 0.433 | 0.896 |
| Cars `km50` g 0.1 | 0.934 / 0.867 | 0.948 | 0.808 | 0.680 | 0.657 | 21.2 | 0.407 | 0.421 | 0.914 |
| Cars `km50` g 0 + v 1 | 0.879 / 0.779 | 0.942 | 0.742 | 0.636 | 0.534 | 24.0 | 0.387 | 0.400 | 0.849 |
| Cars `K` (uses K) | 0.948 / 0.946 | 0.950 | 0.807 | 0.721 | 0.701 | 17.7 | 0.410 | 0.424 | 0.902 |
| CIFAR frozen DINOv3-L | 0.964 / 0.966 | 0.961 | 0.892 | 0.778 | 0.706 | 5.8 | 0.733 | 0.773 | 0.891 |
| CIFAR `km50` / u 0.3 / u 1 | 0.965 / 0.961 | 0.961-0.962 | 0.892-0.897 | 0.760 / 0.700 / 0.757 | 0.688 / 0.633 / 0.686 | 6.0-8.1 | 0.71-0.75 | 0.75-0.79 | 0.89-0.90 |
| CIFAR `K` (uses K) | 0.965 / 0.955 | 0.960 | 0.897 | **0.835** | **0.754** | 4.9 | 0.717 | 0.758 | **0.934** |

**Training diagnostic** (Cars, `scripts/diagnose_geosub.py`, train split; the run's own starting prototypes rebuilt from its config and seed, pruned
with the same min_count rule, no training):

| | ACC untrained start | ACC trained | trained prototypes on frozen f | images that changed prototype | K-hat start -> end | 1 - cos(z, f) | kNN@10 fine purity z (f 0.906) |
|---|---|---|---|---|---|---|---|
| random init | 0.608 | 0.667 | 0.666 | 33% | 29.7 -> 21.2 | 0.25 | 0.905 |
| `km50` | 0.686 | 0.758 | 0.762 | 12% | 33.0 -> 28.7 | 0.11 | 0.904 |
| `km50` g 0 + v 1 | 0.686 | 0.718 | 0.736 | 14% | 33.0 -> 28.9 | 0.34 | 0.877 |
| `km50` u 1 | 0.686 | 0.652 | 0.657 | 11% | 33.0 -> 34.7 | 0.02 | 0.904 |
| `K` (uses K) | 0.791 | 0.793 | 0.793 | 2% | 24.5 fixed | 0.02 | 0.906 |

Training logs (every term logged at every weight, `consist` = swapped-prediction part of L_sub): K-hat is fixed by epoch 4-5 of 100 in every run,
inside warmup; `consist` falls from 0.54 to 0.06 (0.004 without L_geo); L_var is 0 throughout (peak 8e-4) at every weight, also without L_geo.

- **Stanford Cars, label-free: `km50` is the val choice and the best label-free clustering so far** (test 0.759 vs 0.660 for frozen k-means K-hat,
  within 0.04 of oracle K). Val and test rank every variant the same way.
- **The gain is the prototype count, not the representation.** Trained prototypes on frozen f score as the full model, kNN fine purity and oracle-K ACC
  of z equal frozen; the gain over the untrained start comes with pruning the over-split k-means start (33 -> 28.7 per class: the sharp teacher moves
  images from small prototypes to large ones until they fall below min_count). With the count fixed (`K`) training moves 2% of the images and adds 0.002.
- **The loss weights do not matter once L_geo > 0** (Cars, all within one seed std). The L_geo weight only sets how far z drifts; without L_geo z
  collapses onto the prototypes (coarse probe 0.973 -> 0.779, own ACC -0.03). L_var is never active. A larger usage weight stops the pruning and hurts.
- **CIFAR-100: the fixes fail.** k-means initialisation does not stop the collapse (46 -> 5.6 prototypes per class by epoch 5; people, aquatic and
  small mammals, trees and large natural scenes collapse to 1-2); usage 0.3 / 1 over-split. Given K, training moves away from its k-means start and ends
  below frozen oracle-K k-means (0.841 vs 0.892). k-means with the label-free K-hat on `K`'s z is above frozen (0.835 vs 0.778), but that z was trained with K.
- **Remaining errors are pairs frozen DINOv3 merges** (Rolls-Royce Ghost / Phantom, Silverado 1500 cab variants, Audi TTS / TT RS, Dodge / Mercedes
  Sprinter): L_geo keeps z on that geometry and augmentation consistency carries no trim, cab or model-year information. A signal beyond augmentation
  (e.g. nearest-neighbour consistency) would be needed for the embedding itself to learn.
- The last loss-weight settings (0.5 / 0.5 / 0.5, 0.5 / 0 / 1, 0.5 / 0.5 / 1 on both datasets, 0 / 1 / 1 on CIFAR-100) are in the next section.
- Not varied for geosub: tau_geo, tau_student / tau_teacher, min_count, lr, batch size, epochs. Diffusion-map clustering, re-clustering frequency,
  EMA vs refresh and tau_a are ksub settings (docs/EXPERIMENT_KSUB.md).

## Remaining loss weights on both datasets (DINOv3-L, seeds 0-4)
`configs/geosub_{maskcon_large,cifar100}_weights.yaml`, otherwise as `km50` above (k-means init, n_proto 50, usage 0.1). g / s / v = L_geo / L_sub / L_var:
0.5 / 0.5 / 0.5 (with AdamW the same ratios as 1 / 1 / 1), 0.5 / 0 / 1 (no L_sub: the prototypes get no gradient and stay at their k-means start),
0.5 / 0.5 / 1, and on CIFAR-100 0 / 1 / 1 (Cars: above). Chosen on val with the rule above; test reported for every setting, not used to choose.
Slurm jobs 461668 / 461670 (train), 463731 / 463734 (own clusters, val), 463732 / 463735 (own clusters, test), 463733 / 463737 (R1-R5, test); git
`2b82f53-dirty` (configs unchanged since `1bfc312`). Reports as above (`geosub_geo_sub_km50_{g05_s05_v05,g05_s0_v1,g05_s05_v1,g0_v1}_*`).
Conflicting images excluded: Stanford Cars 211, CIFAR-100 90. Mean ± std over seeds 0-4. Own = the prototypes' clusters (Hungarian ACC, K-hat
counted on train, true 24.5 / 5); z = k-means on the embedding (R2, R1).

| | Cars own val | Cars own test | Cars own K-hat | Cars z: ACC oracle K / K-hat | Cars probe coarse | CIFAR own val | CIFAR own test | CIFAR own K-hat | CIFAR z: ACC oracle K / K-hat | CIFAR probe coarse |
|---|---|---|---|---|---|---|---|---|---|---|
| frozen DINOv3-L k-means, K-hat | 0.686 | 0.660 | 17.0 | 0.800 / 0.660 | 0.973 | 0.679 | **0.778** | 5.8 | 0.892 / **0.778** | 0.966 |
| random init (`geo_sub` p50) | 0.705 | 0.677 | 21.2 | 0.797 / 0.709 | | **0.683 ± 0.017** | 0.685 | 5.0 | 0.894 / 0.782 | |
| `km50` (1 / 1 / 0) | **0.794 ± 0.008** | **0.759 ± 0.009** | 28.7 | 0.809 / 0.706 | 0.952 | 0.602 ± 0.019 | 0.602 | 5.8 | 0.892 / 0.760 | 0.961 |
| 0.5 / 0.5 / 0.5 | 0.795 ± 0.010 | 0.759 ± 0.008 | 28.7 | 0.805 / 0.717 | 0.952 | 0.651 ± 0.005 | 0.645 ± 0.003 | 10.3 | 0.884 / 0.749 | 0.959 |
| 0.5 / 0.5 / 1 | 0.792 ± 0.011 | 0.760 ± 0.009 | 28.7 | 0.807 / 0.708 | 0.952 | 0.651 ± 0.011 | 0.645 ± 0.011 | 10.9 | 0.884 / 0.730 | 0.959 |
| 0.5 / 0 / 1 (no L_sub) | 0.741 ± 0.025 | 0.694 ± 0.029 | 33.0 | 0.794 / 0.674 | 0.973 | 0.214 ± 0.005 | 0.201 ± 0.004 | 49.5 | 0.892 / 0.778 | 0.965 |
| 0 / 1 / 1 | 0.762 ± 0.004 | 0.731 ± 0.006 | 28.9 | 0.742 / 0.636 | 0.779 | 0.640 ± 0.009 | 0.642 ± 0.010 | 9.1 | 0.733 / 0.659 | 0.917 |

- **Stanford Cars: `km50` stays the val choice.** 0.5 / 0.5 / 0.5 and 0.5 / 0.5 / 1 are within 0.003 of it on val and test (std 0.008-0.011), with the
  same K-hat and the same z (R1-R5 within 0.01; fine probe 0.950 throughout). L_var stays 0 in training (peak 2e-4).
- **Without L_sub the prototypes keep their k-means start, and the gain over it is gone** (Cars test 0.694 vs 0.759; K-hat 33.0, the untrained count in
  the diagnostic). z stays at frozen (probe coarse 0.973, ACC oracle K 0.794, ARI boot 0.876 vs 0.874). This confirms the diagnostic: the Cars gain comes
  from L_sub pruning the over-split start. On CIFAR-100 the start is not pruned at all (49.5 of 50 prototypes active, ACC 0.20).
- **CIFAR-100: halving L_geo and L_sub against L_var gives fewer collapses but over-splits** (K-hat 10.3-10.9 vs 5.8 for `km50`, true 5). Val is +0.05 over
  `km50` (0.651 vs 0.602), but still below the random-init incumbent (0.683 ± 0.017) and frozen k-means K-hat (0.679). The best label-free CIFAR-100 geosub
  is unchanged, and no geosub setting reaches frozen k-means K-hat on test (0.645 vs 0.778). Unlike Cars, L_var is non-zero early on CIFAR-100 (peak
  0.002-0.007 at epoch 3, all seeds, while the collapse happens). This is the likely reason 0.5 / 0.5 / 0.5 differs from `km50` here but not on Cars. Untested:
  `km50` v 1 was not run on CIFAR-100.
- **Without L_geo, z collapses on CIFAR-100 too** (probe coarse 0.966 -> 0.917, ACC oracle K 0.892 -> 0.733), as on Cars.
- Net: on Stanford Cars, geosub's label-free gain is prototype-count selection by L_sub on a k-means start. On CIFAR-100 no loss weighting
  gives a geosub clustering above frozen DINOv3-L k-means with K-hat.
