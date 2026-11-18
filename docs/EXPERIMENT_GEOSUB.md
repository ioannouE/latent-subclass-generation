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
