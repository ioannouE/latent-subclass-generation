# Representation metrics: frozen encoders and fine-tuned baselines (Milestones 4 and Step 2)

Provenance: seed 0, dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. Frozen encoders: `configs/eval_representation.yaml`
(`n_boot` 1000; git `f9f4e60`-dirty, `sscd_256`: `e499877`-dirty), oracle row: `configs/eval_repr_oracle.yaml`, fine-tuned baselines: `configs/eval_repr_baselines.yaml`
(git `83549f8`-dirty), GEORGE: `configs/eval_repr_george.yaml` (16 Oct). All rerun on 8 Oct 2026.
Fit/reference = `train`, scored on `test` (7,954 images; train 7,270; 49 makes, 30 with K_c >= 2, 19 with K_c = 1).
**Hidden subclass = make-model (189 classes, years merged)** since 7 Oct 2026; earlier results (6 Oct) used the 196 official make-model-year classes and
are superseded. Conflicting images are excluded everywhere: {'train': 59, 'val': 8, 'test': 87} (train / val / test, 154 in total; docs/DATA.md). Metric definitions:
docs/METRICS.md. All values are macro means over makes; per-make tables and CIs are in `<encoder>_<crop>/*.csv`.

**Status: final** for the make-model labels. R2 is stored as `r2_oracle.csv` (bootstrap CIs) and `r2_khat.csv` (subsample CIs).
`fine_classifier_128` is the supervised oracle reference (penultimate features of the fine classifier trained with the subclass labels), not a baseline.

## Summary (best first among the frozen encoders)

### R1 and within-make information
| encoder             |   R1 kNN make |   R1 probe make |   R2 Recall@1 (within make) |   R2 fine probe (within make) |
|:--------------------|--------------:|----------------:|----------------------------:|------------------------------:|
| dinov3_large_256    |         0.973 |           0.983 |                       0.948 |                         0.967 |
| dinov3_large_128    |         0.968 |           0.981 |                       0.941 |                         0.966 |
| dinov3_base_256     |         0.949 |           0.973 |                       0.947 |                         0.969 |
| dinov3_base_128     |         0.936 |           0.962 |                       0.941 |                         0.966 |
| clip_l_256          |         0.921 |           0.947 |                       0.856 |                         0.928 |
| clip_l_128          |         0.898 |           0.935 |                       0.846 |                         0.919 |
| sscd_256            |         0.242 |           0.343 |                       0.650 |                         0.729 |
| sscd_128            |         0.211 |           0.327 |                       0.625 |                         0.719 |
| fine_classifier_128 |         0.901 |           0.860 |                       0.922 |                         0.929 |

### R2 clustering (k-means per make). Oracle K ("uses K") and label-free K̂
The true mean K_c over makes with K_c >= 2 is 5.7.

| encoder             |   ACC (oracle K) |   NMI (oracle K) |   ARI (oracle K) |   ACC (Khat) |   NMI (Khat) |   ARI (Khat) |   mean Khat |
|:--------------------|-----------------:|-----------------:|-----------------:|-------------:|-------------:|-------------:|------------:|
| dinov3_large_256    |            0.782 |            0.703 |            0.634 |        0.698 |        0.650 |        0.551 |       5.433 |
| dinov3_large_128    |            0.792 |            0.706 |            0.641 |        0.686 |        0.631 |        0.537 |       5.567 |
| dinov3_base_256     |            0.717 |            0.613 |            0.526 |        0.619 |        0.554 |        0.439 |       6.500 |
| dinov3_base_128     |            0.710 |            0.599 |            0.513 |        0.599 |        0.521 |        0.416 |       5.367 |
| clip_l_256          |            0.717 |            0.612 |            0.541 |        0.581 |        0.556 |        0.426 |       3.067 |
| clip_l_128          |            0.704 |            0.586 |            0.510 |        0.566 |        0.534 |        0.411 |       3.000 |
| sscd_256            |            0.452 |            0.170 |            0.128 |        0.333 |        0.024 |        0.017 |       1.467 |
| sscd_128            |            0.445 |            0.168 |            0.120 |        0.333 |        0.025 |        0.014 |       1.800 |
| fine_classifier_128 |            0.928 |            0.827 |            0.829 |        0.849 |        0.792 |        0.764 |       5.167 |

### R2 macro means with 95% CIs (oracle K: bootstrap; K̂: 63.2% subsample)
| encoder             | ACC (oracle K)       | ACC (Khat)           | NMI (Khat)           | mean Khat            |
|:--------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| dinov3_large_256    | 0.782 [0.761, 0.804] | 0.698 [0.674, 0.713] | 0.650 [0.621, 0.675] | 5.433 [5.200, 6.200] |
| dinov3_large_128    | 0.792 [0.753, 0.797] | 0.686 [0.659, 0.704] | 0.631 [0.607, 0.663] | 5.567 [5.033, 6.100] |
| dinov3_base_256     | 0.717 [0.696, 0.746] | 0.619 [0.592, 0.644] | 0.554 [0.517, 0.587] | 6.500 [5.233, 6.801] |
| dinov3_base_128     | 0.710 [0.688, 0.736] | 0.599 [0.581, 0.637] | 0.521 [0.496, 0.572] | 5.367 [4.967, 6.667] |
| clip_l_256          | 0.717 [0.678, 0.729] | 0.581 [0.549, 0.605] | 0.556 [0.509, 0.581] | 3.067 [2.467, 4.100] |
| clip_l_128          | 0.704 [0.669, 0.718] | 0.566 [0.541, 0.598] | 0.534 [0.490, 0.565] | 3.000 [2.400, 3.601] |
| sscd_256            | 0.452 [0.430, 0.468] | 0.333 [0.311, 0.345] | 0.024 [0.007, 0.036] | 1.467 [1.167, 2.800] |
| sscd_128            | 0.445 [0.423, 0.460] | 0.333 [0.311, 0.341] | 0.025 [0.004, 0.030] | 1.800 [1.100, 2.701] |
| fine_classifier_128 | 0.928 [0.902, 0.926] | 0.849 [0.820, 0.860] | 0.792 [0.771, 0.809] | 5.167 [4.767, 5.367] |

### R3 hierarchy geometry (desired: Δ0 < Δ1 < Δ2)
| encoder             |    d0 |    d1 |    d2 |   d0/d1 |   d1/d2 |   % Eq1 |   % Eq2 |
|:--------------------|------:|------:|------:|--------:|--------:|--------:|--------:|
| dinov3_large_256    | 0.702 | 1.303 | 1.844 |   0.570 |   0.709 | 100.000 |  96.667 |
| dinov3_large_128    | 0.721 | 1.302 | 1.817 |   0.585 |   0.719 | 100.000 |  96.667 |
| dinov3_base_256     | 0.829 | 1.355 | 1.834 |   0.630 |   0.744 | 100.000 |  96.667 |
| dinov3_base_128     | 0.851 | 1.358 | 1.814 |   0.645 |   0.754 | 100.000 |  96.667 |
| clip_l_256          | 0.502 | 0.653 | 0.887 |   0.767 |   0.743 | 100.000 |  96.667 |
| clip_l_128          | 0.424 | 0.552 | 0.737 |   0.772 |   0.756 | 100.000 |  96.667 |
| sscd_256            | 1.761 | 1.826 | 1.855 |   0.966 |   0.985 | 100.000 |  86.667 |
| sscd_128            | 1.735 | 1.798 | 1.824 |   0.968 |   0.986 | 100.000 |  86.667 |
| fine_classifier_128 | 0.837 | 1.630 | 1.977 |   0.526 |   0.824 | 100.000 |  96.667 |

### R4 variation and R5 stability
| encoder             |   var_ratio |   eff rank |   ARI boot |   Khat err |
|:--------------------|------------:|-----------:|-----------:|-----------:|
| dinov3_large_256    |       0.635 |      8.920 |      0.886 |      1.579 |
| dinov3_large_128    |       0.649 |      9.346 |      0.863 |      1.211 |
| dinov3_base_256     |       0.690 |      8.202 |      0.837 |      1.158 |
| dinov3_base_128     |       0.703 |      8.611 |      0.836 |      1.211 |
| clip_l_256          |       0.813 |     18.967 |      0.664 |      0.421 |
| clip_l_128          |       0.817 |     18.956 |      0.658 |      0.632 |
| sscd_256            |       0.974 |     31.466 |      0.321 |      1.000 |
| sscd_128            |       0.976 |     31.396 |      0.315 |      0.000 |
| fine_classifier_128 |       0.601 |     13.766 |      0.921 |      1.421 |

## Reading the results
- **DINOv3-L is best on almost every axis** (DINOv3-B is level on within-make Recall@1 and marginally ahead on the fine probe). 256 px is better than 128 px for make kNN; for subclass clustering the two sizes are within the CIs (DINOv3-L oracle-K ACC 0.792 at 128 px vs 0.782 at 256 px). CLIP-L is clearly behind DINOv3 on subclass structure (R2, R4, R5). SSCD is the negative reference: a copy-detection encoder has almost no make or subclass structure (kNN make 0.24, R3 ratios near 1), as it should.
- **The subclass problem is not solved by a frozen encoder.** Even DINOv3-L recovers subclasses only moderately: ACC 0.78 with the true K and 0.70 with K̂. R4 var. ratio 0.64 means most within-make variance is not explained by the subclass (CLIP 0.81). The supervised oracle features reach ACC 0.93, so the information is learnable with subclass labels.
- The subclass information is present, just not recoverable by clustering: the within-make fine probe is 0.97 for DINOv3 (0.93 for CLIP) while oracle-K clustering gets 0.72-0.78.
- Eq. (1) (Δ0 < Δ1) holds for 100% of makes in every encoder; Eq. (2) (Δ1 < Δ2) for 87-97%.
- R4 effective rank is not comparable across encoders of different dimension (SSCD 512, DINOv3-B 768, DINOv3-L / CLIP 1024; the oracle row has the classifier's own width).
- `ari_seeds` is averaged over only 3 seed pairs per make, so its CI is not informative (see `r5.csv`).

## Fine-tuned contrastive baselines (Step 2, seed 0, DINOv3-B, 256 px crops)
Last 4 blocks of DINOv3-B fine-tuned with one fixed recipe (`configs/repr_baselines.yaml`, 50 epochs, untuned), coarse (make) labels only; same R1-R5 protocol as above, macro means with 95% CIs where available. FALCON uses the number of subclasses K = 189. MaskCon and FALCON were added and run separately from the code in `lsgen/methods/repr/`; their implementations are not reviewed here.

| encoder                     | R1 kNN make          | Recall@1             | fine probe           | ACC oracle           | ACC Khat             |   d0/d1 |   d1/d2 | var ratio            |   ARI boot |   Khat err K=1 |   mean Khat |
|:----------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|--------:|--------:|:---------------------|-----------:|---------------:|------------:|
| frozen DINOv3-B (reference) | 0.949 [0.941, 0.956] | 0.947 [0.941, 0.953] | 0.969 [0.964, 0.973] | 0.717 [0.696, 0.746] | 0.619 [0.592, 0.644] |   0.63  |   0.744 | 0.690 [0.686, 0.694] |      0.837 |           1.16 |        6.5  |
| SupCon (make labels)        | 0.968 [0.962, 0.973] | 0.946 [0.940, 0.952] | 0.969 [0.963, 0.974] | 0.718 [0.694, 0.739] | 0.634 [0.596, 0.639] |   0.645 |   0.719 | 0.703 [0.699, 0.708] |      0.83  |           1.26 |        6.3  |
| SimCLR                      | 0.813 [0.801, 0.825] | 0.869 [0.860, 0.877] | 0.945 [0.939, 0.950] | 0.583 [0.578, 0.624] | 0.478 [0.420, 0.494] |   0.778 |   0.847 | 0.821 [0.818, 0.824] |      0.709 |           2.68 |        7.3  |
| SupCon + SimCLR             | 0.942 [0.933, 0.950] | 0.924 [0.917, 0.931] | 0.961 [0.956, 0.967] | 0.695 [0.673, 0.723] | 0.595 [0.540, 0.610] |   0.738 |   0.813 | 0.787 [0.784, 0.790] |      0.758 |           3.68 |        6.57 |
| MaskCon                     | 0.953 [0.946, 0.960] | 0.946 [0.940, 0.952] | 0.971 [0.966, 0.975] | 0.705 [0.685, 0.731] | 0.582 [0.573, 0.622] |   0.648 |   0.744 | 0.707 [0.702, 0.711] |      0.83  |           1.47 |        6.33 |
| FALCON                      | 0.967 [0.961, 0.974] | 0.941 [0.935, 0.948] | 0.967 [0.962, 0.971] | 0.698 [0.673, 0.721] | 0.613 [0.574, 0.620] |   0.7   |   0.71  | 0.754 [0.750, 0.758] |      0.812 |           1.47 |        6.53 |

- **SupCon helps the coarse level only:** make kNN 0.949 -> 0.968 (CIs do not overlap); every subclass metric (oracle-K ACC 0.718 vs 0.717, Recall@1, fine probe) stays at the frozen level and Δ0/Δ1 is slightly worse (0.630 -> 0.645). It tightens makes (Δ1/Δ2 0.744 -> 0.719) without separating subclasses better inside them.
- **SimCLR hurts on every metric** (make kNN 0.813, oracle-K ACC 0.583, K̂ error on K_c = 1 makes 2.68): instance discrimination spreads images of one make apart by pose and background rather than by model. The val kNN fell steadily during training (0.92 at epoch 10 -> 0.86 at epoch 50).
- **SupCon + SimCLR** ends slightly below frozen on R1 and R2 (kNN 0.942, ACC 0.695); the instance term costs part of what the coarse term gains.
- **MaskCon and FALCON** match SupCon on the coarse level (kNN 0.953 / 0.967) and are level with the frozen encoder on subclass clustering (ACC 0.705 / 0.698 vs 0.717, inside the CIs); FALCON has the worst Δ0/Δ1 of the three (0.700) despite using K.
- **No variant improves subclass discovery (R2, Δ0/Δ1, R5)**, and all over-fragment single-subclass makes (K̂ error 1.3-3.7 vs 1.16 frozen). This is the gap the benchmark is meant to expose.
- One seed only: differences of a few hundredths in R2 between frozen, SupCon, MaskCon and FALCON are within the CIs and should not be read as ranking.
- CI flaw: the only point estimates outside their CI are the oracle row's oracle-K ACC (0.928 vs [0.902, 0.926], bootstrap bias at high ACC) and `sscd_128` `ari_khat`. Subsampling is slightly biased when the clustering is very strong or very weak. No conclusion depends on it.

## GEORGE (Step 2, seed 0, DINOv3-B, 256 px crops, 16 Oct 2026)
GEORGE (Sohoni et al. 2020; official code in `third_party/hidden-stratification`, driver `scripts/train_george.py`, `configs/repr_george.yaml`) with the same recipe as the baselines above (last 4 blocks, 50 epochs per stage, untuned), make labels only. Stage 1: ERM on the make (`george_erm`). Stage 2: per make, UMAP (2-d) then a GMM with K chosen by silhouette over 2..24. Stage 3: a fresh backbone trained on the make with group DRO over the (make, cluster) groups (`george`, the primary row). Evaluated on the 768-d CLS embedding with the same R1-R5 protocol; point estimates are macro means over makes (CIs in `george_*_256/*.csv`; one seed). GEORGE's own output is the cluster assignment, which R1-R5 cannot score, so the embedding of the group-DRO model stands in for it. Evaluator git `7a20388`-dirty.

| encoder                     | R1 kNN make | Recall@1 | fine probe | ACC oracle | NMI oracle | ACC Khat | NMI Khat |   d0/d1 |   d1/d2 | var ratio | ARI boot | Khat err K=1 | mean Khat |
|:----------------------------|------------:|---------:|-----------:|-----------:|-----------:|---------:|---------:|--------:|--------:|----------:|---------:|-------------:|----------:|
| frozen DINOv3-B (reference) |       0.949 |    0.947 |      0.969 |      0.717 |      0.613 |    0.619 |    0.554 |   0.630 |   0.744 |     0.690 |    0.837 |         1.16 |       6.5 |
| SupCon (make labels)        |       0.968 |    0.946 |      0.969 |      0.718 |      0.604 |    0.634 |    0.579 |   0.644 |   0.719 |     0.703 |    0.830 |         1.26 |       6.3 |
| FALCON                      |       0.967 |    0.941 |      0.967 |      0.698 |      0.589 |    0.613 |    0.561 |   0.700 |   0.710 |     0.754 |    0.812 |         1.47 |       6.5 |
| GEORGE stage 1 (ERM, make CE) |     0.978 |    0.940 |      0.966 |      0.743 |      0.652 |    0.640 |    0.595 |   0.682 |   0.644 |     0.739 |    0.834 |         1.42 |       5.3 |
| **GEORGE (group DRO)**      |   **0.977** |    0.938 |      0.968 |  **0.746** |  **0.652** |    0.638 |    0.610 |   0.678 |   0.653 |     0.734 |    0.829 |         1.42 |       6.5 |

GEORGE's own clustering (stage 2, on train activations) chose K = 2 for 18 makes and K = 3-7 for most others (mean 4.3, max 17 over the 49 makes; true mean K_c is 5.7 over makes with K_c >= 2). Its search starts at K = 2, so the 19 single-subclass makes are always split.

- **GEORGE is the best fine-tuned baseline on the coarse level and a little above frozen on subclass clustering:** make kNN 0.977 (frozen 0.949, SupCon 0.968), oracle-K ACC 0.746 / NMI 0.652 against 0.717 / 0.613 frozen and 0.718 / 0.604 SupCon. The gain in oracle-K ACC (+0.03) is about the width of the 95% CIs of the other rows (+-0.02-0.03; GEORGE's macro CIs were not computed), with one seed, so it should not be read as a ranking. With estimated K the gain is nil (ACC 0.638 vs 0.619-0.634).
- **The gain comes from stage 1, not from GEORGE's clustering:** the ERM model alone (make cross-entropy, no clusters) scores the same as the group-DRO model (oracle-K ACC 0.743 vs 0.746, kNN 0.978 vs 0.977). Group DRO over the discovered clusters adds nothing measurable here. This is the expected outcome: the DRO objective still only predicts the make, so it gives the backbone no signal to separate the clusters inside a make.
- **No subclass metric improves in a way that matters:** Recall@1 and the fine probe are slightly below frozen (0.938 / 0.968 vs 0.947 / 0.969), Delta1/Delta2 is the worst of all models (0.65; makes become tight and compact relative to the gap between makes, as for any make-supervised model) and the var ratio is 0.73 (frozen 0.69, so a larger share of within-make variance is not subclass). Single-subclass makes are still over-split (K-hat error 1.42).
- **Comparison with the other methods:** GEORGE with fine-tuning on make labels behaves like SupCon, with a somewhat tighter make structure; it is above NHP (oracle-K ACC 0.58) and level with the other baselines inside the CIs. It does not close the gap to the supervised oracle features (ACC 0.93). Hidden-subclass discovery from coarse labels is not solved by GEORGE on Stanford Cars.
- Limits: one seed; stage-2 settings (UMAP 2-d, GMM, K <= 24) and group-DRO settings (step size 0.01, size adjustment 2) are the defaults declared before the run and were not tuned; GEORGE's overclustering and hardness components are not used.

**GEORGE's own clusters (`scripts/eval_george_clusters.py`, `reports/repr/george_clusters_s0/`).** Scored on the *train* images (the only ones GEORGE clusters), so not comparable to the test tables above; macro over the 30 makes with K_c >= 2. Reference: k-means on the frozen DINOv3-B train embeddings, same images and metric code.

| clustering                                   | ACC                  | NMI                  | ARI                  | mean K |
|:---------------------------------------------|:---------------------|:---------------------|:---------------------|-------:|
| GEORGE (ERM activations, UMAP, GMM, own K)   | 0.701 [0.698, 0.721] | 0.667 [0.662, 0.687] | 0.559 [0.551, 0.583] |    5.4 |
| frozen DINOv3-B, k-means, oracle K (uses K)  | 0.704 [0.689, 0.733] | 0.599 [0.586, 0.633] | 0.511 [0.490, 0.550] |    5.7 |
| frozen DINOv3-B, k-means, label-free K-hat   | 0.592 [0.581, 0.631] | 0.530 [0.507, 0.582] | 0.421 [0.391, 0.461] |    6.4 |

- **As a clustering method GEORGE is the best label-free option here:** without knowing K it matches k-means with the oracle K on ACC (0.701 vs 0.704) and is above it on NMI (0.667 vs 0.599, CIs do not overlap) and ARI (0.559 vs 0.511, CIs touch). Against the label-free K-hat pipeline used for the other encoders it is clearly better (ACC +0.11, NMI +0.14, ARI +0.14), and it picks K close to the true mean (5.4 vs 5.7) where the frozen K-hat overshoots (6.4).
- **This contrasts with the embedding result above:** GEORGE's clusters are useful, but training on them (group DRO) does not put that structure into the representation (R2 on the group-DRO embedding is level with the baselines). The cluster-quality gain is not available to the R1-R5 embedding metrics, which is why this separate row exists.
- Caveats: scored on train, where the stage-1 ERM model has fit the make labels (final train loss 0.011) and the clustering sees the same images it is scored on; K is chosen on this data, whereas the frozen-K-hat row also chooses it label-free on train. Which of the two ingredients helps (make-supervised ERM features, or UMAP + GMM) is not separated: k-means on the ERM embedding was not run. Over-splitting of the 19 single-subclass makes (forced K >= 2) is not part of these numbers. One seed.

## Full comparison by objective (DINOv3-B family at 256 px, seed 0; oracle = fine classifier, 128 px, not a baseline)
Macro means over makes, test split, 16 Oct 2026 (point estimates; CIs in `<encoder>_<crop>/*.csv`). Bold = best among the non-oracle rows (not for the R3 ratios, where lower is not simply better). Effective rank depends on the embedding width (NHP 128-d, CLIP-L / DINOv3-L 1024-d), so it is not ranked. Conflicting images excluded.

**O1, coarse distinctions (R1, R3 Eq. 2)**

| encoder                         | R1 kNN make   | R1 probe make   |   R3 % Eq1 |   R3 % Eq2 | R3 d1/d2   |
|:--------------------------------|:--------------|:----------------|-----------:|-----------:|:-----------|
| DINOv3-B frozen                 | 0.949         | 0.973           |        100 |       96.7 | 0.744      |
| DINOv3-L frozen                 | 0.973         | **0.983**       |        100 |       96.7 | 0.709      |
| CLIP-L frozen                   | 0.921         | 0.947           |        100 |       96.7 | 0.743      |
| SupCon                          | 0.968         | 0.975           |        100 |       96.7 | 0.719      |
| SimCLR                          | 0.813         | 0.929           |        100 |       90   | 0.847      |
| SupCon+SimCLR                   | 0.942         | 0.971           |        100 |       96.7 | 0.813      |
| MaskCon                         | 0.953         | 0.972           |        100 |       96.7 | 0.744      |
| FALCON                          | 0.967         | 0.979           |        100 |      100   | 0.710      |
| NHP (z_sub)                     | 0.965         | 0.975           |        100 |      100   | 0.727      |
| GEORGE stage 1 (ERM)            | **0.978**     | 0.982           |        100 |      100   | 0.644      |
| GEORGE (group DRO)              | 0.977         | 0.982           |        100 |      100   | 0.653      |
| Oracle (fine classifier, 128px) | 0.901         | 0.860           |        100 |       96.7 | 0.824      |

**O2, subclasses (R2, R3 Eq. 1, R5)**

| encoder                         | Recall@1   | fine probe   | ACC (true K)   | NMI (true K)   | ARI (true K)   | ACC (K̂)   | NMI (K̂)   | ARI (K̂)   |   mean K̂ | K̂ err (K_c=1)   | ARI boot   | R3 d0/d1   |
|:--------------------------------|:-----------|:-------------|:---------------|:---------------|:---------------|:----------|:----------|:----------|---------:|:----------------|:-----------|:-----------|
| DINOv3-B frozen                 | 0.947      | 0.969        | 0.717          | 0.613          | 0.526          | 0.619     | 0.554     | 0.439     |      6.5 | 1.16            | 0.837      | 0.630      |
| DINOv3-L frozen                 | **0.948**  | 0.967        | **0.782**      | **0.703**      | **0.634**      | **0.698** | **0.650** | **0.551** |      5.4 | 1.58            | **0.886**  | 0.570      |
| CLIP-L frozen                   | 0.856      | 0.928        | 0.717          | 0.612          | 0.541          | 0.581     | 0.556     | 0.426     |      3.1 | **0.42**        | 0.664      | 0.767      |
| SupCon                          | 0.946      | 0.969        | 0.718          | 0.604          | 0.517          | 0.634     | 0.579     | 0.472     |      6.3 | 1.26            | 0.830      | 0.645      |
| SimCLR                          | 0.869      | 0.945        | 0.583          | 0.423          | 0.324          | 0.478     | 0.420     | 0.287     |      7.3 | 2.68            | 0.709      | 0.778      |
| SupCon+SimCLR                   | 0.924      | 0.961        | 0.695          | 0.568          | 0.485          | 0.595     | 0.557     | 0.430     |      6.6 | 3.68            | 0.758      | 0.738      |
| MaskCon                         | 0.946      | **0.971**    | 0.705          | 0.592          | 0.500          | 0.582     | 0.526     | 0.408     |      6.3 | 1.47            | 0.830      | 0.648      |
| FALCON                          | 0.941      | 0.967        | 0.698          | 0.589          | 0.494          | 0.613     | 0.561     | 0.454     |      6.5 | 1.47            | 0.812      | 0.700      |
| NHP (z_sub)                     | 0.912      | 0.930        | 0.580          | 0.437          | 0.306          | 0.541     | 0.421     | 0.293     |      4.3 | 1.32            | 0.881      | 0.614      |
| GEORGE stage 1 (ERM)            | 0.940      | 0.966        | 0.743          | 0.652          | 0.565          | 0.640     | 0.595     | 0.495     |      5.3 | 1.42            | 0.834      | 0.682      |
| GEORGE (group DRO)              | 0.938      | 0.968        | 0.746          | 0.652          | 0.563          | 0.638     | 0.610     | 0.495     |      6.5 | 1.42            | 0.829      | 0.678      |
| Oracle (fine classifier, 128px) | 0.922      | 0.929        | 0.928          | 0.827          | 0.829          | 0.849     | 0.792     | 0.764     |      5.2 | 1.42            | 0.921      | 0.526      |

**O3, variation within a subclass (R4)**

| encoder                         | R4 var ratio (lower = more subclass-explained)   |   R4 effective rank |
|:--------------------------------|:-------------------------------------------------|--------------------:|
| DINOv3-B frozen                 | 0.690                                            |                 8.2 |
| DINOv3-L frozen                 | **0.635**                                        |                 8.9 |
| CLIP-L frozen                   | 0.813                                            |                19   |
| SupCon                          | 0.703                                            |                 8.5 |
| SimCLR                          | 0.821                                            |                11.9 |
| SupCon+SimCLR                   | 0.787                                            |                12.5 |
| MaskCon                         | 0.707                                            |                 7.6 |
| FALCON                          | 0.754                                            |                 9.5 |
| NHP (z_sub)                     | 0.679                                            |                 4.9 |
| GEORGE stage 1 (ERM)            | 0.739                                            |                10.9 |
| GEORGE (group DRO)              | 0.734                                            |                10.6 |
| Oracle (fine classifier, 128px) | 0.601                                            |                13.8 |

## Known problems
1. **K̂ bootstrap CIs were invalid (fixed, verified on the rerun).** Resamples contain duplicate images; duplicates form tight pairs and inflate the silhouette-selected K̂. Example (DINOv3-L 256): macro K̂ = 5.43 with CI [12.97, 16.33]; CLIP-L 128 `acc_khat` 0.56 with CI [0.29, 0.33]. Fix: K̂ metrics now take their CIs from draws of 63.2% of each make's images without replacement (`khat_subsample` in `configs/eval_representation.yaml`, table `r2_khat.csv`); oracle-K, R1 and R3 keep the standard bootstrap. R4 `var_ratio` was found to have the same duplicate-image bias and now also uses the subsample CIs; R4 `pr_subclass` (effective rank) has no CI because it depends on the sample size (rerun 6 Oct 2026, git `3a4acd7`). The point estimates did not change, and the new K̂ CIs contain them (on the 8 Oct rerun the exception is `sscd_128` `ari_khat`).
2. **K̂ never returns 1 for single-subclass makes with DINOv3 features: kept as a finding, not a bug.** `khat_error` on the 19 K_c = 1 makes has a minimum of 1 for every DINOv3 run and every fine-tuned baseline (mean 1.2-3.7), so the rule never returns 1 for them. CLIP-L does return K̂ = 1 for about 55-60% of these makes (mean error 0.4-0.6), and SSCD for almost all, because they have little structure. The analysis below was run on 6 Oct, before the switch to make-model labels, and was not repeated (the 19 K_c = 1 makes and their images are unchanged apart from the exclusion update). A check on `train` embeddings (not test) shows no rule fixes this across encoders: in DINOv3 space the best silhouette of K_c = 1 makes (median 0.24-0.27) is about that of K_c >= 2 makes (0.28-0.29), and the gap statistic alone returns 1 for only 4-7 of the 19. A threshold that fixes CLIP (SIL_MIN = 0.25) underestimates K for K_c >= 2 makes by 1.8 and does nothing for DINOv3. Tuning it per encoder would tune the metric to the encoders, so the fixed rule stays (SIL_MIN = 0.10). Likely explanation (untested): real viewpoint/colour structure inside one official class, which is exactly the over-fragmentation a label-free per-make K̂ pipeline would pass on to a generator. Details: docs/METRICS.md.

## Not covered here
Generation metrics (G1-G7, B1-B6) and the controls are in `reports/report.md` and `reports/metric_validation.md`; no generative method has been scored yet.
