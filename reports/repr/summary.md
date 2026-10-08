# Representation metrics: frozen encoders and fine-tuned baselines (Milestones 4 and Step 2)

Provenance: seed 0, dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. Frozen encoders: `configs/eval_representation.yaml`
(`n_boot` 1000; git `f9f4e60`-dirty, `sscd_256`: `e499877`-dirty), oracle row: `configs/eval_repr_oracle.yaml`, fine-tuned baselines: `configs/eval_repr_baselines.yaml`
(git `83549f8`-dirty). All rerun on 8 Oct 2026.
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

## Known problems
1. **K̂ bootstrap CIs were invalid (fixed, verified on the rerun).** Resamples contain duplicate images; duplicates form tight pairs and inflate the silhouette-selected K̂. Example (DINOv3-L 256): macro K̂ = 5.43 with CI [12.97, 16.33]; CLIP-L 128 `acc_khat` 0.56 with CI [0.29, 0.33]. Fix: K̂ metrics now take their CIs from draws of 63.2% of each make's images without replacement (`khat_subsample` in `configs/eval_representation.yaml`, table `r2_khat.csv`); oracle-K, R1 and R3 keep the standard bootstrap. R4 `var_ratio` was found to have the same duplicate-image bias and now also uses the subsample CIs; R4 `pr_subclass` (effective rank) has no CI because it depends on the sample size (rerun 6 Oct 2026, git `3a4acd7`). The point estimates did not change, and the new K̂ CIs contain them (on the 8 Oct rerun the exception is `sscd_128` `ari_khat`).
2. **K̂ never returns 1 for single-subclass makes with DINOv3 features: kept as a finding, not a bug.** `khat_error` on the 19 K_c = 1 makes has a minimum of 1 for every DINOv3 run and every fine-tuned baseline (mean 1.2-3.7), so the rule never returns 1 for them. CLIP-L does return K̂ = 1 for about 55-60% of these makes (mean error 0.4-0.6), and SSCD for almost all, because they have little structure. The analysis below was run on 6 Oct, before the switch to make-model labels, and was not repeated (the 19 K_c = 1 makes and their images are unchanged apart from the exclusion update). A check on `train` embeddings (not test) shows no rule fixes this across encoders: in DINOv3 space the best silhouette of K_c = 1 makes (median 0.24-0.27) is about that of K_c >= 2 makes (0.28-0.29), and the gap statistic alone returns 1 for only 4-7 of the 19. A threshold that fixes CLIP (SIL_MIN = 0.25) underestimates K for K_c >= 2 makes by 1.8 and does nothing for DINOv3. Tuning it per encoder would tune the metric to the encoders, so the fixed rule stays (SIL_MIN = 0.10). Likely explanation (untested): real viewpoint/colour structure inside one official class, which is exactly the over-fragmentation a label-free per-make K̂ pipeline would pass on to a generator. Details: docs/METRICS.md.

## Not covered here
Generation metrics (G1-G7, B1-B6) and the controls are in `reports/report.md` and `reports/metric_validation.md`; no generative method has been scored yet.
