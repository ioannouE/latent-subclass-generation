# Frozen-encoder representation metrics (Milestone 4, first run)

Provenance: git `ea90a10279b6d0a8ffb736a6e2b7fe0450380b7d`, seed 0, dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`, config `configs/eval_representation.yaml` (`n_boot` 1000), created 2026-10-02.
Fit/reference = `train`, scored on `test` (7,940 images; 49 makes, 30 with K_c >= 2, 19 with K_c = 1). Conflicting images are excluded everywhere: {'train': 68, 'val': 9, 'test': 101} (train / val / test, 178 in total; docs/DATA.md). Metric definitions: docs/METRICS.md. All values are macro means over makes; per-make tables and CIs are in `<encoder>_<crop>/*.csv`.

**Status: final.** Rerun on 5-6 Oct 2026 (git `93ee89b`) after the K̂ CI fix; point estimates are identical to the first run (K̂ rule unchanged, see problem 2). R2 is stored as `r2_oracle.csv` (bootstrap CIs) and `r2_khat.csv` (subsample CIs).

## Summary (best first)

### R1 and within-make information
| encoder          |   R1 kNN make |   R1 probe make |   R2 Recall@1 (within make) |   R2 fine probe (within make) |
|:-----------------|--------------:|----------------:|----------------------------:|------------------------------:|
| dinov3_large_256 |         0.973 |           0.983 |                       0.944 |                         0.963 |
| dinov3_large_128 |         0.968 |           0.98  |                       0.937 |                         0.962 |
| dinov3_base_256  |         0.949 |           0.972 |                       0.944 |                         0.965 |
| dinov3_base_128  |         0.936 |           0.964 |                       0.938 |                         0.962 |
| clip_l_256       |         0.922 |           0.947 |                       0.847 |                         0.923 |
| clip_l_128       |         0.898 |           0.936 |                       0.838 |                         0.914 |
| sscd_256         |         0.241 |           0.343 |                       0.639 |                         0.722 |
| sscd_128         |         0.21  |           0.327 |                       0.614 |                         0.712 |

### R2 clustering (k-means per make). Left: oracle K ("uses K"); right: label-free K̂ (provisional)
Columns 1-3 use the true K_c; columns 4-7 use K̂. The true mean K_c over makes with K_c >= 2 is 5.9.

| encoder          |   ACC (oracle K) |   NMI (oracle K) |   ARI (oracle K) |   ACC (K̂) |   NMI (K̂) |   ARI (K̂) |   mean K̂ |
|:-----------------|-----------------:|-----------------:|-----------------:|----------:|----------:|----------:|---------:|
| dinov3_large_256 |            0.796 |            0.728 |            0.665 |     0.711 |     0.662 |     0.571 |    5.433 |
| dinov3_large_128 |            0.811 |            0.733 |            0.68  |     0.699 |     0.645 |     0.561 |    5.6   |
| dinov3_base_256  |            0.733 |            0.636 |            0.56  |     0.63  |     0.568 |     0.462 |    6.433 |
| dinov3_base_128  |            0.717 |            0.626 |            0.543 |     0.609 |     0.534 |     0.437 |    5.4   |
| clip_l_256       |            0.716 |            0.627 |            0.554 |     0.581 |     0.563 |     0.431 |    3.133 |
| clip_l_128       |            0.706 |            0.605 |            0.527 |     0.56  |     0.536 |     0.412 |    3     |
| sscd_256         |            0.446 |            0.179 |            0.135 |     0.309 |     0.026 |     0.018 |    1.433 |
| sscd_128         |            0.437 |            0.174 |            0.125 |     0.316 |     0.03  |     0.02  |    1.8   |

### R2 macro means with 95% CIs (oracle K: bootstrap; K̂: 63.2% subsample)
| encoder          | ACC (oracle K)       | ACC (K̂)              | NMI (K̂)              | mean K̂               |
|:-----------------|:---------------------|:---------------------|:---------------------|:---------------------|
| dinov3_large_256 | 0.796 [0.777, 0.819] | 0.711 [0.685, 0.723] | 0.662 [0.635, 0.688] | 5.433 [5.200, 6.233] |
| dinov3_large_128 | 0.811 [0.769, 0.814] | 0.699 [0.669, 0.714] | 0.645 [0.618, 0.675] | 5.600 [5.066, 6.067] |
| dinov3_base_256  | 0.733 [0.711, 0.759] | 0.630 [0.601, 0.655] | 0.568 [0.531, 0.600] | 6.433 [5.267, 6.800] |
| dinov3_base_128  | 0.717 [0.702, 0.750] | 0.609 [0.590, 0.646] | 0.534 [0.508, 0.586] | 5.400 [5.000, 6.667] |
| clip_l_256       | 0.716 [0.684, 0.729] | 0.581 [0.544, 0.600] | 0.563 [0.516, 0.585] | 3.133 [2.500, 4.167] |
| clip_l_128       | 0.706 [0.674, 0.719] | 0.560 [0.536, 0.592] | 0.536 [0.495, 0.570] | 3.000 [2.433, 3.633] |
| sscd_256         | 0.446 [0.425, 0.464] | 0.309 [0.284, 0.318] | 0.026 [0.007, 0.038] | 1.433 [1.167, 2.767] |
| sscd_128         | 0.437 [0.418, 0.455] | 0.316 [0.283, 0.312] | 0.030 [0.003, 0.030] | 1.800 [1.067, 2.700] |

### R3 hierarchy geometry (desired: Δ0 < Δ1 < Δ2)
| encoder          |    Δ0 |    Δ1 |    Δ2 |   Δ0/Δ1 |   Δ1/Δ2 |   % makes Eq.(1) |   % makes Eq.(2) |
|:-----------------|------:|------:|------:|--------:|--------:|-----------------:|-----------------:|
| dinov3_large_256 | 0.685 | 1.302 | 1.844 |   0.554 |   0.708 |              100 |           96.667 |
| dinov3_large_128 | 0.705 | 1.301 | 1.817 |   0.57  |   0.718 |              100 |           96.667 |
| dinov3_base_256  | 0.814 | 1.355 | 1.834 |   0.616 |   0.744 |              100 |           93.333 |
| dinov3_base_128  | 0.836 | 1.357 | 1.814 |   0.631 |   0.753 |              100 |           93.333 |
| clip_l_256       | 0.499 | 0.651 | 0.887 |   0.763 |   0.741 |              100 |           96.667 |
| clip_l_128       | 0.422 | 0.551 | 0.737 |   0.769 |   0.754 |              100 |           96.667 |
| sscd_256         | 1.759 | 1.826 | 1.855 |   0.965 |   0.985 |              100 |           86.667 |
| sscd_128         | 1.734 | 1.798 | 1.824 |   0.967 |   0.986 |              100 |           86.667 |

### R4 variation and R5 stability
| encoder          |   R4 var. ratio |   R4 eff. rank |   R5 ARI (bootstrap) |   R5 K̂ error (K_c=1) |
|:-----------------|----------------:|---------------:|---------------------:|---------------------:|
| dinov3_large_256 |           0.619 |          8.981 |                0.878 |                1.579 |
| dinov3_large_128 |           0.634 |          9.4   |                0.858 |                1.211 |
| dinov3_base_256  |           0.676 |          8.225 |                0.843 |                1.158 |
| dinov3_base_128  |           0.691 |          8.616 |                0.845 |                1.211 |
| clip_l_256       |           0.807 |         18.865 |                0.659 |                0.421 |
| clip_l_128       |           0.812 |         18.87  |                0.659 |                0.632 |
| sscd_256         |           0.973 |         30.459 |                0.312 |                1     |
| sscd_128         |           0.975 |         30.386 |                0.301 |                0     |

## Reading the results
- **DINOv3-L is best on almost every axis** (DINOv3-B is level or marginally ahead on within-make Recall@1 and fine probe); 256 px is only slightly better than 128 px. CLIP-L is clearly behind DINOv3 on subclass structure (R2, R5). SSCD is the negative reference: a copy-detection encoder has almost no make or subclass structure (kNN make 0.24, R3 ratios near 1), as it should.
- **The subclass problem is not solved by a frozen encoder.** Even DINOv3-L recovers subclasses only moderately: ACC 0.80 with the true K and 0.71 with K̂. R4 var. ratio 0.62 means most within-make variance is not explained by the subclass (CLIP 0.81).
- Eq. (1) (Δ0 < Δ1) holds for 100% of makes in every encoder; Eq. (2) (Δ1 < Δ2) for 87-97%.
- R4 effective rank is not comparable across encoders of different dimension (SSCD 512, DINOv3-B 768, DINOv3-L / CLIP 1024).
- `ari_seeds` is averaged over only 3 seed pairs per make, so its CI is not informative (see `r5.csv`).

## Known problems
1. **K̂ bootstrap CIs were invalid (fixed, verified on the rerun).** Resamples contain duplicate images; duplicates form tight pairs and inflate the silhouette-selected K̂. Example (DINOv3-L 256): macro K̂ = 5.43 with CI [12.97, 16.33]; CLIP-L 128 `acc_khat` 0.56 with CI [0.29, 0.33]. Fix: K̂ metrics now take their CIs from draws of 63.2% of each make's images without replacement (`khat_subsample` in `configs/eval_representation.yaml`, table `r2_khat.csv`); oracle-K, R1, R3 and R4 keep the standard bootstrap. The point estimates did not change, and the new K̂ CIs contain them (one exception: `sscd_128` `acc_khat`, 0.316 vs CI [0.283, 0.312], a near-structureless space where subsampling is slightly biased).
2. **K̂ never returns 1 for single-subclass makes: kept as a finding, not a bug.** `khat_error` on the 19 K_c = 1 makes has a minimum of 1 for every DINOv3 and CLIP run (mean 1.2-1.6). A check on `train` embeddings (not test) shows no rule fixes this across encoders: in DINOv3 space the best silhouette of K_c = 1 makes (median 0.24-0.27) is about that of K_c >= 2 makes (0.28-0.29), and the gap statistic alone returns 1 for only 4-7 of the 19. A threshold that fixes CLIP (SIL_MIN = 0.25) underestimates K for K_c >= 2 makes by 1.8 and does nothing for DINOv3. Tuning it per encoder would tune the metric to the encoders, so the fixed rule stays (SIL_MIN = 0.10). Likely explanation (untested): real viewpoint/colour structure inside one official class, which is exactly the over-fragmentation a label-free per-make K̂ pipeline would pass on to a generator. Details: docs/METRICS.md.

## Not yet run
- Inception features (`slurm/extract_features.sh`) for G1; no generated images exist yet, so `eval_generation.py` has not been run (Milestone 5 controls produce them).
- DINOv2 is deliberately not used for now.
