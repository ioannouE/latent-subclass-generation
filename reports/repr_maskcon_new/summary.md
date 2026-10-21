# Representation metrics on the maskcon_new label scheme: frozen encoders, fine-tuned baselines, geosub

Provenance: seed 0 everywhere, dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`; per-run git hash and config in each
`<encoder>_<crop>/summary.json`. Configs: `configs/eval_repr_maskcon_new.yaml` (frozen), `configs/eval_repr_maskcon_new_baselines.yaml` (baselines, trained with
`configs/repr_*_maskcon_new.yaml`), `configs/eval_repr_maskcon_geosub.yaml` (geosub, `configs/geosub_maskcon*.yaml`).
Label scheme `maskcon_new` (configs/labels.yaml): coarse = 8 body types, hidden subclass = the 196 official classes (true K_c 6-50, mean 24.5).
Fit/reference = `train` (7,231 images), scored on `test` (7,940). **Conflicting images are excluded everywhere** (`data/splits/exclude_maskcon_new.txt`,
211 images: {'train': 98, 'val': 12, 'test': 101}; docs/DATA.md). Metric definitions: docs/METRICS.md. Values are macro means over the 8 body types;
per-class tables and CIs are in `<encoder>_<crop>/*.csv`. All backbones are fine-tuned from / frozen at DINOv3-B unless marked L.

## R1-R5 (k-means on the embedding for R2 / R5, the same pipeline for every row)
| encoder | kNN coarse | fine probe | Recall@1 | ACC oracle K | NMI oracle K | ACC K-hat | ARI K-hat | K-hat | d0/d1 | var_ratio | ARI boot |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen DINOv3-B | 0.932 | 0.950 | 0.934 | 0.727 | 0.830 | 0.659 | 0.601 | 20.4 | 0.507 | 0.521 | 0.829 |
| frozen DINOv3-L | 0.944 | 0.950 | 0.935 | **0.800** | 0.886 | 0.660 | 0.641 | 17.0 | 0.415 | 0.429 | 0.874 |
| frozen CLIP-L | 0.890 | 0.912 | 0.823 | 0.674 | 0.771 | 0.510 | 0.483 | 13.8 | 0.668 | 0.678 | 0.701 |
| SupCon | 0.967 | 0.952 | 0.929 | 0.728 | 0.830 | 0.656 | 0.609 | 19.8 | 0.551 | 0.565 | 0.814 |
| SimCLR | 0.826 | 0.926 | 0.828 | 0.480 | 0.606 | 0.409 | 0.325 | 19.1 | 0.712 | 0.723 | 0.582 |
| SupCon + SimCLR | 0.928 | 0.942 | 0.893 | 0.664 | 0.766 | 0.553 | 0.489 | 19.4 | 0.677 | 0.689 | 0.706 |
| MaskCon | 0.954 | 0.950 | 0.928 | 0.692 | 0.798 | 0.575 | 0.521 | 20.3 | 0.558 | 0.572 | 0.796 |
| FALCON (uses K) | 0.963 | 0.950 | 0.928 | 0.702 | 0.810 | 0.613 | 0.549 | 20.0 | 0.595 | 0.609 | 0.785 |
| GEORGE | 0.973 | 0.947 | 0.911 | 0.690 | 0.804 | 0.597 | 0.542 | 21.1 | 0.593 | 0.606 | 0.764 |
| GEORGE stage 1 (ERM) | 0.972 | 0.944 | 0.907 | 0.696 | 0.806 | 0.594 | 0.542 | 20.0 | 0.599 | 0.612 | 0.770 |
| NHP (z_sub) | 0.972 | 0.873 | 0.853 | 0.469 | 0.632 | 0.393 | 0.280 | 10.0 | 0.440 | 0.454 | 0.749 |
| NHP (z_make + z_sub) | **0.983** | 0.867 | 0.861 | 0.462 | 0.629 | 0.396 | 0.284 | 10.6 | 0.463 | 0.477 | 0.758 |
| LeJEPA (no labels) | 0.916 | 0.946 | 0.923 | 0.688 | 0.793 | 0.549 | 0.494 | 20.4 | 0.553 | 0.567 | 0.779 |
| geosub `geo` (B; sanity check) | 0.932 | 0.950 | 0.934 | 0.730 | 0.830 | 0.659 | 0.601 | 20.4 | 0.507 | 0.521 | 0.831 |
| geosub `geo_sub` (B) | 0.937 | 0.951 | 0.932 | 0.734 | 0.832 | 0.603 | 0.556 | 19.8 | 0.498 | 0.512 | 0.839 |
| geosub `full` (B) | 0.940 | **0.953** | 0.931 | 0.724 | 0.833 | 0.632 | 0.582 | 20.3 | 0.561 | 0.574 | 0.847 |
| geosub `geo` (L; sanity check) | 0.944 | 0.950 | 0.935 | 0.800 | 0.886 | 0.660 | 0.641 | 17.0 | 0.415 | 0.429 | 0.874 |
| geosub `geo_sub` (L) | 0.947 | 0.949 | 0.934 | 0.795 | **0.889** | **0.724** | **0.696** | 17.5 | 0.402 | 0.416 | **0.896** |
| geosub `full` (L) | 0.953 | 0.949 | 0.934 | 0.799 | 0.887 | 0.704 | 0.686 | 17.9 | 0.486 | 0.500 | 0.889 |
| frozen SSCD | 0.412 | 0.581 | 0.441 | 0.206 | 0.216 | 0.119 | 0.021 | 5.1 | 0.961 | 0.962 | 0.278 |

geosub `full` = L_geo + L_sub + L_supcon (0.1) + L_var (1); `geo_sub` = L_geo + L_sub (the val choice of the L_supcon sweep). Sweep rows (B, L_supcon 0.1 / 0.3 /
1 / 3) are in `geosub_geo_sub_supcon*_bodytype_new_s0_256/`.

## geosub's own prototype clusters on test (own K-hat), against k-means on the frozen backbone of the same images (`*_clusters_test/`)
| | ACC | ARI | flip-ARI | K-hat / class (true 24.5) |
|:--|--:|--:|--:|--:|
| k-means on frozen DINOv3-B, label-free K-hat | 0.659 | 0.601 | - | 20.4 |
| geosub `geo_sub` (B) / `full` (B) | 0.645 / 0.649 | 0.580 / 0.577 | 0.984 / 0.976 | 22.2 / 22.4 |
| k-means on frozen DINOv3-L, label-free K-hat | 0.660 | 0.641 | - | 17.0 |
| geosub `geo_sub` (L) / `full` (L) | **0.713** / 0.710 | **0.663** / 0.663 | 0.990 / 0.987 | 22.0 / 22.2 |

## Reading (one seed; differences of about 0.02 are within noise)
- **No fine-tuned baseline beats frozen DINOv3 on subclass recovery.** They raise coarse kNN (0.95-0.98) and keep the fine probe (0.94-0.95, NHP 0.87),
  but lower oracle-K and K-hat ACC. SupCon is the closest (0.656 vs 0.659 K-hat ACC).
- **geosub on DINOv3-B does not help**: K-hat ACC 0.603-0.632 (k-means) and 0.645-0.649 (prototypes) vs 0.659.
- **geosub on DINOv3-L is the best label-free row**: K-hat ACC 0.724 (k-means on `geo_sub` z) and 0.713 (prototypes) vs 0.660 for frozen DINOv3-L, also
  the highest ARI K-hat and ARI boot. Most of it is one class: on Cab (18 classes) the frozen K-hat rule picks K = 4 (ACC 0.24), on `geo_sub` z it picks 11
  (0.60). Without Cab, k-means on `geo_sub` z is still ahead (0.741 vs 0.720 over the other 7: SUV, Convertible, Coupe, Hatchback +0.03 to +0.06, Sedan -0.04);
  the prototype clusters are level (0.717 vs 0.720).
- **Oracle-K ACC does not move** (0.795 vs 0.800): geosub does not make the subclasses more separable when K is known. Its effect is that label-free
  model selection (silhouette K-hat, or its own prototype count) fails less often.
- L_supcon and L_var (`full`) add nothing over `geo_sub` on the subclass metrics; they raise coarse kNN and var_ratio slightly.
