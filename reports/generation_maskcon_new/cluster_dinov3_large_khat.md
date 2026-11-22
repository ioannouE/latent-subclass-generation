# Generation: cluster_dinov3_large_khat (maskcon_new)

Provenance: git `3349bdea58938af7465de8f1872e12690cd24a72-dirty`, config `configs/gen_cluster_dinov3_large_khat_maskcon_new.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 98, 'val': 12, 'test': 101} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric            |     mean |       std |
|:------------------|---------:|----------:|
| G1 FID            | 21.13    | 0.2546    |
| G1 KID            |  0.01444 | 0.0002068 |
| G1 recall         |  0.4826  | 0.009486  |
| G2 make acc       |  0.728   | 0.007726  |
| G3 TV             |  0.1493  | 0.008261  |
| G3 KL             |  0.07879 | 0.006867  |
| G4 covered        |  0.9451  | 0.007441  |
| G4 rare covered   |  0.8925  | 0.02872   |
| G5 excess KID     |  2.807   | 0.01843   |
| G5 coverage ratio |  0.01555 | 0.002803  |
| G7 copies         |  0.1519  | 0.01171   |
| G7 copies, real   |  0.1297  | 0         |
| G6 matched KID    |  2.918   | 0.01765   |
| G6 purity         |  0.6196  | 0.02751   |
| G6 unmatched real |  2.417   | 0.5907    |

## Per seed

|   seed |   step |   train_hours |   ddim_steps |   guidance |   n_labels |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |   G6 matched KID |   G6 purity |   G6 unmatched real |
|-------:|-------:|--------------:|-------------:|-----------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|-----------------:|------------:|--------------------:|
|      0 |  5e+04 |        0.4012 |           25 |        1.5 |        192 |    20.91 |  0.01436 |      0.4746 |        0.7248 |  0.1475 | 0.07483 |       0.9389 |            0.8594 |           2.801 |             0.01687 |      0.1414 |            0.1297 |            2.932 |      0.6065 |               2.875 |
|      1 |  5e+04 |        0.3997 |           25 |        1.5 |        199 |    21.4  |  0.01468 |      0.4802 |        0.7368 |  0.1583 | 0.08672 |       0.9533 |            0.9089 |           2.792 |             0.01234 |      0.1497 |            0.1297 |            2.898 |      0.6011 |               2.625 |
|      2 |  5e+04 |        0.4012 |           25 |        1.5 |        211 |    21.07 |  0.01429 |      0.4931 |        0.7224 |  0.142  | 0.07482 |       0.943  |            0.9094 |           2.827 |             0.01746 |      0.1645 |            0.1297 |            2.923 |      0.6512 |               1.75  |

Figures (seed 0): [samples.png](cluster_dinov3_large_khat_s0_step050000_ddim25_A/samples.png), [nearest_real.png](cluster_dinov3_large_khat_s0_step050000_ddim25_A/nearest_real.png), [embedding.png](cluster_dinov3_large_khat_s0_step050000_ddim25_A/embedding.png), [g7_nn_grid.png](cluster_dinov3_large_khat_s0_step050000_ddim25_A/g7_nn_grid.png)
