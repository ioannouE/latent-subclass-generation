# Generation: cluster_dinov3_large (maskcon_new)

Provenance: git `3349bdea58938af7465de8f1872e12690cd24a72-dirty`, config `configs/gen_cluster_dinov3_large_maskcon_new.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 98, 'val': 12, 'test': 101} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric            |     mean |       std |
|:------------------|---------:|----------:|
| G1 FID            | 21.18    | 0.3262    |
| G1 KID            |  0.01447 | 0.0003085 |
| G1 recall         |  0.4804  | 0.009546  |
| G2 make acc       |  0.7193  | 0.009117  |
| G3 TV             |  0.1478  | 0.001233  |
| G3 KL             |  0.0852  | 0.005723  |
| G4 covered        |  0.9548  | 0.01687   |
| G4 rare covered   |  0.9132  | 0.02215   |
| G5 excess KID     |  2.803   | 0.03431   |
| G5 coverage ratio |  0.01574 | 0.002523  |
| G7 copies         |  0.1623  | 0.02006   |
| G7 copies, real   |  0.1297  | 0         |
| G6 matched KID    |  3.034   | 0.02148   |
| G6 purity         |  0.6345  | 0.004416  |
| G6 unmatched real |  0.04167 | 0.07217   |

## Per seed

|   seed |   step |   train_hours |   ddim_steps |   guidance |   n_labels |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |   G6 matched KID |   G6 purity |   G6 unmatched real |
|-------:|-------:|--------------:|-------------:|-----------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|-----------------:|------------:|--------------------:|
|      0 |  5e+04 |        0.4026 |           25 |        1.5 |        196 |    21.41 |  0.01451 |      0.4696 |        0.7104 |  0.1464 | 0.08298 |       0.9431 |            0.888  |           2.804 |             0.01659 |      0.1854 |            0.1297 |            3.034 |      0.6342 |               0     |
|      1 |  5e+04 |        0.4007 |           25 |        1.5 |        196 |    21.33 |  0.01475 |      0.4878 |        0.7188 |  0.1488 | 0.0917  |       0.9473 |            0.9219 |           2.837 |             0.0129  |      0.1524 |            0.1297 |            3.056 |      0.6303 |               0.125 |
|      2 |  5e+04 |        0.4007 |           25 |        1.5 |        196 |    20.81 |  0.01414 |      0.4839 |        0.7287 |  0.1481 | 0.08091 |       0.9742 |            0.9297 |           2.768 |             0.01773 |      0.1491 |            0.1297 |            3.013 |      0.6391 |               0     |

Figures (seed 0): [samples.png](cluster_dinov3_large_s0_step050000_ddim25_A/samples.png), [nearest_real.png](cluster_dinov3_large_s0_step050000_ddim25_A/nearest_real.png), [embedding.png](cluster_dinov3_large_s0_step050000_ddim25_A/embedding.png), [g7_nn_grid.png](cluster_dinov3_large_s0_step050000_ddim25_A/g7_nn_grid.png)
