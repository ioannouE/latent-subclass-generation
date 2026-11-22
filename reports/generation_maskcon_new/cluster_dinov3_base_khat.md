# Generation: cluster_dinov3_base_khat (maskcon_new)

Provenance: git `870e9774f24c1337c42f2618e65a60f0a7e3bd59`, config `configs/gen_cluster_dinov3_base_khat_maskcon_new.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 98, 'val': 12, 'test': 101} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric            |     mean |       std |
|:------------------|---------:|----------:|
| G1 FID            | 21.16    | 0.2985    |
| G1 KID            |  0.01449 | 0.0001534 |
| G1 recall         |  0.4791  | 0.002547  |
| G2 make acc       |  0.7184  | 0.00321   |
| G3 TV             |  0.1431  | 0.002735  |
| G3 KL             |  0.07923 | 0.006246  |
| G4 covered        |  0.9534  | 0.001001  |
| G4 rare covered   |  0.9306  | 0.02962   |
| G5 excess KID     |  2.808   | 0.04244   |
| G5 coverage ratio |  0.02058 | 0.005289  |
| G7 copies         |  0.1653  | 0.0134    |
| G7 copies, real   |  0.1297  | 0         |
| G6 matched KID    |  2.978   | 0.02723   |
| G6 purity         |  0.6223  | 0.001636  |
| G6 unmatched real |  0.75    | 0.875     |

## Per seed

|   seed |   step |   train_hours |   ddim_steps |   guidance |   n_labels |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |   G6 matched KID |   G6 purity |   G6 unmatched real |
|-------:|-------:|--------------:|-------------:|-----------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|-----------------:|------------:|--------------------:|
|      0 |  5e+04 |        0.4292 |           25 |        1.5 |        229 |    21.42 |  0.01463 |      0.4781 |        0.7194 |  0.1432 | 0.08536 |       0.9537 |            0.9062 |           2.83  |             0.01572 |      0.1706 |            0.1297 |            2.962 |      0.6205 |               1.75  |
|      1 |  5e+04 |        0.4287 |           25 |        1.5 |        221 |    21.23 |  0.01451 |      0.4772 |        0.721  |  0.1458 | 0.07945 |       0.9543 |            0.9635 |           2.835 |             0.0198  |      0.15   |            0.1297 |            3.01  |      0.6236 |               0.125 |
|      2 |  5e+04 |        0.431  |           25 |        1.5 |        235 |    20.84 |  0.01433 |      0.482  |        0.7148 |  0.1403 | 0.07287 |       0.9523 |            0.9219 |           2.76  |             0.02621 |      0.1751 |            0.1297 |            2.963 |      0.6228 |               0.375 |

Figures (seed 0): [samples.png](cluster_dinov3_base_khat_s0_step050000_ddim25_A/samples.png), [nearest_real.png](cluster_dinov3_base_khat_s0_step050000_ddim25_A/nearest_real.png), [embedding.png](cluster_dinov3_base_khat_s0_step050000_ddim25_A/embedding.png), [g7_nn_grid.png](cluster_dinov3_base_khat_s0_step050000_ddim25_A/g7_nn_grid.png)
