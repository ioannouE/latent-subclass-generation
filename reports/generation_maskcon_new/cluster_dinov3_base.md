# Generation: cluster_dinov3_base (maskcon_new)

Provenance: git `870e9774f24c1337c42f2618e65a60f0a7e3bd59`, config `configs/gen_cluster_dinov3_base_maskcon_new.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 98, 'val': 12, 'test': 101} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric            |     mean |       std |
|:------------------|---------:|----------:|
| G1 FID            | 20.85    | 0.6696    |
| G1 KID            |  0.01423 | 0.0004488 |
| G1 recall         |  0.4841  | 0.008269  |
| G2 make acc       |  0.7276  | 0.005796  |
| G3 TV             |  0.1509  | 0.003982  |
| G3 KL             |  0.08476 | 0.005626  |
| G4 covered        |  0.9447  | 0.006848  |
| G4 rare covered   |  0.9264  | 0.05291   |
| G5 excess KID     |  2.829   | 0.01706   |
| G5 coverage ratio |  0.01624 | 0.0003337 |
| G7 copies         |  0.1535  | 0.00905   |
| G7 copies, real   |  0.1297  | 0         |
| G6 matched KID    |  3.035   | 0.008022  |
| G6 purity         |  0.6114  | 0.005928  |
| G6 unmatched real |  0       | 0         |

## Per seed

|   seed |   step |   train_hours |   ddim_steps |   guidance |   n_labels |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |   G6 matched KID |   G6 purity |   G6 unmatched real |
|-------:|-------:|--------------:|-------------:|-----------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|-----------------:|------------:|--------------------:|
|      0 |  5e+04 |        0.447  |           25 |        1.5 |        196 |    21.61 |  0.0147  |      0.4929 |        0.7309 |  0.1503 | 0.08624 |       0.9525 |            0.8807 |           2.816 |             0.01595 |      0.1636 |            0.1297 |            3.044 |      0.6108 |                   0 |
|      1 |  5e+04 |        0.4565 |           25 |        1.5 |        196 |    20.6  |  0.01419 |      0.4827 |        0.7311 |  0.1552 | 0.08949 |       0.9394 |            0.9141 |           2.822 |             0.01618 |      0.1509 |            0.1297 |            3.03  |      0.6057 |                   0 |
|      2 |  5e+04 |        0.4509 |           25 |        1.5 |        196 |    20.34 |  0.01381 |      0.4766 |        0.721  |  0.1473 | 0.07854 |       0.9424 |            0.9844 |           2.848 |             0.01661 |      0.1461 |            0.1297 |            3.031 |      0.6175 |                   0 |

Figures (seed 0): [samples.png](cluster_dinov3_base_s0_step050000_ddim25_A/samples.png), [nearest_real.png](cluster_dinov3_base_s0_step050000_ddim25_A/nearest_real.png), [embedding.png](cluster_dinov3_base_s0_step050000_ddim25_A/embedding.png), [g7_nn_grid.png](cluster_dinov3_base_s0_step050000_ddim25_A/g7_nn_grid.png)
