# Generation: cluster_dinov3_base_khat (cifar100)

Provenance: git `870e9774f24c1337c42f2618e65a60f0a7e3bd59`, config `configs/gen_cluster_dinov3_base_khat_cifar100.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 68, 'val': 5, 'test': 17} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric            |     mean |       std |
|:------------------|---------:|----------:|
| G1 FID            | 27.06    | 0.2247    |
| G1 KID            |  0.01338 | 0.0003111 |
| G1 recall         |  0.5999  | 0.001402  |
| G2 make acc       |  0.7436  | 0.001618  |
| G3 TV             |  0.07707 | 0.001193  |
| G3 KL             |  0.02241 | 0.0007798 |
| G4 covered        |  0.9967  | 0.005774  |
| G4 rare covered   |  0.9967  | 0.005774  |
| G5 excess KID     |  0.8773  | 0.008895  |
| G5 coverage ratio |  0.5159  | 0.01035   |
| G7 copies         |  0.5577  | 0.00318   |
| G7 copies, real   |  0.7167  | 0         |
| G6 matched KID    |  0.895   | 0.004208  |
| G6 purity         |  0.7467  | 0.0052    |
| G6 unmatched real |  0.4833  | 0.02887   |

## Per seed

|   seed |    step |   train_hours |   ddim_steps |   guidance |   n_labels |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |   G6 matched KID |   G6 purity |   G6 unmatched real |
|-------:|--------:|--------------:|-------------:|-----------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|-----------------:|------------:|--------------------:|
|      0 | 1.5e+05 |         1.211 |           25 |        1.5 |        107 |    26.8  |  0.01313 |      0.5985 |        0.742  | 0.07573 | 0.02154 |         0.99 |              0.99 |          0.8672 |              0.5255 |      0.5582 |            0.7167 |           0.8925 |      0.7517 |                0.45 |
|      1 | 1.5e+05 |         1.212 |           25 |        1.5 |        109 |    27.19 |  0.01329 |      0.5999 |        0.7452 | 0.07743 | 0.02265 |         1    |              1    |          0.8807 |              0.5172 |      0.5606 |            0.7167 |           0.8998 |      0.7413 |                0.5  |
|      2 | 1.5e+05 |         1.205 |           25 |        1.5 |        118 |    27.19 |  0.01373 |      0.6013 |        0.7438 | 0.07803 | 0.02305 |         1    |              1    |          0.884  |              0.505  |      0.5543 |            0.7167 |           0.8926 |      0.747  |                0.5  |

Figures (seed 0): [samples.png](cluster_dinov3_base_khat_s0_step150000_ddim25_A/samples.png), [nearest_real.png](cluster_dinov3_base_khat_s0_step150000_ddim25_A/nearest_real.png), [embedding.png](cluster_dinov3_base_khat_s0_step150000_ddim25_A/embedding.png), [g7_nn_grid.png](cluster_dinov3_base_khat_s0_step150000_ddim25_A/g7_nn_grid.png)
