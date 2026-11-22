# Generation: cluster_dinov3_base (cifar100)

Provenance: git `870e9774f24c1337c42f2618e65a60f0a7e3bd59-dirty`, config `configs/gen_cluster_dinov3_base_cifar100.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 68, 'val': 5, 'test': 17} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric            |     mean |       std |
|:------------------|---------:|----------:|
| G1 FID            | 27.22    | 0.2592    |
| G1 KID            |  0.01354 | 0.0003251 |
| G1 recall         |  0.6133  | 0.01159   |
| G2 make acc       |  0.7427  | 0.003892  |
| G3 TV             |  0.0791  | 0.002988  |
| G3 KL             |  0.02367 | 0.0004817 |
| G4 covered        |  1       | 0         |
| G4 rare covered   |  1       | 0         |
| G5 excess KID     |  0.8741  | 0.01013   |
| G5 coverage ratio |  0.5145  | 0.00199   |
| G7 copies         |  0.5541  | 0.004366  |
| G7 copies, real   |  0.7167  | 0         |
| G6 matched KID    |  0.9287  | 0.006683  |
| G6 purity         |  0.7752  | 0.006105  |
| G6 unmatched real |  0       | 0         |

## Per seed

|   seed |    step |   train_hours |   ddim_steps |   guidance |   n_labels |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |   G6 matched KID |   G6 purity |   G6 unmatched real |
|-------:|--------:|--------------:|-------------:|-----------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|-----------------:|------------:|--------------------:|
|      0 | 1.5e+05 |         1.312 |           25 |        1.5 |        100 |    27.5  |  0.01374 |      0.6207 |        0.743  | 0.07663 | 0.02344 |            1 |                 1 |          0.8858 |              0.5166 |      0.5558 |            0.7167 |           0.9358 |      0.7778 |                   0 |
|      1 | 1.5e+05 |         1.348 |           25 |        1.5 |        100 |    26.98 |  0.01316 |      0.6193 |        0.7386 | 0.08242 | 0.02422 |            1 |                 1 |          0.8676 |              0.5126 |      0.5573 |            0.7167 |           0.9226 |      0.7682 |                   0 |
|      2 | 1.5e+05 |         1.371 |           25 |        1.5 |        100 |    27.2  |  0.01371 |      0.5999 |        0.7464 | 0.07825 | 0.02334 |            1 |                 1 |          0.869  |              0.5143 |      0.5491 |            0.7167 |           0.9276 |      0.7796 |                   0 |

Figures (seed 0): [samples.png](cluster_dinov3_base_s0_step150000_ddim25_A/samples.png), [nearest_real.png](cluster_dinov3_base_s0_step150000_ddim25_A/nearest_real.png), [embedding.png](cluster_dinov3_base_s0_step150000_ddim25_A/embedding.png), [g7_nn_grid.png](cluster_dinov3_base_s0_step150000_ddim25_A/g7_nn_grid.png)
