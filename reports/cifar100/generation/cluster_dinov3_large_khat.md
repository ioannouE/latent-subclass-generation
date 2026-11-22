# Generation: cluster_dinov3_large_khat (cifar100)

Provenance: git `3349bdea58938af7465de8f1872e12690cd24a72-dirty`, config `configs/gen_cluster_dinov3_large_khat_cifar100.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 68, 'val': 5, 'test': 17} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric            |     mean |       std |
|:------------------|---------:|----------:|
| G1 FID            | 27.21    | 0.1874    |
| G1 KID            |  0.01335 | 0.0002124 |
| G1 recall         |  0.6025  | 0.002764  |
| G2 make acc       |  0.7408  | 0.002339  |
| G3 TV             |  0.07691 | 0.002191  |
| G3 KL             |  0.0226  | 0.0007334 |
| G4 covered        |  0.9967  | 0.005774  |
| G4 rare covered   |  0.9967  | 0.005774  |
| G5 excess KID     |  0.8793  | 0.006542  |
| G5 coverage ratio |  0.5136  | 0.01218   |
| G7 copies         |  0.5586  | 0.007047  |
| G7 copies, real   |  0.7167  | 0         |
| G6 matched KID    |  0.8866  | 0.00494   |
| G6 purity         |  0.7518  | 0.002669  |
| G6 unmatched real |  0.45    | 0         |

## Per seed

|   seed |    step |   train_hours |   ddim_steps |   guidance |   n_labels |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |   G6 matched KID |   G6 purity |   G6 unmatched real |
|-------:|--------:|--------------:|-------------:|-----------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|-----------------:|------------:|--------------------:|
|      0 | 1.5e+05 |         1.176 |           25 |        1.5 |        143 |    27.34 |  0.01341 |      0.6004 |        0.7418 | 0.07848 | 0.02316 |         1    |              1    |          0.8792 |              0.5252 |      0.5547 |            0.7167 |           0.8921 |      0.7491 |                0.45 |
|      1 | 1.5e+05 |         1.169 |           25 |        1.5 |        141 |    27.28 |  0.01353 |      0.6056 |        0.7424 | 0.07441 | 0.02177 |         1    |              1    |          0.8858 |              0.5009 |      0.5543 |            0.7167 |           0.8827 |      0.7544 |                0.45 |
|      2 | 1.5e+05 |         1.168 |           25 |        1.5 |        145 |    26.99 |  0.01312 |      0.6014 |        0.7381 | 0.07785 | 0.02288 |         0.99 |              0.99 |          0.8727 |              0.5146 |      0.5667 |            0.7167 |           0.8849 |      0.752  |                0.45 |

Figures (seed 0): [samples.png](cluster_dinov3_large_khat_s0_step150000_ddim25_A/samples.png), [nearest_real.png](cluster_dinov3_large_khat_s0_step150000_ddim25_A/nearest_real.png), [embedding.png](cluster_dinov3_large_khat_s0_step150000_ddim25_A/embedding.png), [g7_nn_grid.png](cluster_dinov3_large_khat_s0_step150000_ddim25_A/g7_nn_grid.png)
