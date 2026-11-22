# Generation: coarse (maskcon_new)

Provenance: git `3349bdea58938af7465de8f1872e12690cd24a72-dirty`, config `configs/gen_coarse_maskcon_new.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 98, 'val': 12, 'test': 101} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric            |      mean |       std |
|:------------------|----------:|----------:|
| G1 FID            | 22.54     | 0.3998    |
| G1 KID            |  0.01577  | 0.0002321 |
| G1 recall         |  0.4194   | 0.006357  |
| G2 make acc       |  0.7202   | 0.01094   |
| G3 TV             |  0.2653   | 0.003886  |
| G3 KL             |  0.2715   | 0.00502   |
| G4 covered        |  0.6841   | 0.0272    |
| G4 rare covered   |  0.5792   | 0.02374   |
| G5 excess KID     |  3.09     | 0.001603  |
| G5 coverage ratio |  0.008811 | 0.001259  |
| G7 copies         |  0.03067  | 0.002682  |
| G7 copies, real   |  0.1297   | 0         |

## Per seed

|   seed |   step |   train_hours |   ddim_steps |   guidance |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |
|-------:|-------:|--------------:|-------------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|
|      0 |  5e+04 |        0.4106 |           25 |        1.5 |    22.13 |  0.01557 |      0.4262 |        0.7142 |  0.2692 |  0.2773 |       0.7134 |            0.6028 |           3.09  |            0.00761  |     0.02937 |            0.1297 |
|      1 |  5e+04 |        0.4018 |           25 |        1.5 |    22.93 |  0.01603 |      0.4136 |        0.7137 |  0.2614 |  0.2687 |       0.6596 |            0.5554 |           3.089 |            0.01012  |     0.02888 |            0.1297 |
|      2 |  5e+04 |        0.4018 |           25 |        1.5 |    22.54 |  0.01572 |      0.4184 |        0.7329 |  0.2654 |  0.2686 |       0.6792 |            0.5793 |           3.092 |            0.008703 |     0.03375 |            0.1297 |

Figures (seed 0): [samples.png](coarse_s0_step050000_ddim25_A/samples.png), [nearest_real.png](coarse_s0_step050000_ddim25_A/nearest_real.png), [embedding.png](coarse_s0_step050000_ddim25_A/embedding.png), [g7_nn_grid.png](coarse_s0_step050000_ddim25_A/g7_nn_grid.png)
