# Generation: coarse (cifar100)

Provenance: git `3349bdea58938af7465de8f1872e12690cd24a72-dirty`, config `configs/gen_coarse_cifar100.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 68, 'val': 5, 'test': 17} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric            |     mean |       std |
|:------------------|---------:|----------:|
| G1 FID            | 28.81    | 0.2644    |
| G1 KID            |  0.0147  | 0.0002945 |
| G1 recall         |  0.5736  | 0.005015  |
| G2 make acc       |  0.7087  | 0.005348  |
| G3 TV             |  0.1342  | 0.002191  |
| G3 KL             |  0.05777 | 0.002153  |
| G4 covered        |  0.98    | 0.01      |
| G4 rare covered   |  0.98    | 0.01      |
| G5 excess KID     |  0.9427  | 0.009819  |
| G5 coverage ratio |  0.4708  | 0.00649   |
| G7 copies         |  0.5414  | 0.001973  |
| G7 copies, real   |  0.7167  | 0         |

## Per seed

|   seed |    step |   train_hours |   ddim_steps |   guidance |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |
|-------:|--------:|--------------:|-------------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|
|      0 | 1.5e+05 |         1.219 |           25 |        1.5 |    28.54 |  0.01446 |      0.5792 |        0.7141 |  0.1353 | 0.05965 |         0.98 |              0.98 |          0.9329 |              0.4777 |      0.5437 |            0.7167 |
|      1 | 1.5e+05 |         1.196 |           25 |        1.5 |    28.81 |  0.01461 |      0.5695 |        0.7034 |  0.1317 | 0.05542 |         0.99 |              0.99 |          0.9426 |              0.4701 |      0.5405 |            0.7167 |
|      2 | 1.5e+05 |         1.165 |           25 |        1.5 |    29.06 |  0.01503 |      0.5722 |        0.7087 |  0.1357 | 0.05824 |         0.97 |              0.97 |          0.9526 |              0.4647 |      0.5401 |            0.7167 |

Figures (seed 0): [samples.png](coarse_s0_step150000_ddim25_A/samples.png), [nearest_real.png](coarse_s0_step150000_ddim25_A/nearest_real.png), [embedding.png](coarse_s0_step150000_ddim25_A/embedding.png), [g7_nn_grid.png](coarse_s0_step150000_ddim25_A/g7_nn_grid.png)
