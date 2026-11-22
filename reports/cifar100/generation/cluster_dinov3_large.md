# Generation: cluster_dinov3_large (cifar100)

Provenance: git `3349bdea58938af7465de8f1872e12690cd24a72-dirty`, config `configs/gen_cluster_dinov3_large_cifar100.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 68, 'val': 5, 'test': 17} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric            |     mean |       std |
|:------------------|---------:|----------:|
| G1 FID            | 27.29    | 0.1077    |
| G1 KID            |  0.01352 | 9.153e-05 |
| G1 recall         |  0.6014  | 0.004604  |
| G2 make acc       |  0.7408  | 0.004334  |
| G3 TV             |  0.07955 | 0.003625  |
| G3 KL             |  0.02177 | 0.00155   |
| G4 covered        |  1       | 0         |
| G4 rare covered   |  1       | 0         |
| G5 excess KID     |  0.8815  | 0.01085   |
| G5 coverage ratio |  0.5121  | 0.007782  |
| G7 copies         |  0.5527  | 0.001692  |
| G7 copies, real   |  0.7167  | 0         |
| G6 matched KID    |  0.9245  | 0.001122  |
| G6 purity         |  0.7868  | 0.005643  |
| G6 unmatched real |  0       | 0         |

## Per seed

|   seed |    step |   train_hours |   ddim_steps |   guidance |   n_labels |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |   G6 matched KID |   G6 purity |   G6 unmatched real |
|-------:|--------:|--------------:|-------------:|-----------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|-----------------:|------------:|--------------------:|
|      0 | 1.5e+05 |         1.178 |           25 |        1.5 |        100 |    27.18 |  0.01347 |      0.605  |        0.7456 | 0.07567 | 0.02036 |            1 |                 1 |          0.894  |              0.5037 |      0.5508 |            0.7167 |           0.9257 |      0.7912 |                   0 |
|      1 | 1.5e+05 |         1.168 |           25 |        1.5 |        100 |    27.31 |  0.01346 |      0.5962 |        0.7371 | 0.08286 | 0.02343 |            1 |                 1 |          0.8761 |              0.5191 |      0.5541 |            0.7167 |           0.9234 |      0.7804 |                   0 |
|      2 | 1.5e+05 |         1.168 |           25 |        1.5 |        100 |    27.39 |  0.01362 |      0.6029 |        0.7397 | 0.08012 | 0.02151 |            1 |                 1 |          0.8744 |              0.5135 |      0.5531 |            0.7167 |           0.9244 |      0.7887 |                   0 |

Figures (seed 0): [samples.png](cluster_dinov3_large_s0_step150000_ddim25_A/samples.png), [nearest_real.png](cluster_dinov3_large_s0_step150000_ddim25_A/nearest_real.png), [embedding.png](cluster_dinov3_large_s0_step150000_ddim25_A/embedding.png), [g7_nn_grid.png](cluster_dinov3_large_s0_step150000_ddim25_A/g7_nn_grid.png)
