# Generation: oracle (cifar100)

Provenance: git `3349bdea58938af7465de8f1872e12690cd24a72-dirty`, config `configs/gen_oracle_cifar100.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 68, 'val': 5, 'test': 17} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric                  |     mean |       std |
|:------------------------|---------:|----------:|
| G1 FID                  | 27.39    | 0.1981    |
| G1 KID                  |  0.0136  | 0.0002213 |
| G1 recall               |  0.5984  | 0.002727  |
| G2 make acc             |  0.7409  | 0.001402  |
| G3 TV                   |  0.07624 | 0.001238  |
| G3 KL                   |  0.02048 | 0.0009543 |
| G4 covered              |  1       | 0         |
| G4 rare covered         |  1       | 0         |
| G5 excess KID           |  0.8793  | 0.0054    |
| G5 coverage ratio       |  0.5117  | 0.009122  |
| G7 copies               |  0.5525  | 0.0006658 |
| G7 copies, real         |  0.7167  | 0         |
| G6 matched KID          |  0.9163  | 0.001612  |
| G6 purity               |  0.7977  | 0.003618  |
| G6 unmatched real       |  0       | 0         |
| E1 fine acc (generated) |  0.6357  | 0.001966  |
| E1 fine acc (real val)  |  0.8917  | 0         |

## Per seed

|   seed |    step |   train_hours |   ddim_steps |   guidance |   n_labels |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |   G6 matched KID |   G6 purity |   G6 unmatched real |   E1 fine acc (generated) |   E1 fine acc (real val) |
|-------:|--------:|--------------:|-------------:|-----------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|-----------------:|------------:|--------------------:|--------------------------:|-------------------------:|
|      0 | 1.5e+05 |         1.188 |           25 |        1.5 |        100 |    27.61 |  0.01377 |      0.5964 |        0.7424 | 0.07496 | 0.01942 |            1 |                 1 |          0.8754 |              0.5152 |      0.5529 |            0.7167 |           0.918  |      0.797  |                   0 |                    0.6372 |                   0.8917 |
|      1 | 1.5e+05 |         1.176 |           25 |        1.5 |        100 |    27.32 |  0.01335 |      0.5973 |        0.7396 | 0.07743 | 0.02075 |            1 |                 1 |          0.8771 |              0.5185 |      0.5517 |            0.7167 |           0.9162 |      0.8016 |                   0 |                    0.6335 |                   0.8917 |
|      2 | 1.5e+05 |         1.169 |           25 |        1.5 |        100 |    27.23 |  0.01367 |      0.6015 |        0.7408 | 0.07634 | 0.02127 |            1 |                 1 |          0.8855 |              0.5013 |      0.5528 |            0.7167 |           0.9148 |      0.7945 |                   0 |                    0.6365 |                   0.8917 |

Figures (seed 0): [samples.png](oracle_s0_step150000_ddim25_A/samples.png), [nearest_real.png](oracle_s0_step150000_ddim25_A/nearest_real.png), [embedding.png](oracle_s0_step150000_ddim25_A/embedding.png), [g7_nn_grid.png](oracle_s0_step150000_ddim25_A/g7_nn_grid.png)
