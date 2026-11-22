# Generation: oracle (maskcon_new)

Provenance: git `3349bdea58938af7465de8f1872e12690cd24a72-dirty`, config `configs/gen_oracle_maskcon_new.yaml`, seeds [0, 1, 2]. Conflicting images (same photo under different fine labels) are left out of every split: {'train': 98, 'val': 12, 'test': 101} (train / val / test). Scored on test.

## Mean and standard deviation over 3 seeds

| metric                  |     mean |       std |
|:------------------------|---------:|----------:|
| G1 FID                  | 21.23    | 0.2151    |
| G1 KID                  |  0.01456 | 0.0003683 |
| G1 recall               |  0.4853  | 0.01092   |
| G2 make acc             |  0.7204  | 0.004957  |
| G3 TV                   |  0.1302  | 0.003454  |
| G3 KL                   |  0.06002 | 0.002729  |
| G4 covered              |  0.9726  | 0.01946   |
| G4 rare covered         |  0.9363  | 0.05251   |
| G5 excess KID           |  2.82    | 0.005677  |
| G5 coverage ratio       |  0.01692 | 0.001982  |
| G7 copies               |  0.1638  | 0.02263   |
| G7 copies, real         |  0.1297  | 0         |
| G6 matched KID          |  2.956   | 0.01835   |
| G6 purity               |  0.7038  | 0.008693  |
| G6 unmatched real       |  0       | 0         |
| E1 fine acc (generated) |  0.5388  | 0.0101    |
| E1 fine acc (real val)  |  0.8468  | 1.36e-16  |

## Per seed

|   seed |   step |   train_hours |   ddim_steps |   guidance |   n_labels |   G1 FID |   G1 KID |   G1 recall |   G2 make acc |   G3 TV |   G3 KL |   G4 covered |   G4 rare covered |   G5 excess KID |   G5 coverage ratio |   G7 copies |   G7 copies, real |   G6 matched KID |   G6 purity |   G6 unmatched real |   E1 fine acc (generated) |   E1 fine acc (real val) |
|-------:|-------:|--------------:|-------------:|-----------:|-----------:|---------:|---------:|------------:|--------------:|--------:|--------:|-------------:|------------------:|----------------:|--------------------:|------------:|------------------:|-----------------:|------------:|--------------------:|--------------------------:|-------------------------:|
|      0 |  5e+04 |        0.4022 |           25 |        1.5 |        196 |    21.35 |  0.01457 |      0.4953 |        0.7158 |  0.1315 | 0.06035 |       0.975  |            0.9286 |           2.815 |             0.01476 |      0.183  |            0.1297 |            2.973 |      0.698  |                   0 |                    0.5282 |                   0.8468 |
|      1 |  5e+04 |        0.4011 |           25 |        1.5 |        196 |    21.35 |  0.01492 |      0.4737 |        0.7199 |  0.1328 | 0.06257 |       0.952  |            0.888  |           2.819 |             0.01735 |      0.1696 |            0.1297 |            2.959 |      0.6997 |                   0 |                    0.5397 |                   0.8468 |
|      2 |  5e+04 |        0.4051 |           25 |        1.5 |        196 |    20.98 |  0.01418 |      0.4869 |        0.7256 |  0.1263 | 0.05714 |       0.9907 |            0.9922 |           2.826 |             0.01865 |      0.1389 |            0.1297 |            2.936 |      0.7138 |                   0 |                    0.5484 |                   0.8468 |

Figures (seed 0): [samples.png](oracle_s0_step050000_ddim25_A/samples.png), [nearest_real.png](oracle_s0_step050000_ddim25_A/nearest_real.png), [embedding.png](oracle_s0_step050000_ddim25_A/embedding.png), [g7_nn_grid.png](oracle_s0_step050000_ddim25_A/g7_nn_grid.png)
