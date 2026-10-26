# ksub (known-K subclass baseline): maskcon_new, frozen dinov3_large, test

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `3209598aaa42a252272526b1498994c419ff56c4-dirty` from `configs/ksub_maskcon_new.yaml`; training git ['3209598aaa42a252272526b1498994c419ff56c4-dirty'], R1-R5 git ['3209598aaa42a252272526b1498994c419ff56c4-dirty'];
seeds [0, 1, 2]; dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. **211 conflicting images ({'train': 98, 'val': 12, 'test': 101}) are excluded from every split**
(the `maskcon_new` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_256/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features). Classes with K_c = 1 are left out of the clustering columns.

| run              | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       |
|:-----------------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen           | 0.944 ± 0.000 | 0.800 ± 0.000  | 0.886 ± 0.000  | 0.415 ± 0.000 | 0.877 ± 0.000 | 0.429 ± 0.000 | 0.874 ± 0.000 | 0.792 ± 0.002 |
| sub_only         | 0.935 ± 0.006 | 0.605 ± 0.014  | 0.793 ± 0.029  | 0.063 ± 0.009 | 0.990 ± 0.070 | 0.067 ± 0.010 | 0.815 ± 0.053 | 0.528 ± 0.047 |
| sub_var          | 0.908 ± 0.009 | 0.653 ± 0.004  | 0.819 ± 0.005  | 0.396 ± 0.010 | 1.029 ± 0.000 | 0.409 ± 0.010 | 0.805 ± 0.008 | 0.610 ± 0.013 |
| sub_var_coarse01 | 0.973 ± 0.003 | 0.652 ± 0.011  | 0.806 ± 0.005  | 0.503 ± 0.012 | 0.851 ± 0.012 | 0.517 ± 0.012 | 0.741 ± 0.004 | 0.588 ± 0.006 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name   |   K_c |   n_test |   frozen ACC oracle K |   sub_only ACC oracle K |   sub_var ACC oracle K |   sub_var_coarse01 ACC oracle K |   frozen ACC own |   sub_only ACC own |   sub_var ACC own |   sub_var_coarse01 ACC own |
|:--------------|------:|---------:|----------------------:|------------------------:|-----------------------:|--------------------------------:|-----------------:|-------------------:|------------------:|---------------------------:|
| Cab           |    18 |      744 |                 0.718 |                   0.51  |                  0.649 |                           0.597 |            0.709 |              0.316 |             0.594 |                      0.534 |
| Sedan         |    50 |     2026 |                 0.732 |                   0.408 |                  0.451 |                           0.521 |            0.738 |              0.293 |             0.307 |                      0.261 |
| SUV           |    36 |     1488 |                 0.821 |                   0.493 |                  0.479 |                           0.529 |            0.845 |              0.337 |             0.392 |                      0.368 |
| Convertible   |    30 |     1176 |                 0.838 |                   0.506 |                  0.541 |                           0.527 |            0.795 |              0.432 |             0.54  |                      0.515 |
| Coupe         |    29 |     1167 |                 0.767 |                   0.521 |                  0.603 |                           0.562 |            0.768 |              0.426 |             0.534 |                      0.516 |
| Hatchback     |    14 |      571 |                 0.993 |                   0.946 |                  0.952 |                           0.953 |            0.936 |              0.945 |             0.952 |                      0.95  |
| Wagon         |     6 |      241 |                 0.846 |                   0.834 |                  0.849 |                           0.845 |            0.837 |              0.846 |             0.851 |                      0.845 |
| Van           |    13 |      527 |                 0.683 |                   0.622 |                  0.704 |                           0.685 |            0.709 |              0.626 |             0.709 |                      0.717 |
