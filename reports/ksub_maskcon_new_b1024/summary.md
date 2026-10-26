# ksub (known-K subclass baseline): maskcon_new, frozen dinov3_large, test

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `3209598aaa42a252272526b1498994c419ff56c4-dirty` from `configs/ksub_maskcon_new_b1024.yaml`; training git ['3209598aaa42a252272526b1498994c419ff56c4-dirty'], R1-R5 git ['3209598aaa42a252272526b1498994c419ff56c4-dirty'];
seeds [0, 1, 2]; dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. **211 conflicting images ({'train': 98, 'val': 12, 'test': 101}) are excluded from every split**
(the `maskcon_new` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_256/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features). Classes with K_c = 1 are left out of the clustering columns.

| run              | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       |
|:-----------------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen           | 0.944 ± 0.000 | 0.800 ± 0.000  | 0.886 ± 0.000  | 0.415 ± 0.000 | 0.877 ± 0.000 | 0.429 ± 0.000 | 0.874 ± 0.000 | 0.792 ± 0.002 |
| sub_var          | 0.951 ± 0.000 | 0.796 ± 0.004  | 0.890 ± 0.003  | 0.485 ± 0.007 | 1.026 ± 0.001 | 0.500 ± 0.007 | 0.973 ± 0.006 | 0.795 ± 0.005 |
| sub_var_coarse01 | 0.985 ± 0.001 | 0.788 ± 0.005  | 0.887 ± 0.001  | 0.564 ± 0.002 | 0.918 ± 0.002 | 0.579 ± 0.002 | 0.953 ± 0.004 | 0.790 ± 0.006 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name   |   K_c |   n_test |   frozen ACC oracle K |   sub_var ACC oracle K |   sub_var_coarse01 ACC oracle K |   frozen ACC own |   sub_var ACC own |   sub_var_coarse01 ACC own |
|:--------------|------:|---------:|----------------------:|-----------------------:|--------------------------------:|-----------------:|------------------:|---------------------------:|
| Cab           |    18 |      744 |                 0.718 |                  0.695 |                           0.673 |            0.709 |             0.686 |                      0.666 |
| Sedan         |    50 |     2026 |                 0.732 |                  0.744 |                           0.75  |            0.738 |             0.732 |                      0.733 |
| SUV           |    36 |     1488 |                 0.821 |                  0.857 |                           0.867 |            0.845 |             0.856 |                      0.854 |
| Convertible   |    30 |     1176 |                 0.838 |                  0.802 |                           0.8   |            0.795 |             0.803 |                      0.8   |
| Coupe         |    29 |     1167 |                 0.767 |                  0.784 |                           0.787 |            0.768 |             0.789 |                      0.78  |
| Hatchback     |    14 |      571 |                 0.993 |                  0.933 |                           0.929 |            0.936 |             0.932 |                      0.93  |
| Wagon         |     6 |      241 |                 0.846 |                  0.853 |                           0.795 |            0.837 |             0.853 |                      0.852 |
| Van           |    13 |      527 |                 0.683 |                  0.701 |                           0.706 |            0.709 |             0.708 |                      0.707 |
