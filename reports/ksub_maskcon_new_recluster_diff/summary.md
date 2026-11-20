# ksub (known-K subclass baseline): maskcon_new, frozen dinov3_large, test

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `1862299774c62c59037631a8482aa74625b608a5-dirty` from `configs/ksub_maskcon_new_recluster_diff.yaml`; training git ['1862299774c62c59037631a8482aa74625b608a5-dirty'], R1-R5 git ['1862299774c62c59037631a8482aa74625b608a5-dirty'];
seeds [0, 1, 2]; dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. **211 conflicting images ({'train': 98, 'val': 12, 'test': 101}) are excluded from every split**
(the `maskcon_new` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_256/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features), with the run's own K; K err = mean |K - K_c| over classes; K (per class) = mean over seeds. Classes with K_c = 1 are left out of the clustering columns.

| run              | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       | K err         |
|:-----------------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen           | 0.944 ± 0.000 | 0.800 ± 0.000  | 0.886 ± 0.000  | 0.415 ± 0.000 | 0.877 ± 0.000 | 0.429 ± 0.000 | 0.874 ± 0.000 | 0.829 ± 0.001 | 0.000 ± 0.000 |
| sub_var          | 0.929 ± 0.003 | 0.811 ± 0.005  | 0.888 ± 0.001  | 0.403 ± 0.002 | 1.032 ± 0.000 | 0.417 ± 0.002 | 0.954 ± 0.008 | 0.820 ± 0.003 | 0.000 ± 0.000 |
| sub_var_coarse01 | 0.977 ± 0.001 | 0.819 ± 0.007  | 0.892 ± 0.003  | 0.479 ± 0.003 | 0.925 ± 0.001 | 0.494 ± 0.003 | 0.950 ± 0.009 | 0.825 ± 0.001 | 0.000 ± 0.000 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name   |   K_c |   n_test |   frozen ACC oracle K |   sub_var ACC oracle K |   sub_var_coarse01 ACC oracle K |   frozen ACC own |   sub_var ACC own |   sub_var_coarse01 ACC own |   frozen K |   sub_var K |   sub_var_coarse01 K |
|:--------------|------:|---------:|----------------------:|-----------------------:|--------------------------------:|-----------------:|------------------:|---------------------------:|-----------:|------------:|---------------------:|
| Cab           |    18 |      744 |                 0.718 |                  0.773 |                           0.791 |            0.778 |             0.771 |                      0.779 |         18 |          18 |                   18 |
| Sedan         |    50 |     2026 |                 0.732 |                  0.752 |                           0.771 |            0.808 |             0.807 |                      0.795 |         50 |          50 |                   50 |
| SUV           |    36 |     1488 |                 0.821 |                  0.905 |                           0.884 |            0.931 |             0.92  |                      0.915 |         36 |          36 |                   36 |
| Convertible   |    30 |     1176 |                 0.838 |                  0.796 |                           0.805 |            0.827 |             0.801 |                      0.813 |         30 |          30 |                   30 |
| Coupe         |    29 |     1167 |                 0.767 |                  0.747 |                           0.759 |            0.759 |             0.748 |                      0.756 |         29 |          29 |                   29 |
| Hatchback     |    14 |      571 |                 0.993 |                  0.995 |                           0.995 |            0.993 |             0.995 |                      0.995 |         14 |          14 |                   14 |
| Wagon         |     6 |      241 |                 0.846 |                  0.831 |                           0.852 |            0.842 |             0.833 |                      0.853 |          6 |           6 |                    6 |
| Van           |    13 |      527 |                 0.683 |                  0.684 |                           0.693 |            0.694 |             0.688 |                      0.696 |         13 |          13 |                   13 |
