# ksub (known-K subclass baseline): cifar100, frozen dinov3_large, test

**K unknown**: every run uses a label-free K-hat per coarse class, estimated per seed on the frozen train features (k_max 60) (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `1862299774c62c59037631a8482aa74625b608a5-dirty` from `configs/ksub_cifar100_khat.yaml`; training git ['1862299774c62c59037631a8482aa74625b608a5-dirty'], R1-R5 git ['1862299774c62c59037631a8482aa74625b608a5-dirty'];
seeds [0, 1, 2]; dataset manifest `1cf015739e9b592dcee9dfbf0050118bced8343b8aa77ea080a6f27d1c394038`. **90 conflicting images ({'train': 68, 'val': 5, 'test': 17}) are excluded from every split**
(the `cifar100` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_32/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features), with the run's own K; K err = mean |K - K_c| over classes; K (per class) = mean over seeds. Classes with K_c = 1 are left out of the clustering columns.

| run              | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       | K err         |
|:-----------------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen           | 0.964 ± 0.000 | 0.892 ± 0.000  | 0.800 ± 0.000  | 0.733 ± 0.000 | 0.921 ± 0.000 | 0.773 ± 0.000 | 0.891 ± 0.000 | 0.720 ± 0.006 | 3.050 ± 0.100 |
| sub_var          | 0.955 ± 0.001 | 0.847 ± 0.005  | 0.775 ± 0.005  | 0.655 ± 0.001 | 1.064 ± 0.001 | 0.702 ± 0.001 | 0.910 ± 0.005 | 0.718 ± 0.010 | 3.050 ± 0.100 |
| sub_var_coarse01 | 0.965 ± 0.001 | 0.850 ± 0.004  | 0.781 ± 0.003  | 0.688 ± 0.002 | 0.956 ± 0.004 | 0.732 ± 0.002 | 0.912 ± 0.002 | 0.725 ± 0.004 | 3.050 ± 0.100 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name                    |   K_c |   n_test |   frozen ACC oracle K |   sub_var ACC oracle K |   sub_var_coarse01 ACC oracle K |   frozen ACC own |   sub_var ACC own |   sub_var_coarse01 ACC own |   frozen K |   sub_var K |   sub_var_coarse01 K |
|:-------------------------------|------:|---------:|----------------------:|-----------------------:|--------------------------------:|-----------------:|------------------:|---------------------------:|-----------:|------------:|---------------------:|
| aquatic_mammals                |     5 |      494 |                 0.83  |                  0.432 |                           0.421 |            0.405 |             0.4   |                      0.402 |      2     |       2     |                2     |
| fish                           |     5 |      500 |                 0.95  |                  0.894 |                           0.873 |            0.802 |             0.833 |                      0.825 |      7     |       7     |                7     |
| flowers                        |     5 |      500 |                 0.958 |                  0.938 |                           0.936 |            0.817 |             0.831 |                      0.853 |      6.333 |       6.333 |                6.333 |
| food_containers                |     5 |      500 |                 0.854 |                  0.801 |                           0.796 |            0.755 |             0.758 |                      0.749 |      4     |       4     |                4     |
| fruit_and_vegetables           |     5 |      499 |                 0.982 |                  0.977 |                           0.978 |            0.982 |             0.977 |                      0.979 |      5     |       5     |                5     |
| household_electrical_devices   |     5 |      500 |                 0.906 |                  0.887 |                           0.885 |            0.813 |             0.812 |                      0.811 |      7.333 |       7.333 |                7.333 |
| household_furniture            |     5 |      500 |                 0.904 |                  0.907 |                           0.923 |            0.457 |             0.473 |                      0.497 |     14.333 |      14.333 |               14.333 |
| insects                        |     5 |      500 |                 0.92  |                  0.932 |                           0.97  |            0.585 |             0.515 |                      0.574 |     12.667 |      12.667 |               12.667 |
| large_carnivores               |     5 |      500 |                 0.914 |                  0.955 |                           0.95  |            0.938 |             0.956 |                      0.951 |      5     |       5     |                5     |
| large_man-made_outdoor_things  |     5 |      500 |                 0.91  |                  0.964 |                           0.966 |            0.816 |             0.794 |                      0.805 |      7     |       7     |                7     |
| large_natural_outdoor_scenes   |     5 |      500 |                 0.906 |                  0.896 |                           0.901 |            0.898 |             0.894 |                      0.902 |      5     |       5     |                5     |
| large_omnivores_and_herbivores |     5 |      500 |                 0.988 |                  0.983 |                           0.988 |            0.992 |             0.983 |                      0.985 |      5     |       5     |                5     |
| medium_mammals                 |     5 |      499 |                 0.908 |                  0.848 |                           0.876 |            0.775 |             0.755 |                      0.725 |      7     |       7     |                7     |
| non-insect_invertebrates       |     5 |      498 |                 0.894 |                  0.942 |                           0.952 |            0.752 |             0.784 |                      0.79  |      7     |       7     |                7     |
| people                         |     5 |      496 |                 0.643 |                  0.669 |                           0.63  |            0.252 |             0.259 |                      0.287 |     20     |      20     |               20     |
| reptiles                       |     5 |      500 |                 0.978 |                  0.978 |                           0.981 |            0.97  |             0.977 |                      0.981 |      5     |       5     |                5     |
| small_mammals                  |     5 |      499 |                 0.844 |                  0.582 |                           0.609 |            0.583 |             0.575 |                      0.588 |      3     |       3     |                3     |
| trees                          |     5 |      499 |                 0.603 |                  0.432 |                           0.439 |            0.401 |             0.4   |                      0.401 |      2     |       2     |                2     |
| vehicles_1                     |     5 |      499 |                 0.982 |                  0.981 |                           0.983 |            0.515 |             0.499 |                      0.497 |     12.333 |      12.333 |               12.333 |
| vehicles_2                     |     5 |      500 |                 0.958 |                  0.951 |                           0.949 |            0.892 |             0.891 |                      0.89  |      6     |       6     |                6     |
