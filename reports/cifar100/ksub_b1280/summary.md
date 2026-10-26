# ksub (known-K subclass baseline): cifar100, frozen dinov3_large, test

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `3209598aaa42a252272526b1498994c419ff56c4-dirty` from `configs/ksub_cifar100_b1280.yaml`; training git ['3209598aaa42a252272526b1498994c419ff56c4-dirty'], R1-R5 git ['3209598aaa42a252272526b1498994c419ff56c4-dirty'];
seeds [0, 1, 2]; dataset manifest `1cf015739e9b592dcee9dfbf0050118bced8343b8aa77ea080a6f27d1c394038`. **90 conflicting images ({'train': 68, 'val': 5, 'test': 17}) are excluded from every split**
(the `cifar100` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_32/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features). Classes with K_c = 1 are left out of the clustering columns.

| run              | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       |
|:-----------------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen           | 0.964 ± 0.000 | 0.892 ± 0.000  | 0.800 ± 0.000  | 0.733 ± 0.000 | 0.921 ± 0.000 | 0.773 ± 0.000 | 0.891 ± 0.000 | 0.902 ± 0.000 |
| sub_var          | 0.954 ± 0.002 | 0.912 ± 0.002  | 0.831 ± 0.004  | 0.691 ± 0.001 | 1.062 ± 0.000 | 0.735 ± 0.001 | 0.992 ± 0.002 | 0.912 ± 0.002 |
| sub_var_coarse01 | 0.967 ± 0.001 | 0.912 ± 0.001  | 0.834 ± 0.003  | 0.725 ± 0.000 | 0.994 ± 0.000 | 0.766 ± 0.000 | 0.988 ± 0.002 | 0.911 ± 0.001 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name                    |   K_c |   n_test |   frozen ACC oracle K |   sub_var ACC oracle K |   sub_var_coarse01 ACC oracle K |   frozen ACC own |   sub_var ACC own |   sub_var_coarse01 ACC own |
|:-------------------------------|------:|---------:|----------------------:|-----------------------:|--------------------------------:|-----------------:|------------------:|---------------------------:|
| aquatic_mammals                |     5 |      494 |                 0.83  |                  0.756 |                           0.753 |            0.78  |             0.749 |                      0.758 |
| fish                           |     5 |      500 |                 0.95  |                  0.951 |                           0.95  |            0.948 |             0.953 |                      0.951 |
| flowers                        |     5 |      500 |                 0.958 |                  0.967 |                           0.965 |            0.956 |             0.965 |                      0.966 |
| food_containers                |     5 |      500 |                 0.854 |                  0.9   |                           0.902 |            0.902 |             0.886 |                      0.891 |
| fruit_and_vegetables           |     5 |      499 |                 0.982 |                  0.98  |                           0.98  |            0.982 |             0.981 |                      0.981 |
| household_electrical_devices   |     5 |      500 |                 0.906 |                  0.931 |                           0.922 |            0.924 |             0.934 |                      0.923 |
| household_furniture            |     5 |      500 |                 0.904 |                  0.928 |                           0.931 |            0.906 |             0.927 |                      0.925 |
| insects                        |     5 |      500 |                 0.92  |                  0.944 |                           0.939 |            0.926 |             0.944 |                      0.943 |
| large_carnivores               |     5 |      500 |                 0.914 |                  0.96  |                           0.937 |            0.938 |             0.962 |                      0.937 |
| large_man-made_outdoor_things  |     5 |      500 |                 0.91  |                  0.941 |                           0.948 |            0.922 |             0.939 |                      0.946 |
| large_natural_outdoor_scenes   |     5 |      500 |                 0.906 |                  0.885 |                           0.898 |            0.898 |             0.887 |                      0.897 |
| large_omnivores_and_herbivores |     5 |      500 |                 0.988 |                  0.985 |                           0.988 |            0.992 |             0.985 |                      0.988 |
| medium_mammals                 |     5 |      499 |                 0.908 |                  0.964 |                           0.968 |            0.952 |             0.969 |                      0.971 |
| non-insect_invertebrates       |     5 |      498 |                 0.894 |                  0.962 |                           0.96  |            0.959 |             0.963 |                      0.966 |
| people                         |     5 |      496 |                 0.643 |                  0.72  |                           0.721 |            0.639 |             0.72  |                      0.722 |
| reptiles                       |     5 |      500 |                 0.978 |                  0.981 |                           0.985 |            0.97  |             0.98  |                      0.979 |
| small_mammals                  |     5 |      499 |                 0.844 |                  0.798 |                           0.808 |            0.804 |             0.798 |                      0.802 |
| trees                          |     5 |      499 |                 0.603 |                  0.749 |                           0.733 |            0.697 |             0.749 |                      0.735 |
| vehicles_1                     |     5 |      499 |                 0.982 |                  0.981 |                           0.983 |            0.978 |             0.981 |                      0.98  |
| vehicles_2                     |     5 |      500 |                 0.958 |                  0.965 |                           0.963 |            0.96  |             0.963 |                      0.961 |
