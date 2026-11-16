# ksub (known-K subclass baseline): cifar100, frozen dinov3_large, test

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `734259fe7363017bda5790ae738150eb9f62cd24-dirty` from `configs/ksub_cifar100_recluster.yaml`; training git ['734259fe7363017bda5790ae738150eb9f62cd24-dirty'], R1-R5 git ['734259fe7363017bda5790ae738150eb9f62cd24-dirty'];
seeds [0, 1, 2]; dataset manifest `1cf015739e9b592dcee9dfbf0050118bced8343b8aa77ea080a6f27d1c394038`. **90 conflicting images ({'train': 68, 'val': 5, 'test': 17}) are excluded from every split**
(the `cifar100` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_32/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features). Classes with K_c = 1 are left out of the clustering columns.

| run              | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       |
|:-----------------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen           | 0.964 ± 0.000 | 0.892 ± 0.000  | 0.800 ± 0.000  | 0.733 ± 0.000 | 0.921 ± 0.000 | 0.773 ± 0.000 | 0.891 ± 0.000 | 0.902 ± 0.000 |
| sub_var          | 0.959 ± 0.001 | 0.909 ± 0.001  | 0.830 ± 0.005  | 0.650 ± 0.003 | 1.073 ± 0.000 | 0.698 ± 0.003 | 0.992 ± 0.001 | 0.910 ± 0.001 |
| sub_var_coarse01 | 0.968 ± 0.001 | 0.911 ± 0.000  | 0.834 ± 0.001  | 0.685 ± 0.001 | 0.996 ± 0.000 | 0.730 ± 0.001 | 0.991 ± 0.001 | 0.911 ± 0.001 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name                    |   K_c |   n_test |   frozen ACC oracle K |   sub_var ACC oracle K |   sub_var_coarse01 ACC oracle K |   frozen ACC own |   sub_var ACC own |   sub_var_coarse01 ACC own |
|:-------------------------------|------:|---------:|----------------------:|-----------------------:|--------------------------------:|-----------------:|------------------:|---------------------------:|
| aquatic_mammals                |     5 |      494 |                 0.83  |                  0.776 |                           0.771 |            0.78  |             0.778 |                      0.779 |
| fish                           |     5 |      500 |                 0.95  |                  0.943 |                           0.945 |            0.948 |             0.946 |                      0.947 |
| flowers                        |     5 |      500 |                 0.958 |                  0.967 |                           0.964 |            0.956 |             0.968 |                      0.965 |
| food_containers                |     5 |      500 |                 0.854 |                  0.899 |                           0.893 |            0.902 |             0.893 |                      0.887 |
| fruit_and_vegetables           |     5 |      499 |                 0.982 |                  0.979 |                           0.979 |            0.982 |             0.977 |                      0.979 |
| household_electrical_devices   |     5 |      500 |                 0.906 |                  0.935 |                           0.925 |            0.924 |             0.936 |                      0.926 |
| household_furniture            |     5 |      500 |                 0.904 |                  0.923 |                           0.93  |            0.906 |             0.923 |                      0.928 |
| insects                        |     5 |      500 |                 0.92  |                  0.938 |                           0.942 |            0.926 |             0.94  |                      0.94  |
| large_carnivores               |     5 |      500 |                 0.914 |                  0.951 |                           0.933 |            0.938 |             0.951 |                      0.933 |
| large_man-made_outdoor_things  |     5 |      500 |                 0.91  |                  0.938 |                           0.941 |            0.922 |             0.938 |                      0.94  |
| large_natural_outdoor_scenes   |     5 |      500 |                 0.906 |                  0.894 |                           0.903 |            0.898 |             0.894 |                      0.899 |
| large_omnivores_and_herbivores |     5 |      500 |                 0.988 |                  0.986 |                           0.989 |            0.992 |             0.987 |                      0.988 |
| medium_mammals                 |     5 |      499 |                 0.908 |                  0.966 |                           0.969 |            0.952 |             0.968 |                      0.972 |
| non-insect_invertebrates       |     5 |      498 |                 0.894 |                  0.954 |                           0.959 |            0.96  |             0.956 |                      0.963 |
| people                         |     5 |      496 |                 0.643 |                  0.709 |                           0.684 |            0.639 |             0.718 |                      0.685 |
| reptiles                       |     5 |      500 |                 0.978 |                  0.976 |                           0.983 |            0.97  |             0.975 |                      0.981 |
| small_mammals                  |     5 |      499 |                 0.844 |                  0.814 |                           0.822 |            0.804 |             0.81  |                      0.818 |
| trees                          |     5 |      499 |                 0.603 |                  0.693 |                           0.738 |            0.697 |             0.709 |                      0.739 |
| vehicles_1                     |     5 |      499 |                 0.982 |                  0.981 |                           0.983 |            0.978 |             0.981 |                      0.983 |
| vehicles_2                     |     5 |      500 |                 0.958 |                  0.961 |                           0.961 |            0.96  |             0.96  |                      0.959 |
