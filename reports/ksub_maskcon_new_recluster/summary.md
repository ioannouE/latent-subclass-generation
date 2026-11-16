# ksub (known-K subclass baseline): maskcon_new, frozen dinov3_large, test

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `734259fe7363017bda5790ae738150eb9f62cd24-dirty` from `configs/ksub_maskcon_new_recluster.yaml`; training git ['734259fe7363017bda5790ae738150eb9f62cd24-dirty'], R1-R5 git ['734259fe7363017bda5790ae738150eb9f62cd24-dirty'];
seeds [0, 1, 2]; dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. **211 conflicting images ({'train': 98, 'val': 12, 'test': 101}) are excluded from every split**
(the `maskcon_new` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_256/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features). Classes with K_c = 1 are left out of the clustering columns.

| run              | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       |
|:-----------------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen           | 0.944 ± 0.000 | 0.800 ± 0.000  | 0.886 ± 0.000  | 0.415 ± 0.000 | 0.877 ± 0.000 | 0.429 ± 0.000 | 0.874 ± 0.000 | 0.792 ± 0.002 |
| sub_var          | 0.929 ± 0.002 | 0.810 ± 0.007  | 0.890 ± 0.001  | 0.401 ± 0.004 | 1.032 ± 0.000 | 0.415 ± 0.004 | 0.941 ± 0.003 | 0.810 ± 0.007 |
| sub_var_coarse01 | 0.978 ± 0.002 | 0.810 ± 0.008  | 0.886 ± 0.006  | 0.480 ± 0.003 | 0.927 ± 0.002 | 0.495 ± 0.003 | 0.927 ± 0.003 | 0.811 ± 0.007 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name   |   K_c |   n_test |   frozen ACC oracle K |   sub_var ACC oracle K |   sub_var_coarse01 ACC oracle K |   frozen ACC own |   sub_var ACC own |   sub_var_coarse01 ACC own |
|:--------------|------:|---------:|----------------------:|-----------------------:|--------------------------------:|-----------------:|------------------:|---------------------------:|
| Cab           |    18 |      744 |                 0.718 |                  0.783 |                           0.773 |            0.709 |             0.784 |                      0.779 |
| Sedan         |    50 |     2026 |                 0.732 |                  0.763 |                           0.75  |            0.738 |             0.748 |                      0.776 |
| SUV           |    36 |     1488 |                 0.821 |                  0.852 |                           0.867 |            0.845 |             0.856 |                      0.855 |
| Convertible   |    30 |     1176 |                 0.838 |                  0.791 |                           0.802 |            0.795 |             0.802 |                      0.802 |
| Coupe         |    29 |     1167 |                 0.767 |                  0.802 |                           0.789 |            0.768 |             0.779 |                      0.783 |
| Hatchback     |    14 |      571 |                 0.993 |                  0.963 |                           0.924 |            0.936 |             0.964 |                      0.924 |
| Wagon         |     6 |      241 |                 0.846 |                  0.84  |                           0.853 |            0.837 |             0.844 |                      0.851 |
| Van           |    13 |      527 |                 0.683 |                  0.688 |                           0.724 |            0.709 |             0.705 |                      0.719 |
