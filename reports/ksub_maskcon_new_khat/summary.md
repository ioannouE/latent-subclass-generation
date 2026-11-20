# ksub (known-K subclass baseline): maskcon_new, frozen dinov3_large, test

**K unknown**: every run uses a label-free K-hat per coarse class, estimated per seed on the frozen train features (k_max 60) (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `1862299774c62c59037631a8482aa74625b608a5-dirty` from `configs/ksub_maskcon_new_khat.yaml`; training git ['1862299774c62c59037631a8482aa74625b608a5-dirty'], R1-R5 git ['1862299774c62c59037631a8482aa74625b608a5-dirty'];
seeds [0, 1, 2]; dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. **211 conflicting images ({'train': 98, 'val': 12, 'test': 101}) are excluded from every split**
(the `maskcon_new` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_256/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features), with the run's own K; K err = mean |K - K_c| over classes; K (per class) = mean over seeds. Classes with K_c = 1 are left out of the clustering columns.

| run              | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       | K err         |
|:-----------------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen           | 0.944 ± 0.000 | 0.800 ± 0.000  | 0.886 ± 0.000  | 0.415 ± 0.000 | 0.877 ± 0.000 | 0.429 ± 0.000 | 0.874 ± 0.000 | 0.719 ± 0.034 | 5.417 ± 0.191 |
| sub_var          | 0.931 ± 0.005 | 0.764 ± 0.023  | 0.863 ± 0.019  | 0.409 ± 0.006 | 1.028 ± 0.003 | 0.423 ± 0.006 | 0.870 ± 0.039 | 0.717 ± 0.028 | 5.417 ± 0.191 |
| sub_var_coarse01 | 0.978 ± 0.001 | 0.767 ± 0.024  | 0.862 ± 0.014  | 0.488 ± 0.011 | 0.912 ± 0.003 | 0.502 ± 0.011 | 0.861 ± 0.030 | 0.721 ± 0.023 | 5.417 ± 0.191 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name   |   K_c |   n_test |   frozen ACC oracle K |   sub_var ACC oracle K |   sub_var_coarse01 ACC oracle K |   frozen ACC own |   sub_var ACC own |   sub_var_coarse01 ACC own |   frozen K |   sub_var K |   sub_var_coarse01 K |
|:--------------|------:|---------:|----------------------:|-----------------------:|--------------------------------:|-----------------:|------------------:|---------------------------:|-----------:|------------:|---------------------:|
| Cab           |    18 |      744 |                 0.718 |                  0.527 |                           0.526 |            0.358 |             0.363 |                      0.359 |      6.667 |       6.667 |                6.667 |
| Sedan         |    50 |     2026 |                 0.732 |                  0.749 |                           0.762 |            0.723 |             0.735 |                      0.734 |     47.333 |      47.333 |               47.333 |
| SUV           |    36 |     1488 |                 0.821 |                  0.844 |                           0.844 |            0.842 |             0.842 |                      0.851 |     36.333 |      36.333 |               36.333 |
| Convertible   |    30 |     1176 |                 0.838 |                  0.787 |                           0.788 |            0.778 |             0.785 |                      0.773 |     34     |      34     |               34     |
| Coupe         |    29 |     1167 |                 0.767 |                  0.766 |                           0.763 |            0.736 |             0.714 |                      0.719 |     38.333 |      38.333 |               38.333 |
| Hatchback     |    14 |      571 |                 0.993 |                  0.975 |                           0.98  |            0.799 |             0.788 |                      0.786 |     22     |      22     |               22     |
| Wagon         |     6 |      241 |                 0.846 |                  0.806 |                           0.798 |            0.83  |             0.827 |                      0.842 |      5     |       5     |                5     |
| Van           |    13 |      527 |                 0.683 |                  0.66  |                           0.679 |            0.686 |             0.681 |                      0.703 |     11     |      11     |               11     |
