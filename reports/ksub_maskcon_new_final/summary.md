# ksub (known-K subclass baseline): maskcon_new, frozen dinov3_large, test

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `2b82f53f8e709b93d4d8c9fb54a8a8934121ec25-dirty` from `configs/ksub_maskcon_new_final.yaml`; training git ['1862299774c62c59037631a8482aa74625b608a5-dirty'], R1-R5 git ['2b82f53f8e709b93d4d8c9fb54a8a8934121ec25-dirty'];
seeds [0, 1, 2]; dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. **211 conflicting images ({'train': 98, 'val': 12, 'test': 101}) are excluded from every split**
(the `maskcon_new` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_256/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse / probe coarse = R1 kNN@20 / linear probe; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features), with the run's own K; K err = mean |K - K_c| over classes; K (per class) = mean over seeds. Classes with K_c = 1 are left out of the clustering columns.

| run     | kNN coarse    | probe coarse   | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       | K err         |
|:--------|:--------------|:---------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen  | 0.944 ± 0.000 | 0.973 ± 0.000  | 0.800 ± 0.000  | 0.886 ± 0.000  | 0.415 ± 0.000 | 0.877 ± 0.000 | 0.429 ± 0.000 | 0.874 ± 0.000 | 0.792 ± 0.002 | 0.000 ± 0.000 |
| tau0025 | 0.974 ± 0.001 | 0.971 ± 0.001  | 0.820 ± 0.009  | 0.887 ± 0.006  | 0.473 ± 0.002 | 0.937 ± 0.001 | 0.487 ± 0.002 | 0.951 ± 0.000 | 0.813 ± 0.011 | 0.000 ± 0.000 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name   |   K_c |   n_test |   frozen ACC oracle K |   tau0025 ACC oracle K |   frozen ACC own |   tau0025 ACC own |   frozen K |   tau0025 K |
|:--------------|------:|---------:|----------------------:|-----------------------:|-----------------:|------------------:|-----------:|------------:|
| Cab           |    18 |      744 |                 0.718 |                  0.787 |            0.709 |             0.78  |         18 |          18 |
| Sedan         |    50 |     2026 |                 0.732 |                  0.778 |            0.738 |             0.761 |         50 |          50 |
| SUV           |    36 |     1488 |                 0.821 |                  0.859 |            0.845 |             0.86  |         36 |          36 |
| Convertible   |    30 |     1176 |                 0.838 |                  0.792 |            0.795 |             0.792 |         30 |          30 |
| Coupe         |    29 |     1167 |                 0.767 |                  0.806 |            0.768 |             0.781 |         29 |          29 |
| Hatchback     |    14 |      571 |                 0.993 |                  0.96  |            0.936 |             0.957 |         14 |          14 |
| Wagon         |     6 |      241 |                 0.846 |                  0.848 |            0.837 |             0.845 |          6 |           6 |
| Van           |    13 |      527 |                 0.683 |                  0.727 |            0.709 |             0.727 |         13 |          13 |
