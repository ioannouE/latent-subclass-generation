# ksub (known-K subclass baseline): maskcon_new, frozen dinov3_large, val

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `1862299774c62c59037631a8482aa74625b608a5-dirty` from `configs/ksub_maskcon_new_tau.yaml`; training git ['1862299774c62c59037631a8482aa74625b608a5-dirty'], R1-R5 git ['1862299774c62c59037631a8482aa74625b608a5-dirty'];
seeds [0, 1, 2]; dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. **211 conflicting images ({'train': 98, 'val': 12, 'test': 101}) are excluded from every split**
(the `maskcon_new` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_256/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features), with the run's own K; K err = mean |K - K_c| over classes; K (per class) = mean over seeds. Classes with K_c = 1 are left out of the clustering columns.

| run      | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       | K err         |
|:---------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen   | 0.954 ± 0.000 | 0.810 ± 0.000  | 0.910 ± 0.000  | 0.409 ± 0.000 | 0.880 ± 0.000 | 0.421 ± 0.000 | 0.772 ± 0.000 | 0.818 ± 0.002 | 0.000 ± 0.000 |
| refresh1 | 0.966 ± 0.005 | 0.816 ± 0.003  | 0.911 ± 0.004  | 0.485 ± 0.001 | 0.929 ± 0.001 | 0.497 ± 0.001 | 0.779 ± 0.003 | 0.831 ± 0.006 | 0.000 ± 0.000 |
| tau0025  | 0.962 ± 0.004 | 0.834 ± 0.013  | 0.914 ± 0.005  | 0.480 ± 0.002 | 0.940 ± 0.001 | 0.491 ± 0.002 | 0.792 ± 0.005 | 0.824 ± 0.010 | 0.000 ± 0.000 |
| tau010   | 0.930 ± 0.001 | 0.529 ± 0.009  | 0.667 ± 0.006  | 0.834 ± 0.006 | 0.382 ± 0.004 | 0.844 ± 0.006 | 0.264 ± 0.010 | 0.527 ± 0.010 | 0.000 ± 0.000 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name   |   K_c |   n_test |   frozen ACC oracle K |   refresh1 ACC oracle K |   tau0025 ACC oracle K |   tau010 ACC oracle K |   frozen ACC own |   refresh1 ACC own |   tau0025 ACC own |   tau010 ACC own |   frozen K |   refresh1 K |   tau0025 K |   tau010 K |
|:--------------|------:|---------:|----------------------:|------------------------:|-----------------------:|----------------------:|-----------------:|-------------------:|------------------:|-----------------:|-----------:|-------------:|------------:|-----------:|
| Cab           |    18 |       75 |                 0.693 |                   0.716 |                  0.756 |                 0.391 |            0.716 |              0.742 |             0.751 |            0.476 |         18 |           18 |          18 |         18 |
| Sedan         |    50 |      205 |                 0.761 |                   0.748 |                  0.78  |                 0.572 |            0.766 |              0.772 |             0.795 |            0.654 |         50 |           50 |          50 |         50 |
| SUV           |    36 |      149 |                 0.886 |                   0.868 |                  0.875 |                 0.662 |            0.872 |              0.884 |             0.886 |            0.743 |         36 |           36 |          36 |         36 |
| Convertible   |    30 |      123 |                 0.764 |                   0.799 |                  0.805 |                 0.482 |            0.797 |              0.797 |             0.759 |            0.504 |         30 |           30 |          30 |         30 |
| Coupe         |    29 |      117 |                 0.821 |                   0.809 |                  0.84  |                 0.524 |            0.789 |              0.838 |             0.801 |            0.573 |         29 |           29 |          29 |         29 |
| Hatchback     |    14 |       57 |                 0.982 |                   0.971 |                  0.982 |                 0.637 |            0.971 |              0.982 |             0.982 |            0.515 |         14 |           14 |          14 |         14 |
| Wagon         |     6 |       24 |                 0.833 |                   0.875 |                  0.875 |                 0.528 |            0.875 |              0.861 |             0.847 |            0.417 |          6 |            6 |           6 |          6 |
| Van           |    13 |       53 |                 0.736 |                   0.742 |                  0.755 |                 0.434 |            0.755 |              0.774 |             0.767 |            0.34  |         13 |           13 |          13 |         13 |
