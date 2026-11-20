# ksub (known-K subclass baseline): maskcon_new, frozen dinov3_large, val

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `1862299774c62c59037631a8482aa74625b608a5-dirty` from `configs/ksub_maskcon_new_refresh.yaml`; training git ['1862299774c62c59037631a8482aa74625b608a5-dirty'], R1-R5 git ['1862299774c62c59037631a8482aa74625b608a5-dirty'];
seeds [0, 1, 2]; dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. **211 conflicting images ({'train': 98, 'val': 12, 'test': 101}) are excluded from every split**
(the `maskcon_new` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_256/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features), with the run's own K; K err = mean |K - K_c| over classes; K (per class) = mean over seeds. Classes with K_c = 1 are left out of the clustering columns.

| run      | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       | K err         |
|:---------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen   | 0.954 ± 0.000 | 0.810 ± 0.000  | 0.910 ± 0.000  | 0.409 ± 0.000 | 0.880 ± 0.000 | 0.421 ± 0.000 | 0.772 ± 0.000 | 0.818 ± 0.002 | 0.000 ± 0.000 |
| ema099   | 0.957 ± 0.002 | 0.683 ± 0.010  | 0.849 ± 0.005  | 0.512 ± 0.014 | 0.857 ± 0.015 | 0.522 ± 0.014 | 0.668 ± 0.007 | 0.630 ± 0.011 | 0.000 ± 0.000 |
| refresh1 | 0.966 ± 0.005 | 0.816 ± 0.003  | 0.911 ± 0.004  | 0.485 ± 0.001 | 0.929 ± 0.001 | 0.497 ± 0.001 | 0.779 ± 0.003 | 0.831 ± 0.006 | 0.000 ± 0.000 |
| refresh5 | 0.967 ± 0.002 | 0.767 ± 0.009  | 0.888 ± 0.004  | 0.510 ± 0.005 | 0.917 ± 0.002 | 0.522 ± 0.005 | 0.725 ± 0.007 | 0.771 ± 0.002 | 0.000 ± 0.000 |
| ema095   | 0.962 ± 0.004 | 0.669 ± 0.017  | 0.840 ± 0.005  | 0.505 ± 0.005 | 0.831 ± 0.007 | 0.516 ± 0.004 | 0.650 ± 0.011 | 0.602 ± 0.009 | 0.000 ± 0.000 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name   |   K_c |   n_test |   frozen ACC oracle K |   ema099 ACC oracle K |   refresh1 ACC oracle K |   refresh5 ACC oracle K |   ema095 ACC oracle K |   frozen ACC own |   ema099 ACC own |   refresh1 ACC own |   refresh5 ACC own |   ema095 ACC own |   frozen K |   ema099 K |   refresh1 K |   refresh5 K |   ema095 K |
|:--------------|------:|---------:|----------------------:|----------------------:|------------------------:|------------------------:|----------------------:|-----------------:|-----------------:|-------------------:|-------------------:|-----------------:|-----------:|-----------:|-------------:|-------------:|-----------:|
| Cab           |    18 |       75 |                 0.693 |                 0.676 |                   0.716 |                   0.729 |                 0.64  |            0.716 |            0.569 |              0.742 |              0.698 |            0.542 |         18 |         18 |           18 |           18 |         18 |
| Sedan         |    50 |      205 |                 0.761 |                 0.496 |                   0.748 |                   0.633 |                 0.499 |            0.766 |            0.363 |              0.772 |              0.693 |            0.332 |         50 |         50 |           50 |           50 |         50 |
| SUV           |    36 |      149 |                 0.886 |                 0.528 |                   0.868 |                   0.707 |                 0.51  |            0.872 |            0.452 |              0.884 |              0.767 |            0.416 |         36 |         36 |           36 |           36 |         36 |
| Convertible   |    30 |      123 |                 0.764 |                 0.569 |                   0.799 |                   0.762 |                 0.542 |            0.797 |            0.556 |              0.797 |              0.726 |            0.474 |         30 |         30 |           30 |           30 |         30 |
| Coupe         |    29 |      117 |                 0.821 |                 0.615 |                   0.809 |                   0.783 |                 0.644 |            0.789 |            0.558 |              0.838 |              0.764 |            0.496 |         29 |         29 |           29 |           29 |         29 |
| Hatchback     |    14 |       57 |                 0.982 |                 0.971 |                   0.971 |                   0.918 |                 0.965 |            0.971 |            0.971 |              0.982 |              0.93  |            0.971 |         14 |         14 |           14 |           14 |         14 |
| Wagon         |     6 |       24 |                 0.833 |                 0.875 |                   0.875 |                   0.875 |                 0.861 |            0.875 |            0.833 |              0.861 |              0.833 |            0.833 |          6 |          6 |            6 |            6 |          6 |
| Van           |    13 |       53 |                 0.736 |                 0.736 |                   0.742 |                   0.73  |                 0.692 |            0.755 |            0.736 |              0.774 |              0.755 |            0.748 |         13 |         13 |           13 |           13 |         13 |
