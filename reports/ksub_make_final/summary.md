# ksub (known-K subclass baseline): make, frozen dinov3_large, test

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `da600777157d2d6d5c55e3c5b668d32c1387ed29-dirty` from `configs/ksub_make_final.yaml`; training git ['da600777157d2d6d5c55e3c5b668d32c1387ed29-dirty'], R1-R5 git ['da600777157d2d6d5c55e3c5b668d32c1387ed29-dirty'];
seeds [0, 1, 2]; dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. **154 conflicting images ({'train': 59, 'val': 8, 'test': 87}) are excluded from every split**
(the `make` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_256/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse / probe coarse = R1 kNN@20 / linear probe; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features), with the run's own K; K err = mean |K - K_c| over classes; K (per class) = mean over seeds. Classes with K_c = 1 are left out of the clustering columns.

| run     | kNN coarse    | probe coarse   | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       | K err         |
|:--------|:--------------|:---------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen  | 0.973 ± 0.000 | 0.983 ± 0.000  | 0.782 ± 0.000  | 0.703 ± 0.000  | 0.570 ± 0.000 | 0.709 ± 0.000 | 0.635 ± 0.000 | 0.886 ± 0.000 | 0.787 ± 0.003 | 0.000 ± 0.000 |
| tau0025 | 0.980 ± 0.000 | 0.979 ± 0.000  | 0.803 ± 0.004  | 0.710 ± 0.003  | 0.656 ± 0.002 | 0.794 ± 0.001 | 0.707 ± 0.002 | 0.981 ± 0.001 | 0.801 ± 0.004 | 0.000 ± 0.000 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| coarse_name   |   K_c |   n_test |   frozen ACC oracle K |   tau0025 ACC oracle K |   frozen ACC own |   tau0025 ACC own |   frozen K |   tau0025 K |
|:--------------|------:|---------:|----------------------:|-----------------------:|-----------------:|------------------:|-----------:|------------:|
| AM General    |     1 |       44 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Acura         |     6 |      235 |                 0.749 |                  0.962 |            0.901 |             0.957 |          6 |           6 |
| Aston Martin  |     4 |      153 |                 0.497 |                  0.449 |            0.44  |             0.444 |          4 |           4 |
| Audi          |    13 |      551 |                 0.508 |                  0.561 |            0.498 |             0.55  |         13 |          13 |
| BMW           |    13 |      519 |                 0.609 |                  0.658 |            0.59  |             0.635 |         13 |          13 |
| Bentley       |     5 |      232 |                 0.625 |                  0.739 |            0.743 |             0.741 |          5 |           5 |
| Bugatti       |     2 |       70 |                 0.686 |                  0.686 |            0.686 |             0.686 |          2 |           2 |
| Buick         |     4 |      154 |                 0.818 |                  0.857 |            0.818 |             0.857 |          4 |           4 |
| Cadillac      |     3 |      127 |                 1     |                  1     |            1     |             1     |          3 |           3 |
| Chevrolet     |    22 |      884 |                 0.655 |                  0.674 |            0.658 |             0.673 |         22 |          22 |
| Chrysler      |     6 |      256 |                 0.992 |                  0.996 |            0.992 |             0.996 |          6 |           6 |
| Daewoo        |     1 |       45 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Dodge         |    13 |      618 |                 0.806 |                  0.787 |            0.755 |             0.796 |         13 |          13 |
| Eagle         |     1 |       46 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| FIAT          |     2 |       60 |                 0.75  |                  0.678 |            0.667 |             0.672 |          2 |           2 |
| Ferrari       |     4 |      159 |                 0.66  |                  0.669 |            0.656 |             0.669 |          4 |           4 |
| Fisker        |     1 |       43 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Ford          |    11 |      513 |                 0.998 |                  0.993 |            0.998 |             0.992 |         11 |          11 |
| GMC           |     5 |      234 |                 0.987 |                  0.99  |            0.991 |             0.99  |          5 |           5 |
| Geo           |     1 |       44 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| HUMMER        |     2 |       81 |                 0.963 |                  0.712 |            0.733 |             0.708 |          2 |           2 |
| Honda         |     3 |      160 |                 0.525 |                  0.512 |            0.512 |             0.512 |          3 |           3 |
| Hyundai       |    11 |      430 |                 0.802 |                  0.886 |            0.804 |             0.888 |         11 |          11 |
| Infiniti      |     2 |       66 |                 1     |                  1     |            1     |             1     |          2 |           2 |
| Isuzu         |     1 |       40 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Jaguar        |     1 |       46 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Jeep          |     5 |      218 |                 0.803 |                  0.798 |            0.803 |             0.798 |          5 |           5 |
| Lamborghini   |     4 |      158 |                 0.525 |                  0.652 |            0.568 |             0.648 |          4 |           4 |
| Land Rover    |     2 |       84 |                 1     |                  0.988 |            0.988 |             0.988 |          2 |           2 |
| Lincoln       |     1 |       39 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| MINI          |     1 |       36 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Maybach       |     1 |       29 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Mazda         |     1 |       36 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| McLaren       |     1 |       44 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Mercedes-Benz |     6 |      256 |                 0.797 |                  0.803 |            0.797 |             0.803 |          6 |           6 |
| Mitsubishi    |     1 |       47 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Nissan        |     4 |      170 |                 1     |                  1     |            1     |             1     |          4 |           4 |
| Plymouth      |     1 |       44 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Porsche       |     1 |       43 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Ram           |     1 |       40 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Rolls-Royce   |     3 |      110 |                 0.764 |                  0.764 |            0.761 |             0.758 |          3 |           3 |
| Scion         |     1 |       41 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Spyker        |     2 |       84 |                 0.631 |                  0.837 |            0.833 |             0.837 |          2 |           2 |
| Suzuki        |     4 |      166 |                 0.657 |                  0.771 |            0.753 |             0.767 |          4 |           4 |
| Tesla         |     1 |       38 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
| Toyota        |     4 |      161 |                 0.994 |                  1     |            1     |             1     |          4 |           4 |
| Volkswagen    |     2 |      131 |                 0.672 |                  0.672 |            0.672 |             0.672 |          2 |           2 |
| Volvo         |     3 |      129 |                 1     |                  1     |            1     |             1     |          3 |           3 |
| smart         |     1 |       40 |               nan     |                nan     |          nan     |           nan     |          1 |           1 |
