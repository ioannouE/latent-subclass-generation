# ksub (known-K subclass baseline): make, frozen dinov3_large, test

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `3209598aaa42a252272526b1498994c419ff56c4-dirty` from `configs/ksub_cars.yaml`; training git ['3209598aaa42a252272526b1498994c419ff56c4-dirty'], R1-R5 git ['3209598aaa42a252272526b1498994c419ff56c4-dirty'];
seeds [0, 1, 2]; dataset manifest `053a42cff21ab0f03c69b2c250be3e71672b9ba87740686f3e78d7fd75912fea`. **154 conflicting images ({'train': 59, 'val': 8, 'test': 87}) are excluded from every split**
(the `make` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_256/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features). Classes with K_c = 1 are left out of the clustering columns.

| run    | kNN coarse    | ACC oracle K   | NMI oracle K   | d0/d1         | d1/d2         | var ratio     | ARI boot      | ACC own       |
|:-------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| frozen | 0.973 ± 0.000 | 0.782 ± 0.000  | 0.703 ± 0.000  | 0.570 ± 0.000 | 0.709 ± 0.000 | 0.635 ± 0.000 | 0.886 ± 0.000 | 0.787 ± 0.003 |
| coarse | 0.988 ± 0.000 | 0.410 ± 0.008  | 0.132 ± 0.002  | 0.904 ± 0.000 | 0.057 ± 0.000 | 0.928 ± 0.000 | 0.424 ± 0.020 | 0.549 ± 0.010 |
| sub    | 0.988 ± 0.001 | 0.411 ± 0.002  | 0.136 ± 0.001  | 0.905 ± 0.001 | 0.052 ± 0.000 | 0.929 ± 0.001 | 0.435 ± 0.012 | 0.581 ± 0.012 |
| full   | 0.971 ± 0.001 | 0.543 ± 0.011  | 0.292 ± 0.007  | 0.936 ± 0.001 | 0.489 ± 0.001 | 0.951 ± 0.000 | 0.231 ± 0.006 | 0.460 ± 0.007 |

## Per class (mean over seeds; all metrics: `per_class.csv`)

| make          |   K_c |   n_test |   frozen ACC oracle K |   coarse ACC oracle K |   sub ACC oracle K |   full ACC oracle K |   frozen ACC own |   coarse ACC own |   sub ACC own |   full ACC own |
|:--------------|------:|---------:|----------------------:|----------------------:|-------------------:|--------------------:|-----------------:|-----------------:|--------------:|---------------:|
| AM General    |     1 |       44 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Acura         |     6 |      235 |                 0.749 |                 0.277 |              0.251 |               0.418 |            0.901 |            0.518 |         0.587 |          0.47  |
| Aston Martin  |     4 |      153 |                 0.497 |                 0.285 |              0.298 |               0.399 |            0.44  |            0.42  |         0.416 |          0.37  |
| Audi          |    13 |      551 |                 0.508 |                 0.313 |              0.319 |               0.515 |            0.498 |            0.37  |         0.377 |          0.306 |
| BMW           |    13 |      519 |                 0.609 |                 0.463 |              0.467 |               0.634 |            0.59  |            0.559 |         0.564 |          0.387 |
| Bentley       |     5 |      232 |                 0.625 |                 0.375 |              0.366 |               0.375 |            0.743 |            0.487 |         0.486 |          0.374 |
| Bugatti       |     2 |       70 |                 0.686 |                 0.586 |              0.586 |               0.59  |            0.686 |            0.619 |         0.557 |          0.586 |
| Buick         |     4 |      154 |                 0.818 |                 0.372 |              0.364 |               0.539 |            0.818 |            0.472 |         0.571 |          0.535 |
| Cadillac      |     3 |      127 |                 1     |                 0.378 |              0.381 |               0.651 |            1     |            0.675 |         0.698 |          0.606 |
| Chevrolet     |    22 |      884 |                 0.655 |                 0.36  |              0.352 |               0.584 |            0.658 |            0.27  |         0.319 |          0.342 |
| Chrysler      |     6 |      256 |                 0.992 |                 0.35  |              0.305 |               0.372 |            0.992 |            0.543 |         0.676 |          0.432 |
| Daewoo        |     1 |       45 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Dodge         |    13 |      618 |                 0.806 |                 0.399 |              0.405 |               0.53  |            0.755 |            0.451 |         0.526 |          0.286 |
| Eagle         |     1 |       46 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| FIAT          |     2 |       60 |                 0.75  |                 0.567 |              0.606 |               0.728 |            0.667 |            0.511 |         0.533 |          0.539 |
| Ferrari       |     4 |      159 |                 0.66  |                 0.346 |              0.39  |               0.474 |            0.656 |            0.56  |         0.618 |          0.434 |
| Fisker        |     1 |       43 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Ford          |    11 |      513 |                 0.998 |                 0.547 |              0.49  |               0.628 |            0.998 |            0.59  |         0.683 |          0.288 |
| GMC           |     5 |      234 |                 0.987 |                 0.325 |              0.323 |               0.434 |            0.991 |            0.567 |         0.578 |          0.447 |
| Geo           |     1 |       44 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| HUMMER        |     2 |       81 |                 0.963 |                 0.543 |              0.543 |               0.737 |            0.733 |            0.724 |         0.691 |          0.588 |
| Honda         |     3 |      160 |                 0.525 |                 0.504 |              0.508 |               0.456 |            0.512 |            0.392 |         0.413 |          0.404 |
| Hyundai       |    11 |      430 |                 0.802 |                 0.237 |              0.304 |               0.57  |            0.804 |            0.555 |         0.628 |          0.28  |
| Infiniti      |     2 |       66 |                 1     |                 0.53  |              0.53  |               0.763 |            1     |            0.525 |         0.662 |          0.586 |
| Isuzu         |     1 |       40 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Jaguar        |     1 |       46 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Jeep          |     5 |      218 |                 0.803 |                 0.372 |              0.393 |               0.431 |            0.803 |            0.683 |         0.711 |          0.365 |
| Lamborghini   |     4 |      158 |                 0.525 |                 0.287 |              0.308 |               0.456 |            0.568 |            0.483 |         0.496 |          0.441 |
| Land Rover    |     2 |       84 |                 1     |                 0.548 |              0.567 |               0.69  |            0.988 |            0.687 |         0.694 |          0.627 |
| Lincoln       |     1 |       39 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| MINI          |     1 |       36 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Maybach       |     1 |       29 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Mazda         |     1 |       36 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| McLaren       |     1 |       44 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Mercedes-Benz |     6 |      256 |                 0.797 |                 0.326 |              0.345 |               0.465 |            0.797 |            0.529 |         0.577 |          0.378 |
| Mitsubishi    |     1 |       47 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Nissan        |     4 |      170 |                 1     |                 0.3   |              0.296 |               0.465 |            1     |            0.514 |         0.606 |          0.51  |
| Plymouth      |     1 |       44 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Porsche       |     1 |       43 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Ram           |     1 |       40 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Rolls-Royce   |     3 |      110 |                 0.764 |                 0.4   |              0.397 |               0.585 |            0.761 |            0.639 |         0.655 |          0.497 |
| Scion         |     1 |       41 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Spyker        |     2 |       84 |                 0.631 |                 0.56  |              0.552 |               0.571 |            0.833 |            0.817 |         0.79  |          0.591 |
| Suzuki        |     4 |      166 |                 0.657 |                 0.297 |              0.279 |               0.494 |            0.753 |            0.448 |         0.482 |          0.51  |
| Tesla         |     1 |       38 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
| Toyota        |     4 |      161 |                 0.994 |                 0.371 |              0.362 |               0.48  |            1     |            0.609 |         0.675 |          0.46  |
| Volkswagen    |     2 |      131 |                 0.672 |                 0.687 |              0.687 |               0.656 |            0.672 |            0.664 |         0.59  |          0.545 |
| Volvo         |     3 |      129 |                 1     |                 0.395 |              0.349 |               0.589 |            1     |            0.579 |         0.556 |          0.61  |
| smart         |     1 |       40 |               nan     |               nan     |            nan     |             nan     |          nan     |          nan     |       nan     |        nan     |
