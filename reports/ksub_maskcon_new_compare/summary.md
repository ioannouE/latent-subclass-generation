# ksub per class and per subclass: maskcon_new, test

Written at git `1862299774c62c59037631a8482aa74625b608a5-dirty` from `configs/ksub_compare_maskcon_new.yaml`. **211 conflicting images ({'train': 98, 'val': 12, 'test': 101}) are excluded from every
split.** Runs (config, method): `frozen_K` = frozen in `configs/ksub_maskcon_new_recluster.yaml`, `ema_sub_var` = sub_var in `configs/ksub_maskcon_new.yaml`, `ema_b1024_sub_var` = sub_var in `configs/ksub_maskcon_new_b1024.yaml`, `ema_b1024_sub_var_coarse01` = sub_var_coarse01 in `configs/ksub_maskcon_new_b1024.yaml`, `recluster_sub_var` = sub_var in `configs/ksub_maskcon_new_recluster.yaml`, `recluster_sub_var_coarse01` = sub_var_coarse01 in `configs/ksub_maskcon_new_recluster.yaml`, `frozen_khat` = frozen in `configs/ksub_maskcon_new_khat.yaml`, `khat_sub_var` = sub_var in `configs/ksub_maskcon_new_khat.yaml`, `khat_sub_var_coarse01` = sub_var_coarse01 in `configs/ksub_maskcon_new_khat.yaml`. Means over seeds.
ACC oracle K = k-means with the true K_c on z (evaluation only); ACC own and K = the run's own clustering and its number of clusters;
R@1 within = within-class fine Recall@1. Per subclass: matched = share of its test images in its Hungarian-matched own cluster (0 if
unmatched); failed = matched < 0.5; merged = its largest cluster is also the largest of another subclass (counts are means over seeds).

## Per coarse class

### ACC oracle K

| coarse_name   |   K_c |   frozen_K |   ema_sub_var |   ema_b1024_sub_var |   ema_b1024_sub_var_coarse01 |   recluster_sub_var |   recluster_sub_var_coarse01 |   frozen_khat |   khat_sub_var |   khat_sub_var_coarse01 |
|:--------------|------:|-----------:|--------------:|--------------------:|-----------------------------:|--------------------:|-----------------------------:|--------------:|---------------:|------------------------:|
| Cab           |    18 |      0.718 |         0.649 |               0.695 |                        0.673 |               0.783 |                        0.773 |         0.718 |          0.527 |                   0.526 |
| Sedan         |    50 |      0.732 |         0.451 |               0.744 |                        0.75  |               0.763 |                        0.75  |         0.732 |          0.749 |                   0.762 |
| SUV           |    36 |      0.821 |         0.479 |               0.857 |                        0.867 |               0.852 |                        0.867 |         0.821 |          0.844 |                   0.844 |
| Convertible   |    30 |      0.838 |         0.541 |               0.802 |                        0.8   |               0.791 |                        0.802 |         0.838 |          0.787 |                   0.788 |
| Coupe         |    29 |      0.767 |         0.603 |               0.784 |                        0.787 |               0.802 |                        0.789 |         0.767 |          0.766 |                   0.763 |
| Hatchback     |    14 |      0.993 |         0.952 |               0.933 |                        0.929 |               0.963 |                        0.924 |         0.993 |          0.975 |                   0.98  |
| Wagon         |     6 |      0.846 |         0.849 |               0.853 |                        0.795 |               0.84  |                        0.853 |         0.846 |          0.806 |                   0.798 |
| Van           |    13 |      0.683 |         0.704 |               0.701 |                        0.706 |               0.688 |                        0.724 |         0.683 |          0.66  |                   0.679 |
### ACC own

| coarse_name   |   K_c |   frozen_K |   ema_sub_var |   ema_b1024_sub_var |   ema_b1024_sub_var_coarse01 |   recluster_sub_var |   recluster_sub_var_coarse01 |   frozen_khat |   khat_sub_var |   khat_sub_var_coarse01 |
|:--------------|------:|-----------:|--------------:|--------------------:|-----------------------------:|--------------------:|-----------------------------:|--------------:|---------------:|------------------------:|
| Cab           |    18 |      0.709 |         0.594 |               0.686 |                        0.666 |               0.784 |                        0.779 |         0.358 |          0.363 |                   0.359 |
| Sedan         |    50 |      0.738 |         0.307 |               0.732 |                        0.733 |               0.748 |                        0.776 |         0.723 |          0.735 |                   0.734 |
| SUV           |    36 |      0.845 |         0.392 |               0.856 |                        0.854 |               0.856 |                        0.855 |         0.842 |          0.842 |                   0.851 |
| Convertible   |    30 |      0.795 |         0.54  |               0.803 |                        0.8   |               0.802 |                        0.802 |         0.778 |          0.785 |                   0.773 |
| Coupe         |    29 |      0.768 |         0.534 |               0.789 |                        0.78  |               0.779 |                        0.783 |         0.736 |          0.714 |                   0.719 |
| Hatchback     |    14 |      0.936 |         0.952 |               0.932 |                        0.93  |               0.964 |                        0.924 |         0.799 |          0.788 |                   0.786 |
| Wagon         |     6 |      0.837 |         0.851 |               0.853 |                        0.852 |               0.844 |                        0.851 |         0.83  |          0.827 |                   0.842 |
| Van           |    13 |      0.709 |         0.709 |               0.708 |                        0.707 |               0.705 |                        0.719 |         0.686 |          0.681 |                   0.703 |
### K

| coarse_name   |   K_c |   frozen_K |   ema_sub_var |   ema_b1024_sub_var |   ema_b1024_sub_var_coarse01 |   recluster_sub_var |   recluster_sub_var_coarse01 |   frozen_khat |   khat_sub_var |   khat_sub_var_coarse01 |
|:--------------|------:|-----------:|--------------:|--------------------:|-----------------------------:|--------------------:|-----------------------------:|--------------:|---------------:|------------------------:|
| Cab           |    18 |         18 |            18 |                  18 |                           18 |                  18 |                           18 |         6.667 |          6.667 |                   6.667 |
| Sedan         |    50 |         50 |            50 |                  50 |                           50 |                  50 |                           50 |        47.333 |         47.333 |                  47.333 |
| SUV           |    36 |         36 |            36 |                  36 |                           36 |                  36 |                           36 |        36.333 |         36.333 |                  36.333 |
| Convertible   |    30 |         30 |            30 |                  30 |                           30 |                  30 |                           30 |        34     |         34     |                  34     |
| Coupe         |    29 |         29 |            29 |                  29 |                           29 |                  29 |                           29 |        38.333 |         38.333 |                  38.333 |
| Hatchback     |    14 |         14 |            14 |                  14 |                           14 |                  14 |                           14 |        22     |         22     |                  22     |
| Wagon         |     6 |          6 |             6 |                   6 |                            6 |                   6 |                            6 |         5     |          5     |                   5     |
| Van           |    13 |         13 |            13 |                  13 |                           13 |                  13 |                           13 |        11     |         11     |                  11     |
### R@1 within

| coarse_name   |   K_c |   frozen_K |   ema_sub_var |   ema_b1024_sub_var |   ema_b1024_sub_var_coarse01 |   recluster_sub_var |   recluster_sub_var_coarse01 |   frozen_khat |   khat_sub_var |   khat_sub_var_coarse01 |
|:--------------|------:|-----------:|--------------:|--------------------:|-----------------------------:|--------------------:|-----------------------------:|--------------:|---------------:|------------------------:|
| Cab           |    18 |      0.872 |         0.84  |               0.841 |                        0.838 |               0.868 |                        0.865 |         0.872 |          0.834 |                   0.833 |
| Sedan         |    50 |      0.971 |         0.951 |               0.954 |                        0.95  |               0.966 |                        0.965 |         0.971 |          0.966 |                   0.968 |
| SUV           |    36 |      0.979 |         0.955 |               0.961 |                        0.965 |               0.969 |                        0.97  |         0.979 |          0.97  |                   0.972 |
| Convertible   |    30 |      0.956 |         0.945 |               0.925 |                        0.929 |               0.956 |                        0.95  |         0.956 |          0.952 |                   0.95  |
| Coupe         |    29 |      0.954 |         0.927 |               0.927 |                        0.929 |               0.946 |                        0.942 |         0.954 |          0.945 |                   0.942 |
| Hatchback     |    14 |      0.996 |         0.996 |               0.996 |                        0.995 |               0.996 |                        0.996 |         0.996 |          0.996 |                   0.996 |
| Wagon         |     6 |      0.892 |         0.889 |               0.889 |                        0.896 |               0.906 |                        0.895 |         0.892 |          0.892 |                   0.891 |
| Van           |    13 |      0.863 |         0.839 |               0.818 |                        0.837 |               0.836 |                        0.84  |         0.863 |          0.839 |                   0.843 |
### d1/d2

| coarse_name   |   K_c |   frozen_K |   ema_sub_var |   ema_b1024_sub_var |   ema_b1024_sub_var_coarse01 |   recluster_sub_var |   recluster_sub_var_coarse01 |   frozen_khat |   khat_sub_var |   khat_sub_var_coarse01 |
|:--------------|------:|-----------:|--------------:|--------------------:|-----------------------------:|--------------------:|-----------------------------:|--------------:|---------------:|------------------------:|
| Cab           |    18 |      0.692 |         1.025 |               1.011 |                        0.837 |               1.03  |                        0.853 |         0.692 |          0.997 |                   0.759 |
| Sedan         |    50 |      0.914 |         1.012 |               1.01  |                        1.001 |               1.012 |                        0.943 |         0.914 |          1.012 |                   0.958 |
| SUV           |    36 |      0.873 |         1.012 |               1.015 |                        0.971 |               1.019 |                        0.936 |         0.873 |          1.018 |                   0.946 |
| Convertible   |    30 |      0.919 |         1.016 |               1.014 |                        0.984 |               1.02  |                        0.968 |         0.919 |          1.02  |                   0.965 |
| Coupe         |    29 |      0.938 |         1.018 |               1.015 |                        0.982 |               1.021 |                        0.975 |         0.938 |          1.022 |                   0.957 |
| Hatchback     |    14 |      0.947 |         1.039 |               1.036 |                        0.925 |               1.041 |                        0.988 |         0.947 |          1.043 |                   0.966 |
| Wagon         |     6 |      0.908 |         1.074 |               1.08  |                        0.833 |               1.074 |                        0.892 |         0.908 |          1.078 |                   0.888 |
| Van           |    13 |      0.823 |         1.038 |               1.025 |                        0.814 |               1.039 |                        0.857 |         0.823 |          1.038 |                   0.854 |
### ARI boot

| coarse_name   |   K_c |   frozen_K |   ema_sub_var |   ema_b1024_sub_var |   ema_b1024_sub_var_coarse01 |   recluster_sub_var |   recluster_sub_var_coarse01 |   frozen_khat |   khat_sub_var |   khat_sub_var_coarse01 |
|:--------------|------:|-----------:|--------------:|--------------------:|-----------------------------:|--------------------:|-----------------------------:|--------------:|---------------:|------------------------:|
| Cab           |    18 |      0.816 |         0.734 |               0.944 |                        0.83  |               0.958 |                        0.876 |         0.816 |          0.484 |                   0.468 |
| Sedan         |    50 |      0.819 |         0.732 |               0.922 |                        0.927 |               0.835 |                        0.867 |         0.819 |          0.861 |                   0.85  |
| SUV           |    36 |      0.907 |         0.724 |               0.977 |                        0.983 |               0.92  |                        0.93  |         0.907 |          0.935 |                   0.929 |
| Convertible   |    30 |      0.883 |         0.7   |               0.98  |                        0.977 |               0.95  |                        0.938 |         0.883 |          0.941 |                   0.923 |
| Coupe         |    29 |      0.9   |         0.696 |               0.979 |                        0.973 |               0.924 |                        0.902 |         0.9   |          0.926 |                   0.913 |
| Hatchback     |    14 |      0.97  |         0.997 |               0.997 |                        0.997 |               0.993 |                        0.986 |         0.97  |          0.983 |                   0.989 |
| Wagon         |     6 |      0.817 |         0.997 |               0.998 |                        0.95  |               0.997 |                        0.995 |         0.817 |          0.952 |                   0.943 |
| Van           |    13 |      0.884 |         0.862 |               0.987 |                        0.986 |               0.948 |                        0.919 |         0.884 |          0.878 |                   0.877 |

## Subclasses per coarse class: failed (matched < 0.5)

| coarse_name   |   frozen_K |   ema_sub_var |   ema_b1024_sub_var |   ema_b1024_sub_var_coarse01 |   recluster_sub_var |   recluster_sub_var_coarse01 |   frozen_khat |   khat_sub_var |   khat_sub_var_coarse01 |
|:--------------|-----------:|--------------:|--------------------:|-----------------------------:|--------------------:|-----------------------------:|--------------:|---------------:|------------------------:|
| Cab           |          3 |             7 |                   4 |                            5 |                   3 |                            3 |            14 |             14 |                      14 |
| Sedan         |          9 |            38 |                  10 |                           10 |                   7 |                            7 |            11 |              8 |                       8 |
| SUV           |          4 |            25 |                   4 |                            4 |                   3 |                            3 |             3 |              2 |                       1 |
| Convertible   |          6 |            13 |                   6 |                            6 |                   6 |                            5 |             5 |              6 |                       6 |
| Coupe         |          5 |            14 |                   5 |                            5 |                   5 |                            5 |             5 |              5 |                       4 |
| Hatchback     |          1 |             0 |                   1 |                            1 |                   0 |                            1 |             0 |              0 |                       0 |
| Wagon         |          1 |             1 |                   1 |                            1 |                   1 |                            1 |             1 |              1 |                       1 |
| Van           |          4 |             3 |                   4 |                            4 |                   4 |                            4 |             3 |              4 |                       4 |

## Subclasses per coarse class: merged

| coarse_name   |   frozen_K |   ema_sub_var |   ema_b1024_sub_var |   ema_b1024_sub_var_coarse01 |   recluster_sub_var |   recluster_sub_var_coarse01 |   frozen_khat |   khat_sub_var |   khat_sub_var_coarse01 |
|:--------------|-----------:|--------------:|--------------------:|-----------------------------:|--------------------:|-----------------------------:|--------------:|---------------:|------------------------:|
| Cab           |        6.3 |          10.7 |                 7.3 |                          8   |                 3.7 |                          4   |          15.3 |           15.3 |                    15.3 |
| Sedan         |       18.7 |          50   |                19   |                         19.7 |                13.7 |                         12.3 |          22.7 |           16   |                    19.3 |
| SUV           |        7.7 |          32.7 |                 7.7 |                          7.7 |                 5.3 |                          6.7 |           7.3 |            6.7 |                     5.3 |
| Convertible   |        8.3 |          19.7 |                 8.7 |                          8.3 |                 7.3 |                          9   |           7.3 |            7   |                     7.3 |
| Coupe         |       11.3 |          21.7 |                11.3 |                         11.3 |                10   |                          9.3 |           8.7 |            8   |                     7.3 |
| Hatchback     |        2   |           1.3 |                 2   |                          2   |                 0.7 |                          2   |           0   |            0   |                     0   |
| Wagon         |        2   |           2   |                 2   |                          2   |                 2   |                          2   |           2   |            2   |                     2   |
| Van           |        6.3 |           6.3 |                 6.3 |                          6.3 |                 7   |                          4.7 |           7.7 |            6.3 |                     6.3 |

## 25 worst subclasses of `khat_sub_var` (all subclasses: `subclasses.csv`)

| coarse_name   | fine_name                                     |   n |   matched |   merged |
|:--------------|:----------------------------------------------|----:|----------:|---------:|
| Cab           | Chevrolet Silverado 1500 Extended Cab 2012    |  43 |     0     |    1     |
| Cab           | Chevrolet Silverado 1500 Regular Cab 2012     |  41 |     0     |    1     |
| Cab           | Dodge Dakota Club Cab 2007                    |  38 |     0     |    1     |
| Cab           | Ford F-150 Regular Cab 2012                   |  42 |     0     |    1     |
| Cab           | HUMMER H3T Crew Cab 2010                      |  39 |     0     |    1     |
| SUV           | Chevrolet Tahoe Hybrid SUV 2012               |  37 |     0     |    1     |
| Wagon         | Dodge Caliber Wagon 2012                      |  35 |     0     |    1     |
| Van           | Chevrolet Express Cargo Van 2007              |  26 |     0     |    1     |
| Sedan         | Bentley Mulsanne Sedan 2011                   |  35 |     0.133 |    1     |
| Coupe         | Audi TT RS Coupe 2012                         |  35 |     0.133 |    1     |
| Cab           | Chevrolet Silverado 2500HD Regular Cab 2012   |  38 |     0.158 |    0.667 |
| Sedan         | BMW ActiveHybrid 5 Sedan 2012                 |  34 |     0.206 |    1     |
| Sedan         | Buick Regal GS 2012                           |  35 |     0.21  |    1     |
| Sedan         | Audi S6 Sedan 2011                            |  46 |     0.217 |    0.667 |
| Coupe         | Audi S5 Coupe 2012                            |  40 |     0.225 |    1     |
| Coupe         | Bentley Continental GT Coupe 2007             |  43 |     0.256 |    1     |
| Van           | Ram C/V Cargo Van Minivan 2012                |  40 |     0.258 |    1     |
| Sedan         | Hyundai Accent Sedan 2012                     |  24 |     0.278 |    0.667 |
| Sedan         | Rolls-Royce Ghost Sedan 2012                  |  37 |     0.279 |    0.667 |
| Cab           | Chevrolet Silverado 1500 Hybrid Crew Cab 2012 |  39 |     0.291 |    1     |
| Van           | Chevrolet Express Van 2007                    |  33 |     0.303 |    1     |
| Convertible   | Chevrolet Corvette Convertible 2012           |  39 |     0.308 |    1     |
| Cab           | Dodge Dakota Crew Cab 2010                    |  39 |     0.325 |    1     |
| Cab           | GMC Canyon Extended Cab 2012                  |  39 |     0.325 |    0.667 |
| Cab           | Dodge Ram Pickup 3500 Crew Cab 2010           |  42 |     0.325 |    0.667 |
