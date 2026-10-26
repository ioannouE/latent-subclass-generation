# Representation metrics on CIFAR-100: frozen encoders, fine-tuned baselines and ksub

**Setup**
- Label scheme `cifar100`: coarse = 20 superclasses, hidden subclass = 100 classes, K_c = 5 in every superclass.
- Fit / reference set = `train` (44,932 images), scored on `test` (9,983); native 32 px images.
- **Conflicting images are excluded everywhere:** `data/cifar100/splits/exclude.txt`, 90 images: {'train': 68, 'val': 5, 'test': 17}; see docs/DATA_CIFAR100.md.
- Values are macro means over the 20 superclasses. Per-class tables and 95% bootstrap CIs: `repr/<encoder>_32/`, `ksub/<run>_32/`, `ksub_b1280/<run>_32/`. Metric definitions: docs/METRICS.md.

**Provenance**
- Dataset manifest `1cf015739e9b592dcee9dfbf0050118bced8343b8aa77ea080a6f27d1c394038`; the git hash and config of every row are in its `summary.json`.
- Configs:
  - frozen encoders: `configs/eval_repr_cifar100.yaml`;
  - baselines: `configs/repr_baselines_cifar100.yaml`, `configs/repr_george_cifar100.yaml`, scored with `configs/eval_repr_cifar100_baselines.yaml`;
  - ksub: `configs/ksub_cifar100{,_b1280}.yaml`, scored with `configs/eval_repr_ksub_cifar100{,_b1280}.yaml`.
- Frozen rows and baselines: seed 0 (baselines fine-tune DINOv3-B). ksub rows: mean ± std over seeds 0-2, head on frozen DINOv3-L.

## What R1-R5 assess
| | Question | Columns below |
|---|---|---|
| **R1** coarse | Are the given (coarse) classes recoverable? kNN (k = 20) and a linear probe fitted on train | kNN coarse, probe coarse |
| **R2** subclasses | Is the hidden subclass recoverable inside each coarse class? Supervised (a linear probe on the fine labels inside the class; Recall@1 = is the nearest same-class image of the same subclass), and unsupervised (k-means on the class's top-50 PCs with the true K = K_c, "uses K", or with the label-free K-hat from a silhouette / gap rule; ACC by Hungarian matching) | fine probe, Recall@1, ACC / NMI oracle K, ACC / ARI K-hat, K-hat |
| **R3** hierarchy geometry | Is the hierarchy nested? d0 = squared distance between same-subclass pairs, d1 = same class / other subclass, d2 = other class; d0/d1 < 1 and d1/d2 < 1 are desired | d0/d1, d1/d2 |
| **R4** within-subclass variation | How much spread is left inside a subclass, relative to its class (within-subclass trace covariance / class trace covariance). Lower means tighter subclasses; near 1 means the class is unstructured | var ratio |
| **R5** stability | Is the oracle-K clustering stable? ARI between the full-data clustering and fits on bootstrap resamples | ARI boot |
| ksub only | Accuracy of the method's own final subclass assignment (Hungarian ACC; for frozen DINOv3-L: nearest k-means centroid of the train features) | ACC own |

## R1-R5
| encoder | kNN coarse | probe coarse | fine probe | Recall@1 | ACC oracle K | NMI oracle K | ACC K-hat | ARI K-hat | K-hat | d0/d1 | d1/d2 | var ratio | ARI boot | ACC own |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen DINOv3-B | 0.943 | 0.938 | 0.946 | 0.894 | 0.849 | 0.727 | 0.738 | 0.636 | 6.1 | 0.790 | 0.916 | 0.824 | 0.857 | - |
| frozen DINOv3-L | 0.964 | 0.966 | 0.961 | 0.923 | 0.892 | 0.800 | 0.778 | 0.706 | 5.8 | 0.733 | 0.921 | 0.773 | 0.891 | 0.902 |
| frozen CLIP-L | 0.874 | 0.910 | 0.914 | 0.782 | 0.547 | 0.365 | 0.426 | 0.190 | 3.4 | 0.835 | 0.872 | 0.863 | 0.659 | - |
| frozen SSCD | 0.608 | 0.613 | 0.763 | 0.613 | 0.461 | 0.238 | 0.401 | 0.154 | 4.0 | 0.954 | 0.967 | 0.963 | 0.525 | - |
| SupCon | 0.966 | 0.963 | 0.953 | 0.919 | 0.879 | 0.778 | 0.772 | 0.677 | 4.7 | 0.733 | 0.900 | 0.773 | 0.897 | - |
| SimCLR | 0.893 | 0.923 | 0.928 | 0.828 | 0.697 | 0.529 | 0.630 | 0.457 | 5.7 | 0.829 | 0.918 | 0.858 | 0.729 | - |
| SupCon + SimCLR | 0.959 | 0.959 | 0.948 | 0.903 | 0.856 | 0.738 | 0.773 | 0.659 | 4.8 | 0.788 | 0.924 | 0.822 | 0.846 | - |
| MaskCon | 0.965 | 0.960 | 0.952 | 0.923 | 0.884 | 0.789 | 0.775 | 0.682 | 4.7 | 0.723 | 0.916 | 0.764 | 0.902 | - |
| FALCON (uses K) | 0.967 | 0.966 | 0.951 | 0.918 | 0.876 | 0.781 | 0.788 | 0.696 | 4.8 | 0.694 | 0.861 | 0.737 | 0.902 | - |
| GEORGE | 0.966 | 0.964 | 0.950 | 0.913 | 0.862 | 0.763 | 0.761 | 0.666 | 4.7 | 0.705 | 0.786 | 0.747 | 0.888 | - |
| GEORGE stage 1 (ERM) | 0.966 | 0.964 | 0.949 | 0.912 | 0.860 | 0.767 | 0.766 | 0.668 | 4.5 | 0.697 | 0.763 | 0.740 | 0.902 | - |
| LeJEPA (no labels) | 0.946 | 0.952 | 0.946 | 0.896 | 0.838 | 0.716 | 0.733 | 0.624 | 5.8 | 0.733 | 0.927 | 0.773 | 0.869 | - |
| ksub `coarse`, batch 320 | 0.970 ± 0.000 | 0.970 ± 0.000 | 0.797 ± 0.004 | 0.731 ± 0.008 | 0.224 ± 0.001 | 0.038 ± 0.001 | 0.217 ± 0.000 | 0.001 ± 0.000 | 2.4 | 0.974 ± 0.000 | 0.063 ± 0.001 | 0.979 ± 0.000 | 0.845 ± 0.007 | 0.229 ± 0.003 |
| ksub `sub`, batch 320 | 0.970 ± 0.000 | 0.969 ± 0.001 | 0.783 ± 0.004 | 0.731 ± 0.005 | 0.224 ± 0.000 | 0.039 ± 0.001 | 0.217 ± 0.001 | 0.001 ± 0.000 | 2.3 | 0.974 ± 0.000 | 0.062 ± 0.000 | 0.979 ± 0.000 | 0.850 ± 0.002 | 0.230 ± 0.002 |
| ksub `full`, batch 320 | 0.971 ± 0.001 | 0.971 ± 0.001 | 0.818 ± 0.002 | 0.764 ± 0.004 | 0.269 ± 0.005 | 0.045 ± 0.004 | 0.231 ± 0.002 | 0.004 ± 0.000 | 2.0 | 0.992 ± 0.000 | 0.556 ± 0.000 | 0.994 ± 0.000 | 0.212 ± 0.009 | 0.233 ± 0.002 |
| ksub `sub_only`, batch 320 | 0.865 ± 0.019 | 0.579 ± 0.161 | 0.790 ± 0.019 | 0.850 ± 0.015 | 0.575 ± 0.117 | 0.410 ± 0.111 | 0.460 ± 0.102 | 0.255 ± 0.095 | 3.5 | 0.662 ± 0.079 | 0.452 ± 0.371 | 0.698 ± 0.065 | 0.738 ± 0.064 | 0.249 ± 0.017 |
| ksub `sub_var`, batch 320 | 0.960 ± 0.001 | 0.338 ± 0.008 | 0.948 ± 0.002 | 0.927 ± 0.002 | 0.917 ± 0.002 | 0.833 ± 0.003 | 0.917 ± 0.001 | 0.831 ± 0.003 | 5.0 | 0.649 ± 0.001 | 1.074 ± 0.000 | 0.697 ± 0.001 | 0.994 ± 0.001 | 0.917 ± 0.002 |
| ksub `sub_var_coarse01`, batch 320 | 0.968 ± 0.001 | 0.951 ± 0.002 | 0.949 ± 0.001 | 0.928 ± 0.002 | 0.919 ± 0.001 | 0.840 ± 0.004 | 0.918 ± 0.002 | 0.836 ± 0.001 | 5.0 | 0.683 ± 0.000 | 0.998 ± 0.000 | 0.728 ± 0.000 | 0.990 ± 0.002 | 0.918 ± 0.001 |
| ksub `sub_var`, batch 1280 | 0.954 ± 0.002 | 0.478 ± 0.006 | 0.939 ± 0.003 | 0.923 ± 0.003 | 0.912 ± 0.002 | 0.831 ± 0.004 | 0.912 ± 0.002 | 0.826 ± 0.004 | 5.0 | 0.691 ± 0.001 | 1.062 ± 0.000 | 0.735 ± 0.001 | 0.992 ± 0.002 | 0.912 ± 0.002 |
| ksub `sub_var_coarse01`, batch 1280 | 0.967 ± 0.001 | 0.909 ± 0.001 | 0.936 ± 0.003 | 0.923 ± 0.002 | 0.912 ± 0.001 | 0.834 ± 0.003 | 0.911 ± 0.001 | 0.825 ± 0.002 | 5.0 | 0.725 ± 0.000 | 0.994 ± 0.000 | 0.766 ± 0.000 | 0.988 ± 0.002 | 0.911 ± 0.001 |

ksub runs: `coarse` = L_coarse, `sub` = L_coarse + L_sub, `full` = L_coarse + L_sub + L_var, `sub_only` = L_sub, `sub_var` = L_sub + L_var,
`sub_var_coarse01` = 0.1 L_coarse + L_sub + L_var (weight fixed in advance). All are trained with the true K_c ("uses K"), so their K-hat columns apply the
label-free rule to an embedding trained knowing K. Batch = P classes x m images per class; the larger batch keeps the number of optimiser steps.

## Reading
- **Frozen encoders and baselines:** DINOv3-L is the best frozen encoder. The fine-tuned baselines (on DINOv3-B) gain coarse accuracy but no
  subclass accuracy over it: the best baseline ACC oracle K is MaskCon's 0.884 vs frozen DINOv3-L 0.892.
- **ksub with full-weight L_coarse (`coarse`, `sub`, `full`):** every superclass collapses. Subclass ACC is at chance (about 0.22 for 5 classes).
- **ksub `sub_var_coarse01`, batch 320: the best row of the table.**
  - Subclasses: ACC oracle K 0.919 (frozen DINOv3-L 0.892, best baseline 0.884); ACC K-hat 0.918 (best other row: FALCON 0.788), with K-hat = 5.0.
  - Own assignment: ACC own 0.918. Stability: ARI boot 0.990.
  - Coarse classes: probe coarse 0.951, kNN coarse 0.968.
  - Costs: fine probe 0.949 vs 0.961; d1/d2 about 1, so superclasses are separable but not further apart than their subclasses.
- **`sub_var`** has the same subclass numbers but loses the linear coarse structure (probe coarse 0.338).
- **Batch 1280 instead of 320:** slightly worse on every subclass metric (-0.005 to -0.007 ACC) and on probe coarse (0.909 vs 0.951). CIFAR-100
  already has about 3 images per subclass in a batch of 320, so a larger batch adds nothing here.
- Std over seeds is <= 0.004 for every `sub_var*` row.
