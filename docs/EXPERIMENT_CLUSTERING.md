# Clustering experiment: Leiden and diffusion maps in place of k-means on frozen DINOv3

Question: frozen DINOv3 separates the coarse classes well and the fine classes reasonably (fine probe 0.95, oracle-K ACC 0.73 / 0.80 for B / L), and no
fine-tuned baseline beats it on subclass recovery (`reports/repr_maskcon_new/summary.md`). Is the bottleneck the clustering algorithm rather than the
embedding? We keep the embedding fixed and replace the k-means step of R2 / R5 with two graph-based algorithms.

Label scheme `maskcon_new` (configs/labels.yaml: coarse = 8 body types, hidden subclass = the 196 official classes, true K_c 6-50, mean 24.5). The coarse
labels are known, so clustering runs within each body type and targets the fine classes, as in R2. The scheme's conflicting images
(`data/splits/exclude_maskcon_new.txt`, 211 images, docs/DATA.md) are left out of every run: 7,231 train / 7,940 test. Fine labels are used in evaluation only.

## Algorithms
Both run on the R2 feature space (each body type's L2-normalised embeddings projected on its own top-50 PCs, `cluster_space`), so only the clustering
changes. Both use the same graph: symmetric 15-NN graph with self-tuning Gaussian weights exp(-d_ij^2 / (s_i s_j)), s_i = distance to the 15th
neighbour (`knn_graph`, `lsgen/eval/clustering.py`).
- **Leiden** (Traag, Waltman & van Eck, Sci. Rep. 2019; igraph's implementation): maximises modularity with a resolution parameter and guarantees
  connected communities. Label-free K-hat = number of communities at resolution 1 (standard modularity; no K range, so it is not capped at 25).
  Oracle K: bisection of the resolution (log10 in [-3, 3]) to the partition with K_c communities (the closest if none has exactly K_c).
- **Diffusion maps + k-means** (Coifman & Lafon 2006): random walk P = D^-1 A on the graph, diffusion coordinates lambda_i^t psi_i (t = 1) of the
  non-trivial right eigenvectors, k-means (10 inits) on the first K - 1 of them. Label-free K-hat = largest eigengap lambda_K - lambda_{K+1}, K = 2..25
  (the range of the k-means K-hat rule).

All parameters (15 neighbours, resolution 1, t = 1) are standard defaults fixed before any run and not tuned: no choice is made in this experiment, so
`val` is not used and `test` is scored once.

## Metrics
R1-R5 as in docs/METRICS.md, fit/reference = `train`, scored on `test`. R1 (coarse kNN / probe), R2 within-make (Recall@1, fine probe), R3 and R4 are
properties of the embedding and do not involve clustering, so they are those of the frozen rows. What changes:
- **R2** oracle K (uses K) and label-free K-hat: ACC (Hungarian), NMI, ARI, K-hat, with the same bootstrap / 63.2% subsample CIs.
- **R5** ari_seeds (3 seeds) and ari_boot (20 bootstrap fits, oracle K). Difference from k-means: a graph partition has no out-of-sample assignment,
  so each bootstrap fit runs on the distinct images of the resample and is compared with the full-data partition on those images.

Rule fixed before the comparison (as for geosub): an algorithm improves on k-means if its macro K-hat ACC is > +0.03 above k-means K-hat on the same
frozen embedding (0.659 B, 0.660 L). Oracle-K ACC is reported to separate "better K selection" from "better partition at known K".

## Code
- `lsgen/eval/clustering.py`: `knn_graph`, `leiden_fit`, `diffusion_fit`; each `fit(z, k, seed)` returns labels (k = None: label-free).
- `lsgen/eval/representation.py`: `r2_metric`, `r5_make`, `khat_error` take `fit` (default k-means, so every earlier result is unchanged).
- `scripts/eval_representation.py`: a `clustering:` entry in the config selects the algorithm; configs `configs/eval_repr_maskcon_new_{leiden,diffusion}.yaml`
  (tables r2_oracle, r2_khat, r5; DINOv3-B and L at 256 px). Tests: `tests/test_clustering.py`.
- New dependency `igraph==0.11.8` (`pip install -e ".[dev]"` again).

## Commands
```
pip install -e ".[dev]"                                                  # once, in the diffusion env (adds igraph)
python -m pytest -q tests/test_clustering.py tests/test_representation.py
CONFIG=configs/eval_repr_maskcon_new_leiden.yaml sbatch slurm/eval_representation.sh
CONFIG=configs/eval_repr_maskcon_new_diffusion.yaml sbatch slurm/eval_representation.sh
```
Output: `<reports_root>/repr_maskcon_new_{leiden,diffusion}/dinov3_{base,large}_256/{r2_oracle,r2_khat,r5}.csv` + summary.json.

## Results (test, single seed 0, 256 px, maskcon_new; conflicting images excluded: 7,231 train / 7,940 test)
Macro over the 8 body types. All rows except "+ diffusion" use k-means for R2 / R5 (`reports/repr_maskcon_new/`); the diffusion rows' R1 / R3 / R4
are the frozen encoder's (`reports/repr_maskcon_new_diffusion/`). oK = oracle K (uses K), K^ = label-free K-hat. Leiden: pending (job running).

| Method | kNN coarse | probe coarse | Recall@1 | fine probe | ACC oK | NMI oK | ARI oK | ACC K^ | NMI K^ | ARI K^ | K^ | d0/d1 | d1/d2 | var_ratio | ARI seeds | ARI boot |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen DINOv3-B (k-means) | 0.932 | 0.953 | 0.934 | 0.950 | 0.727 | 0.830 | 0.658 | 0.659 | 0.807 | 0.601 | 20.4 | 0.507 | 0.858 | 0.521 | 0.865 | 0.829 |
| frozen DINOv3-B + diffusion | 0.932 | 0.953 | 0.934 | 0.950 | **0.773** | 0.846 | 0.669 | 0.654 | 0.794 | 0.518 | 19.1 | 0.507 | 0.858 | 0.521 | 0.980 | 0.845 |
| frozen DINOv3-L (k-means) | 0.944 | 0.973 | 0.935 | 0.950 | 0.800 | 0.886 | 0.759 | 0.660 | 0.848 | 0.641 | 17.0 | 0.415 | 0.877 | 0.429 | 0.917 | 0.874 |
| frozen DINOv3-L + diffusion | 0.944 | 0.973 | 0.935 | 0.950 | **0.812** | **0.890** | 0.751 | 0.703 | 0.868 | 0.659 | 17.6 | 0.415 | 0.877 | 0.429 | **0.998** | 0.864 |
| frozen CLIP-L | 0.890 | 0.928 | 0.823 | 0.912 | 0.674 | 0.771 | 0.597 | 0.510 | 0.714 | 0.483 | 13.8 | 0.668 | 0.857 | 0.678 | 0.741 | 0.701 |
| SupCon | 0.967 | 0.967 | 0.929 | 0.952 | 0.728 | 0.830 | 0.658 | 0.656 | 0.816 | 0.609 | 19.8 | 0.551 | 0.812 | 0.565 | 0.851 | 0.814 |
| SimCLR | 0.826 | 0.897 | 0.828 | 0.926 | 0.480 | 0.606 | 0.353 | 0.409 | 0.586 | 0.325 | 19.1 | 0.712 | 0.888 | 0.723 | 0.650 | 0.582 |
| SupCon + SimCLR | 0.928 | 0.942 | 0.893 | 0.942 | 0.664 | 0.766 | 0.564 | 0.553 | 0.729 | 0.489 | 19.4 | 0.677 | 0.891 | 0.689 | 0.763 | 0.706 |
| MaskCon | 0.954 | 0.961 | 0.928 | 0.950 | 0.692 | 0.798 | 0.598 | 0.575 | 0.765 | 0.521 | 20.3 | 0.558 | 0.856 | 0.572 | 0.844 | 0.796 |
| FALCON (uses K) | 0.963 | 0.972 | 0.928 | 0.950 | 0.702 | 0.810 | 0.615 | 0.613 | 0.787 | 0.549 | 20.0 | 0.595 | 0.811 | 0.609 | 0.824 | 0.785 |
| GEORGE | 0.973 | 0.971 | 0.911 | 0.947 | 0.690 | 0.804 | 0.613 | 0.597 | 0.774 | 0.542 | 21.1 | 0.593 | 0.642 | 0.606 | 0.830 | 0.764 |
| GEORGE stage 1 (ERM) | 0.972 | 0.973 | 0.907 | 0.944 | 0.696 | 0.806 | 0.613 | 0.594 | 0.774 | 0.542 | 20.0 | 0.599 | 0.629 | 0.612 | 0.813 | 0.770 |
| NHP (z_sub) | 0.972 | 0.976 | 0.853 | 0.873 | 0.469 | 0.632 | 0.343 | 0.393 | 0.584 | 0.280 | 10.0 | 0.440 | 0.856 | 0.454 | 0.762 | 0.749 |
| NHP (z_make + z_sub) | **0.983** | 0.982 | 0.861 | 0.867 | 0.462 | 0.629 | 0.329 | 0.396 | 0.588 | 0.284 | 10.6 | 0.463 | **0.452** | 0.477 | 0.789 | 0.758 |
| LeJEPA (no labels) | 0.916 | 0.943 | 0.923 | 0.946 | 0.688 | 0.793 | 0.600 | 0.549 | 0.744 | 0.494 | 20.4 | 0.553 | 0.893 | 0.567 | 0.829 | 0.779 |
| geosub `geo_sub` (B) | 0.937 | 0.922 | 0.932 | 0.951 | 0.734 | 0.832 | 0.670 | 0.603 | 0.784 | 0.556 | 19.8 | 0.498 | 0.867 | 0.512 | 0.866 | 0.839 |
| geosub `full` (B) | 0.940 | 0.973 | 0.931 | **0.953** | 0.724 | 0.833 | 0.658 | 0.632 | 0.803 | 0.582 | 20.3 | 0.561 | 0.827 | 0.574 | 0.880 | 0.847 |
| geosub `geo_sub` (L) | 0.947 | 0.948 | 0.934 | 0.949 | 0.795 | 0.889 | 0.758 | **0.724** | **0.883** | **0.696** | 17.5 | **0.402** | 0.886 | **0.416** | 0.944 | **0.896** |
| geosub `full` (L) | 0.953 | **0.983** | 0.934 | 0.949 | 0.799 | 0.887 | **0.763** | 0.704 | 0.878 | 0.686 | 17.9 | 0.486 | 0.834 | 0.500 | 0.914 | 0.889 |
| frozen SSCD | 0.412 | 0.495 | 0.441 | 0.581 | 0.206 | 0.216 | 0.056 | 0.119 | 0.077 | 0.021 | 5.1 | 0.961 | 0.988 | 0.962 | 0.337 | 0.278 |

- **Oracle K: diffusion gives the best partition on both backbones**, 0.773 on B (+0.046 over k-means, above every B-based row) and 0.812 on L (the highest
  in the table), without training.
- **K-hat (rule: > +0.03 over k-means K-hat)**: passes on L (0.703, +0.043; geosub-L 0.724 / 0.704), not on B (0.654 vs 0.659, ARI 0.518 vs 0.601).
  Per-class breakdown still to check (for geosub-L the gain came mostly from Cab).
- Diffusion partitions are nearly seed-independent (ARI seeds 0.98 / 1.00 vs 0.87 / 0.92); ARI boot is level with k-means.
- One seed: differences of about 0.02 are within noise.
