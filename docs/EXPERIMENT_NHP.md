# NHP experiment plan (nuisance-factored hierarchical prototypes)

Method and code: `lsgen/methods/repr/nhp.py`, `configs/repr_nhp.yaml`. Conflicting images (`data/splits/exclude.txt`, docs/DATA.md) are left out
of every experiment below. Fine labels are used in evaluation only; no test data in training, selection or tuning.
This document grows milestone by milestone; so far: Gate 0 and the assumptions made.

## Gate 0: does the premise hold?
NHP assumes that nuisance factors (viewpoint, colour, background) dominate the within-make geometry of the frozen embedding. Gate 0 measures it
before any further NHP training. `sbatch slurm/gate0.sh` (`scripts/gate0_nuisance.py`, `configs/gate0.yaml`; GPU only for the mirror images).

**Data.** Frozen DINOv3-B, bbox15 256 px crops, `train` split (cached embeddings; the mirror images are embedded and cached).

**Per make** (makes with fewer than 30 images are skipped and logged):
1. The top 5 principal components of the make's L2-normalised embeddings.
2. Proxies, computed without fine labels (`lsgen/features/nuisance.py`):
   - colour (13): mean RGB, mean saturation and value, saturation-weighted 8-bin hue histogram of the car's bbox;
   - shape (2): log aspect ratio and area fraction of the bbox;
   - background (2): mean and std of the grey value outside the bbox;
   - pose (2): cosine between the embedding of an image and of its mirror image (view symmetry), and the signed coordinate of their difference on its top
     direction (which way the car faces). Both are functions of the embedding itself, so they are reported apart from the pixel proxies.
3. Out-of-fold (5-fold) R^2 of a ridge regression (standardised proxies, alpha by internal CV) onto each PC; the **variance explained** is their
   eigenvalue-weighted mean, so it is the share of the top-5 subspace's variance that the proxies predict. A control repeats the regression with the
   proxies of other images of the same make (shuffled); it should be near zero.
4. Macro mean over makes with a 95% CI from resampling the makes. Reported for: all pixel proxies (`pixel`), each group alone, `pose`, `pixel+pose`,
   `shuffled`; plus the R^2 of the first PC.

**Decision rule, fixed before the first run** (on `pixel_explained`, the macro value; configs/gate0.yaml):
- below 0.15: the premise is weak, NHP is not expected to help: stop and report;
- 0.15 to 0.30: inconclusive, report which groups carry it;
- 0.30 or more: the premise is supported (it does not show that NHP can exploit it).
For orientation: R4's `var_ratio` of 0.69 says that most within-make variance is not the subclass, but it does not say what it is.

Limits: the proxies are crude and linear, so a low value does not prove that nuisance is absent (it may be non-linear or unmeasured); a high value shows
that the factors are present, not that they hide the subclass. `explained` needs several hundred images per make to be stable; per-make values are noisy.

## Assumptions
- The Gate 0 thresholds (0.15 / 0.30), 5 PCs, ridge on standardised proxies and the 30-image minimum are my choices; the prompt fixes none of them.
- NHP's evaluated embedding is `z_sub` (128-d, L2-normalised) and `z_make + z_sub` (256-d); the baselines are evaluated on the 768-d backbone CLS.
- NHP weights in `configs/repr_nhp.yaml` are untuned.
