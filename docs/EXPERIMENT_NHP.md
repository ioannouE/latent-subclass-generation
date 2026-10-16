# NHP experiment plan (nuisance-factored hierarchical prototypes)

Method and code: `lsgen/methods/repr/nhp.py`, `configs/repr_nhp.yaml`. Conflicting images (`data/splits/exclude.txt`, docs/DATA.md) are left out
of every experiment below. Fine labels are used in evaluation only; no test data in training, selection or tuning.
This document grows milestone by milestone; so far: Gate 0, the NHP training run and the assumptions made.

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

## Results (single seed 0, 256 px, `train` split for training, evaluation as in docs/METRICS.md)

**Gate 0 (`reports/gate0/`, 7,270 images, 48 makes).** The premise is not supported. Macro `pixel_explained` is -0.015 (95% CI [-0.16, 0.08]), far below
the 0.15 cut-off, so by the rule fixed beforehand the verdict is *weak: NHP is not expected to help, stop*. No group carries it: colour -0.10, background -0.04,
shape 0.07, pose 0.01, pixel+pose 0.05; the shuffled control is -0.05 (as expected, not positive). Negative values are out-of-fold R^2 below the mean predictor.
Caveat (see Limits above): the proxies are crude and linear, so this does not rule out non-linear or unmeasured nuisance.

**NHP training run (`reports/repr/nhp_*_256/`)**, run although Gate 0 said stop. Macro over makes (R1, R4) or groups of the R2 tables; conflicting images excluded.
Baselines are the cached rows, which reproduce the numbers in the prompt.

| Encoder | kNN make | ACC oracle K | ACC K-hat | var ratio |
|---|---|---|---|---|
| frozen DINOv3-B | 0.949 | 0.717 | 0.619 | 0.690 |
| SupCon | 0.968 | 0.718 | 0.634 | 0.703 |
| NHP (`z_sub`, 128-d) | 0.965 | 0.580 | 0.541 | 0.679 |
| NHP (`z_make + z_sub`, 256-d) | 0.983 | 0.578 | 0.541 | 0.701 |
| NHP w/o invariance | 0.951 | 0.590 | 0.553 | 0.680 |
| NHP w/o hierarchy | 0.963 | 0.591 | 0.550 | 0.674 |
| NHP w/o sparsity | 0.966 | 0.611 | 0.539 | 0.694 |
| NHP w/o nuisance branch | 0.964 | 0.590 | 0.547 | 0.676 |

Reading:
- Make-level kNN is as good as SupCon (the concatenated embedding is best at 0.983), so NHP keeps make information.
- Subclass recovery is clearly worse: ACC with oracle K is about 0.58 against 0.72 for SupCon and frozen DINOv3-B, and ACC with estimated K 0.54 against 0.62-0.63.
  The var ratio does not move (0.67-0.70), so NHP did not reduce the non-subclass share of the within-make variance.
- No ablation recovers the gap (0.58-0.61 for all); removing sparsity helps most (+0.03) but stays far below the baselines. The ablations are not distinguishable
  from each other with one seed and no CIs on these macro numbers, so no component can be called useful or harmful from this run.
- This agrees with Gate 0: if nuisance does not dominate the within-make geometry, factoring it out has nothing to remove.

Limits: one seed; NHP loss weights untuned (see Assumptions); the macro numbers above are plain means of the per-group values in the CSVs, the CIs are in the CSVs.
Not done: tuning on `val`, a second seed, non-linear nuisance probes.

## Assumptions
- The Gate 0 thresholds (0.15 / 0.30), 5 PCs, ridge on standardised proxies and the 30-image minimum are my choices; the prompt fixes none of them.
- NHP's evaluated embedding is `z_sub` (128-d, L2-normalised) and `z_make + z_sub` (256-d); the baselines are evaluated on the 768-d backbone CLS.
- NHP weights in `configs/repr_nhp.yaml` are untuned.
