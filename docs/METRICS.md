# Metric definitions (Step 1, Milestone 4)

Every number in a report is defined here and implemented in `lsgen/eval/` (tests: `tests/test_*.py`, synthetic
cases with known answers). Entry points: `scripts/eval_representation.py` (R1-R5), `scripts/eval_generation.py`
(G1-G7 and B1-B6). Design rationale: docs/PLAN.md A3-A5.

## Conventions
- **Data.** Reference = `test` (8,041 images minus excluded ones); `train` is used for fitting probes, kNN references,
  the real ceilings of G5 and the nearest-train search of G7; `val` only for the evaluator calibration and for choosing
  tau (G4). `test` is never used for tuning. The 178 conflicting images (`data/splits/exclude.txt`, docs/DATA.md) are
  dropped by `load_split` everywhere; every `summary.json` records how many were removed per split.
- **Fine labels** are used only inside `lsgen/eval/` and the scripts that call it (`tests/test_no_fine_labels.py`).
- **Embeddings** are the cached bbox15 crops (`scripts/extract_features.py`). R-metrics L2-normalise first. "Squared
  distance" is on L2-normalised vectors, so it equals `2 - 2 cos`.
- **Per make and macro.** Every R and G metric except G1/G7 and the set-level G2 prevalence is computed per make
  (coarse class); the macro value is the unweighted mean over makes (makes where the metric is undefined are ignored and
  logged: NaN rows in the CSV). 95% CIs: 1000 bootstrap resamples of the images *within each make*
  (`lsgen/eval/stats.py`); the macro CI averages the makes inside every resample. B-metrics resample episodes within
  make. G1 and G7 are set-level numbers; their variance comes from the 3 generation seeds.
- **Small n.** Subclass-level sets have about 40 images: KID and coverage are used, never FID, and `n` is stored.
- **Same-n.** FID/KID/FD (G1) compare `n = min(|samples|, |test|)` images (random subsample of the larger set, seeded).

## Evaluators (docs/PLAN.md A3)
| Role | Model | Used for |
|---|---|---|
| Fine / make classifier | ConvNeXt-T (ImageNet init) trained on `train` at 128 px, temperature-scaled on `val` | G2-G5, B1, B5 |
| Inception pool3 (clean-fid, 2048-d) | `inception` | G1 FID, KID, precision/recall, density/coverage |
| CLIP ViT-L/14 | `clip_l` | G4, G5, G6 (KID, coverage), B2, B4 |
| SSCD disc_mixup | `sscd` | G7, B3 |

A method declares the encoders it used in `method.json` (`"uses"`). `eval_generation.py` refuses to score it if one of
them is an evaluator, and records a warning for same-family pairs (e.g. a ConvNeXt-based method vs the ConvNeXt classifier).

## Representation metrics (use `test` embeddings, `train` as the fitted / reference set)
| ID | Definition |
|---|---|
| **R1** kNN | Make accuracy of kNN (k = 20, cosine, majority vote) with `train` as reference, per test image, averaged per make |
| **R1** probe | Make accuracy of a multinomial logistic regression (C = 10, fixed, 500 lbfgs iterations) fit on `train` embeddings |
| **R2** clustering | Per make with K_c >= 2: test embeddings of the make are projected on the make's own top-50 PCs and clustered with k-means (10 inits). **Oracle K** = K_c (labelled "uses K"). **K-hat**: silhouette sweep over K = 2..25 (3 inits); if the best silhouette is < 0.10 (fixed constant, not tuned) the gap statistic (Tibshirani; 10 uniform references; smallest K with gap(K) >= gap(K+1) - s(K+1); can return 1) decides. Reported (`r2_oracle.csv`, `r2_khat.csv`): ACC (Hungarian one-to-one matching of clusters to fine classes), NMI, ARI, and K-hat for the K-hat rule. **CIs of the K-hat table** use 1000 draws of 63.2% of the make's images *without* replacement instead of the bootstrap: resamples contain duplicate images, which form tight pairs and inflate the silhouette-selected K (measured: macro K-hat 5.4 with a bootstrap CI of [13, 16]). The point estimates always use all images. |
| **R2** within-make | Recall@1 (MaskCon protocol): for each test image of a make, is its nearest other image of the same make (cosine) of the same fine class. Fine linear probe: logistic regression (as R1) fit on the make's `train` images, accuracy on its test images. |
| **R3** geometry | For one make: Delta0 = mean squared distance over pairs with the same fine class, Delta1 = same make and different fine class, Delta2 = different make; self-pairs excluded (also when bootstrap resampling repeats an image). Ratios Delta0/Delta1 and Delta1/Delta2 (< 1 desired). Indicators Eq. (1): Delta0 < Delta1 and Eq. (2): Delta1 < Delta2; the macro value is the % of makes satisfying them. Delta1 and the Eq. indicators are NaN for K_c = 1 makes. |
| **R4** variation | Per make with K_c >= 2: `var_ratio` = size-weighted mean over subclasses (>= 3 images) of the within-subclass trace covariance, divided by the trace covariance of the make's images. `pr_subclass` = mean participation ratio (sum lambda)^2 / sum lambda^2 of the subclass covariance spectrum (effective rank); subclasses whose points all coincide are ignored. **CIs:** `var_ratio` uses the same 63.2% subsample draws as K-hat, because bootstrap resamples repeat images and lower it (the point estimate fell outside its own bootstrap CI for every encoder in the first runs). `pr_subclass` has no CI: an effective rank grows with the number of points, so both resampling with duplicates and subsampling bias it; compare it across encoders only as a point estimate. |
| **R5** stability | Per make with K_c >= 2, oracle K, on the R2 feature space: `ari_seeds` = mean ARI between the k-means runs of 3 seeds; `ari_boot` = mean ARI between the full-data clustering and 20 clusterings fitted on bootstrap resamples (then applied to all points). The macro CI of `ari_boot` is the 2.5/97.5 percentile of the make-averaged ARI of the i-th bootstrap fit. For K_c = 1 makes: `khat_error` = K-hat - 1 (> 0 means over-fragmentation; point estimate, no CI). |

## Use case A (generated images, `samples.csv`: filename, make_id [, group]; all vs `test`)
| ID | Definition |
|---|---|
| **G1** | FID (Frechet distance of Inception features), KID (unbiased MMD^2, kernel (x.y/d + 1)^3; sets up to 1000 compared whole, larger ones averaged over 100 random subsets of 1000), precision/recall (k = 5 nearest-neighbour balls, Kynkaanniemi 2019), density/coverage (Naeem 2020), all at matched n. |
| **G2** | Make accuracy: the make classifier predicts the requested make (per make, macro). Make-prevalence TV: 0.5 * sum \|p_samples(make) - p_test(make)\| with the make classifier's predicted makes. |
| **G3** | Per requested make, over that make's fine classes: TV and add-one-smoothed KL(q_test \|\| p_samples) between the fine classifier's argmax histogram of the make's samples and the fine histogram of `test`. Restricting to the make's classes isolates O2 from O1. **No CI**: a histogram distance computed from resampled images is biased upward (resamples repeat images), so G3 is a point estimate per make and macro, judged against the real-vs-real floor. |
| **G4** | Rare = bottom quartile of fine classes by `train` count (count <= 25% quantile). tau = smallest confidence at which the calibrated fine classifier has >= 90% accuracy on `val`. A sample is assigned to class g if the classifier predicts g with confidence >= tau and g belongs to the requested make; g is *covered* if >= 5 samples are assigned. Reported: % covered overall, rare, common (per make, macro). **No CI** (counts depend on the number of samples). |
| **G5** | For every covered class g (CLIP space): KID and coverage (k = 3) between its assigned samples and `test_g`, and the same for the real ceiling (a random `train_g` subset of the same size against `test_g`). `kid_excess` = mean KID - mean ceiling KID (0 = as good as real; the comparison used for validation); `kid_ratio` = mean KID / mean ceiling KID is kept for reference only: the ceiling of an unbiased KID is centred on 0 (negative for about half of the classes), so the ratio is ill-conditioned, even for the real-vs-real floor; `coverage_ratio` likewise (1 = as good as real). `g4_g5_per_class.csv` holds the point estimates for every class, with n. **CIs:** `kid` and `kid_excess` use 63.2% subsample draws (an unbiased KID has no sample-size bias); `coverage`, `coverage_ratio` and `kid_ratio` have no CI (coverage depends on the number of samples). |
| **G6** | Only with a `group` column. Per make: KID cost matrix between generated groups (>= 2 samples) and the make's real subclasses (CLIP space), Hungarian matching; `matched_kid` = mean matched cost, `purity` = mean over groups of the largest share of the evaluator's fine histogram, `entropy` = mean entropy of those histograms, `n_unmatched_real` = real subclasses without a group. |
| **G7** | Cosine similarity of every sample to its nearest `train` image in SSCD space; summary: mean, share > 0.5 (`frac_gt`), and the same for the null = nearest-train similarity of every `test` image. The 64 most similar (sample, nearest train) pairs are saved as `g7_nn_grid.png`; all similarities in `g7_samples.csv`. A method copies if its numbers are clearly above the null. |
| **E1** | Only when `samples.csv` has a `fine_id` column (the requested subclass: methods conditioned on the fine class, e.g. the oracle upper bound). Evaluator-domain check: accuracy of the fine classifier against the requested class on the generated images, its mean confidence, the share of confident (>= tau) and confident-and-correct predictions, next to the same numbers on real `val` images (`e1_evaluator.csv`). `g4_g5_requested.csv` repeats G4 / G5 with the requested class as the assignment (a perfect evaluator): its difference from `g4_g5.csv` is the effect of the evaluator's errors on generated images. |

## Use case B (episodes in `data/episodes/`; N = 16 generated images per episode; `samples.csv`: filename, episode_id)
S = support (k images), T = remaining test images of the subclass (mixed episodes: unions over the two subclasses).
Episode results are in `episodes.csv`; `by_set.csv` has per make and macro values with CIs for every episode set
(`single_test_k1/5/10`, `single_train_k*`, `mixed_k10`); the macro rows over k are **B6**.
| ID | Definition |
|---|---|
| **B1** | Hit rate: share of the 16 images the fine classifier assigns to the support's subclass (single-class episodes). |
| **B2** | KID and coverage (CLIP, k = 3) of the samples against T (`kid_T`, `coverage_T`, the beyond-support fidelity) and against S (`kid_S`, `coverage_S`). Against S the values are NaN when S is too small (KID needs >= 2 images, coverage needs > 3). |
| **B3** | SSCD exemplar dependence: `sim_S` / `sim_T` = mean over samples of the max similarity to S / T; `sim_T_to_S` = the same for real T images against S (the real reference); `ratio_S` = sim_S / sim_T_to_S (> 1: samples are closer to the support than an unseen real image is). |
| **B4** | Mean pairwise CLIP cosine distance among the samples (`div_gen`), within T (`div_T`), and their ratio (`div_ratio`; ~ 0 for prototype collapse). |
| **B5** | Mixed episodes (subclasses a, b of one make; S holds round(r * 10) images of a): abs(share of samples with p(a) > p(b) - r), the fine classifier choosing between the two subclasses. |
| **B6** | B1-B4 (and B5) as a function of k: the macro rows of `by_set.csv`, one set per k. |

## Note on K-hat and K_c = 1 makes (checked on `train` embeddings, 2 Oct 2026)
The silhouette/gap rule above is fixed in advance and was not tuned. Checked on `train` (never `test`) for DINOv3-L-256,
DINOv3-B-128 and CLIP-L-128, against K_c, per make:
- DINOv3: the best silhouette of K_c = 1 makes (median 0.24-0.27) is about as high as that of K_c >= 2 makes (median 0.28-0.29),
  so no silhouette threshold separates them; the gap statistic alone also splits them (gap-K of 1 for only 4-7 of 19
  single-subclass makes). A threshold that helps one encoder hurts another: for CLIP, SIL_MIN = 0.25 gives K-hat = 1 for all
  K_c = 1 makes but underestimates K_c >= 2 makes by 1.8 on average, while for DINOv3 it changes little.
- Hypothesis (not tested): inside one official class the DINOv3 space has real structure (viewpoint, colour, background),
  which k-means finds. K-hat > 1 on K_c = 1 makes is then a property of the encoder space and of the label-free per-make
  pipeline, i.e. the over-fragmentation that this benchmark is meant to expose, not a defect to tune away. It is reported
  as measured (`khat_error`), and the rule is not changed per encoder.

## Controls and metric validation (Milestone 5)
Controls are trivial "generators" with a known defect, written in the format of a real method (`scripts/make_controls.py`,
`lsgen/eval/controls.py`) and scored by the unchanged evaluation (`slurm/eval_controls.sh`). `scripts/metric_validation.py`
compares each control with the real-vs-real floor (C2) and writes `reports/metric_validation.{csv,md,png}`.
Real images are symlinked from the stored bbox15 crops; the 178 conflicting images are excluded as everywhere.

| Control | Use case A (N = number of test images, requested makes follow test) | Use case B (16 images per episode, test-support episodes) |
|---|---|---|
| C1 copy-train | random train image of a subclass drawn with train prevalence in the make | -- |
| C1b copy-support | -- | the support images cycled to 16, each flipped / cropped (70-100%) / brightness-jittered |
| C1b retrieve-NN | -- | the 16 train images nearest (cosine, DINOv3-B) to the mean support embedding |
| C2 oracle | `test_A` images scored against `test_B` (`reference_split` in method.json) | 16 other real images of the subclass from `train` (mixed: round(16 r) of a, the rest of b) |
| C3 coarse-only | subclass uniform within the make, random train image of it | -- |
| C4 prototype collapse | subclass with train prevalence, always its medoid (DINOv3-B, train) | the medoid of the subclass (mixed: in proportion r) repeated |
| C5 majority-only | the largest (train) subclass of the make | -- |
| C6 degraded | C1 images after Gaussian blur (sigma 2) and JPEG q20 | -- |

C4 and retrieve-NN declare `uses: [dinov3_base]` (not an evaluator). Verdict per control x metric
(`configs/metric_validation.yaml`, fixed before the first run): a metric has *moved* if the control is worse than C2
with non-overlapping 95% CIs, or, for set-level numbers without CI, by more than 25% of |C2|. `ok` = a listed expected
failure that moved; `MISSED` = listed but did not move (the script exits with 1; stop and report, do not tune);
`also` = moved though not listed (for example every control that returns train images also raises G7). C3 is listed for
G3 only: uniform sampling covers every subclass of a make, so G4 cannot fail by construction (the plan listed G3 and G4).
