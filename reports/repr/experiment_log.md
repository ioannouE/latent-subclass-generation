# Representation experiment log

Running log of the representation approaches tried, newest last. One entry per approach: idea, setup, results, verdict, open items.
Common setup unless stated otherwise: frozen/fine-tuned DINOv3-B, bbox15 256 px crops, `train` split for training, evaluation as in docs/METRICS.md,
conflicting images (`data/splits/exclude.txt`) excluded, fine labels used in evaluation only, single seed 0.
Reference rows (macro over makes): frozen DINOv3-B ACC oracle-K 0.717, K-hat 0.619, var ratio 0.690; SupCon 0.718 / 0.634 / 0.703.

Entry template: **Idea** / **Setup** / **Results** / **Verdict** / **Not done**.

---

## 1. NHP: nuisance-factored hierarchical prototypes

Code: `lsgen/methods/repr/nhp.py`, `configs/repr_nhp.yaml`. Details and Gate 0 protocol: docs/EXPERIMENT_NHP.md. Results: `reports/gate0/`, `reports/repr/nhp_*_256/`.

**Idea.** Assume nuisance factors (viewpoint, colour, background) dominate the within-make geometry of the frozen embedding. Factor them out and the hidden
subclasses should show. Trained with make labels and augmentations only. A head (2-layer MLP, 2048 hidden) on the backbone output (last 4 blocks fine-tuned)
gives three codes:
- `z_make` (128-d): SupCon on make labels.
- `z_sub` (128-d): per-make mixture of 24 learnable prototypes, cosine soft assignment, trained by DINO-style self-distillation between two views (EMA teacher,
  sharpened targets) under nuisance-heavy augmentation (colour permutation, strong jitter, background swap/mask, flip, crop) so the assignment is nuisance-invariant.
  Hierarchy hinge (same prototype < same make other prototype < other make), hard-concrete gates + L0 penalty for sparse prototype use, weak entropy
  anti-collapse term (no equipartition).
- `z_nuis` (64-d): within-subclass variation, kept alive by VICReg variance/covariance. Not evaluated.

Evaluated embeddings: `z_sub` (128-d) and `z_make + z_sub` (256-d).

**Gate 0 (premise check, rule fixed beforehand: < 0.15 means stop).** Out-of-fold ridge R^2 of crude pixel proxies (colour, bbox shape, background, pose via
mirror-image cosine) onto the top 5 within-make PCs, 48 makes. Macro explained variance **-0.015** (CI [-0.16, 0.08]); colour -0.10, background -0.04,
shape 0.07, pose 0.01; shuffled control -0.05. Verdict by the rule: premise weak, stop. Caveat: proxies are crude and linear.

**NHP training** (run despite Gate 0):

| Encoder | kNN make | ACC oracle K | ACC K-hat | var ratio |
|---|---|---|---|---|
| frozen DINOv3-B | 0.949 | 0.717 | 0.619 | 0.690 |
| SupCon | 0.968 | 0.718 | 0.634 | 0.703 |
| NHP `z_sub` | 0.965 | 0.580 | 0.541 | 0.679 |
| NHP `z_make + z_sub` | 0.983 | 0.578 | 0.541 | 0.701 |
| w/o invariance | 0.951 | 0.590 | 0.553 | 0.680 |
| w/o hierarchy | 0.963 | 0.591 | 0.550 | 0.674 |
| w/o sparsity | 0.966 | 0.611 | 0.539 | 0.694 |
| w/o nuisance branch | 0.964 | 0.590 | 0.547 | 0.676 |

**Verdict.** Did not work. Make information is kept (kNN make on par with SupCon, 0.983 for the concatenation), but subclass recovery is clearly worse
(ACC oracle K ~0.58 vs 0.72; K-hat 0.54 vs 0.62-0.63) and the var ratio does not move (0.67-0.70). No ablation closes the gap (0.58-0.61; sparsity removal
helps most, +0.03); with one seed and no CIs the ablations cannot be ranked. Consistent with Gate 0: if nuisance does not dominate within-make geometry,
factoring it out removes nothing.

**Not done.** Loss weights untuned; no tuning on `val`; second seed; non-linear nuisance probes; the `nhp_adversary` variant is configured but has no results.
