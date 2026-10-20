# Identity-score experiment (DINO minus DreamSim)

Idea (docs/dino_dreamsim_subclass_pipeline.png): a DINO backbone with two heads, SupCon on make labels and a subclass head trained on soft
targets from an *identity score*: how similar two cars of a make are under frozen DINO, minus how much they simply look alike under frozen
DreamSim. Conflicting images (`data/splits/exclude.txt`, docs/DATA.md) are left out. No training is done until the pre-check below passes.

## Pre-check: are the targets right?
`sbatch slurm/identity_check.sh` (`scripts/identity_check.py`, `lsgen/eval/identity.py`, `configs/identity_check.yaml`). CPU only, reads the
cached frozen embeddings `dinov3_base_256` and `dreamsim_ensemble_256` of the `train` split (the split the method would train on).

**Score, per make.** Cosine similarities of all pairs under each encoder, z-scored over the make's pairs (the two encoders have different
ranges), then `identity = z(DINO) - z(DreamSim)`. **Gated**: an image's score is only used for its 20 nearest same-make DINO neighbours, so
pairs that DINO finds dissimilar but DreamSim finds even more dissimilar never count as "same subclass".

**Measured per make with K_c >= 2** (fine labels only score the rankings):
- `auroc_*`: AUROC of DINO, DreamSim and the identity score for separating same-subclass from different-subclass pairs;
- `p5_*`: share of each image's 5 top-scored neighbours with its subclass, for DINO, DreamSim, identity, gated identity; `p5_chance` for reference;
- `gain_auroc` = identity - DINO, `gain_p5_gated` = gated identity - DINO.
Macro mean over makes; 95% CIs from 1000 draws of 63.2% of each make's images without replacement.

**Decision rule, fixed before the first run:** go to training if the macro `gain_p5_gated` is >= 0.02 and its CI lower bound is > 0; otherwise
stop. The 0.02 margin, k = 20 and top 5 are my choices.

## Result (`reports/identity_check/`, single run, 7,270 train images, 30 makes with K_c >= 2)

**Verdict: stop.** The gated identity score is worse than DINO alone, so the training method was not built.

| Score | AUROC same vs different subclass | precision@5 |
|---|---|---|
| frozen DINOv3-B | 0.799 | 0.903 |
| DreamSim ensemble | 0.714 | 0.700 |
| identity = z(DINO) - z(DreamSim) | 0.619 | 0.557 |
| identity, gated to the DINO top 20 | - | 0.795 |
| chance | 0.5 | 0.274 |

`gain_p5_gated` = -0.109 (rule: >= +0.02); `gain_auroc` = -0.180. Subtracting DreamSim lowers AUROC in 30/30 makes and the gated precision@5 in 29/30.

Reading: the premise that DreamSim measures only nuisance (colour, pose, background) does not hold. DreamSim alone finds same-subclass
neighbours far above chance (0.70 vs 0.27): same-model cars share body shape, which is part of "looks alike". Subtracting it removes subclass
signal. In the makes where DreamSim beats DINO (makes 14, 27: AUROC 0.86 vs 0.62-0.65) the loss is largest. Consistent with Gate 0
(docs/EXPERIMENT_NHP.md): nuisance does not dominate the within-make geometry of DINOv3.

Exploratory follow-up (point estimates on the same data, no CIs, not part of the decision rule): no combination beats DINO alone.
DINO - 0.5 DreamSim 0.754 / 0.771 (AUROC / p@5); DINO + DreamSim 0.787 / 0.852; min(DINO, DreamSim) 0.758 / 0.851; DreamSim re-ranking
the DINO top 20: p@5 0.853; DINO + the DreamSim DINOv2 or CLIP variant 0.753 / 0.817 and 0.784 / 0.853. The two z-scored similarities correlate
at 0.59 per make; DINO + DreamSim beats DINO on p@5 in 4/30 makes.

Note: frozen DINO's same-make top-5 neighbours already share the subclass 90% of the time while its oracle-K clustering ACC is 0.72, so the
subclass signal is local in DINO; a DINO-only soft-neighbour target (the diagram without DreamSim) is the remaining untested variant.

Limits: one DreamSim model (ensemble) fixed beforehand; linear z-score combination. The p@5 CIs in `table.csv` lie below the point
estimates: a 63.2% subsample has fewer same-subclass neighbours, so subsampled p@5 is biased low. The verdict does not depend on it (gap -0.11
against +0.02), but those CIs should not be quoted.
