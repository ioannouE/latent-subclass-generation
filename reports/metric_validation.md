# Metric validation (controls x metrics)

Provenance: git `f3146004bf560129d5ae005067006c80c99a03e9`, seed 0, config `configs/metric_validation.yaml`. conflicting images (same photo under different fine labels) are left out of every split: {'train': 68, 'val': 9, 'test': 101} (train / val / test).

Floor = c2_oracle_split (A), c2_oracle_real (B, set single_test_k5). Verdicts: ok = expected and clearly worse than the floor; MISSED = expected, not worse; also = worse though not expected; same = unchanged.

## Expected failures that were missed: 0

none

## All controls

| use_case   | control               | metric                         |     value |       lo |       hi |   floor | expected   | worse_than_floor   | verdict   |
|:-----------|:----------------------|:-------------------------------|----------:|---------:|---------:|--------:|:-----------|:-------------------|:----------|
| A          | c2_oracle_split       | G1 FID                         |    4.9897 | nan      | nan      |  4.9897 | False      | False              | floor     |
| A          | c1_copy_train         | G1 FID                         |    3.8046 | nan      | nan      |  4.9897 | False      | False              | same      |
| A          | c3_coarse_only        | G1 FID                         |    3.8352 | nan      | nan      |  4.9897 | False      | False              | same      |
| A          | c4_prototype_collapse | G1 FID                         |   39.0532 | nan      | nan      |  4.9897 | False      | True               | also      |
| A          | c5_majority_only      | G1 FID                         |   14.054  | nan      | nan      |  4.9897 | False      | True               | also      |
| A          | c6_degraded           | G1 FID                         |  103.779  | nan      | nan      |  4.9897 | True       | True               | ok        |
| A          | c2_oracle_split       | G1 KID                         |    0      | nan      | nan      |  0      | False      | False              | floor     |
| A          | c1_copy_train         | G1 KID                         |    0.0001 | nan      | nan      |  0      | False      | False              | same      |
| A          | c3_coarse_only        | G1 KID                         |    0      | nan      | nan      |  0      | False      | False              | same      |
| A          | c4_prototype_collapse | G1 KID                         |    0.0023 | nan      | nan      |  0      | False      | True               | also      |
| A          | c5_majority_only      | G1 KID                         |    0.0024 | nan      | nan      |  0      | False      | True               | also      |
| A          | c6_degraded           | G1 KID                         |    0.0943 | nan      | nan      |  0      | True       | True               | ok        |
| A          | c2_oracle_split       | G1 recall                      |    0.7975 | nan      | nan      |  0.7975 | False      | False              | floor     |
| A          | c1_copy_train         | G1 recall                      |    0.7108 | nan      | nan      |  0.7975 | False      | False              | same      |
| A          | c3_coarse_only        | G1 recall                      |    0.7178 | nan      | nan      |  0.7975 | False      | False              | same      |
| A          | c4_prototype_collapse | G1 recall                      |    0      | nan      | nan      |  0.7975 | True       | True               | ok        |
| A          | c5_majority_only      | G1 recall                      |    0.4699 | nan      | nan      |  0.7975 | False      | True               | also      |
| A          | c6_degraded           | G1 recall                      |    0.0103 | nan      | nan      |  0.7975 | False      | True               | also      |
| A          | c2_oracle_split       | G2 make acc                    |    0.8684 |   0.8533 |   0.8828 |  0.8684 | False      | False              | floor     |
| A          | c1_copy_train         | G2 make acc                    |    1      |   1      |   1      |  0.8684 | False      | False              | same      |
| A          | c3_coarse_only        | G2 make acc                    |    1      |   1      |   1      |  0.8684 | False      | False              | same      |
| A          | c4_prototype_collapse | G2 make acc                    |    1      |   1      |   1      |  0.8684 | False      | False              | same      |
| A          | c5_majority_only      | G2 make acc                    |    1      |   1      |   1      |  0.8684 | False      | False              | same      |
| A          | c6_degraded           | G2 make acc                    |    0.1258 |   0.1175 |   0.1343 |  0.8684 | False      | True               | also      |
| A          | c2_oracle_split       | G2 prevalence TV               |    0.0244 | nan      | nan      |  0.0244 | False      | False              | floor     |
| A          | c1_copy_train         | G2 prevalence TV               |    0      | nan      | nan      |  0.0244 | False      | False              | same      |
| A          | c3_coarse_only        | G2 prevalence TV               |    0      | nan      | nan      |  0.0244 | False      | False              | same      |
| A          | c4_prototype_collapse | G2 prevalence TV               |    0      | nan      | nan      |  0.0244 | False      | False              | same      |
| A          | c5_majority_only      | G2 prevalence TV               |    0      | nan      | nan      |  0.0244 | False      | False              | same      |
| A          | c6_degraded           | G2 prevalence TV               |    0.536  | nan      | nan      |  0.0244 | False      | True               | also      |
| A          | c2_oracle_split       | G3 TV                          |    0.0219 | nan      | nan      |  0.0219 | False      | False              | floor     |
| A          | c1_copy_train         | G3 TV                          |    0.032  | nan      | nan      |  0.0219 | False      | True               | also      |
| A          | c3_coarse_only        | G3 TV                          |    0.0428 | nan      | nan      |  0.0219 | True       | True               | ok        |
| A          | c4_prototype_collapse | G3 TV                          |    0.0292 | nan      | nan      |  0.0219 | False      | True               | also      |
| A          | c5_majority_only      | G3 TV                          |    0.4388 | nan      | nan      |  0.0219 | True       | True               | ok        |
| A          | c6_degraded           | G3 TV                          |    0.1614 | nan      | nan      |  0.0219 | False      | True               | also      |
| A          | c2_oracle_split       | G3 KL                          |    0.0032 | nan      | nan      |  0.0032 | False      | False              | floor     |
| A          | c1_copy_train         | G3 KL                          |    0.0051 | nan      | nan      |  0.0032 | False      | True               | also      |
| A          | c3_coarse_only        | G3 KL                          |    0.0087 | nan      | nan      |  0.0032 | True       | True               | ok        |
| A          | c4_prototype_collapse | G3 KL                          |    0.0046 | nan      | nan      |  0.0032 | False      | True               | also      |
| A          | c5_majority_only      | G3 KL                          |    1.4317 | nan      | nan      |  0.0032 | True       | True               | ok        |
| A          | c6_degraded           | G3 KL                          |    0.184  | nan      | nan      |  0.0032 | False      | True               | also      |
| A          | c2_oracle_split       | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | floor     |
| A          | c1_copy_train         | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c3_coarse_only        | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c4_prototype_collapse | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c5_majority_only      | G4 covered                     |    0.5476 | nan      | nan      |  1      | True       | True               | ok        |
| A          | c6_degraded           | G4 covered                     |    0.144  | nan      | nan      |  1      | False      | True               | also      |
| A          | c2_oracle_split       | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | floor     |
| A          | c1_copy_train         | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c3_coarse_only        | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c4_prototype_collapse | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c5_majority_only      | G4 rare covered                |    0.2143 | nan      | nan      |  1      | True       | True               | ok        |
| A          | c6_degraded           | G4 rare covered                |    0.1173 | nan      | nan      |  1      | False      | True               | also      |
| A          | c2_oracle_split       | G5 excess KID                  |    0.0135 |   0.0018 |   0.0389 |  0.0135 | False      | False              | floor     |
| A          | c1_copy_train         | G5 excess KID                  |    0.0729 |   0.0679 |   0.0806 |  0.0135 | False      | True               | also      |
| A          | c3_coarse_only        | G5 excess KID                  |    0.0715 |   0.0654 |   0.079  |  0.0135 | False      | True               | also      |
| A          | c4_prototype_collapse | G5 excess KID                  |    2.004  |   2.0029 |   2.0076 |  0.0135 | True       | True               | ok        |
| A          | c5_majority_only      | G5 excess KID                  |    0.0632 |   0.0583 |   0.0704 |  0.0135 | False      | True               | also      |
| A          | c6_degraded           | G5 excess KID                  |    4.4832 |   4.2633 |   4.599  |  0.0135 | False      | True               | also      |
| A          | c2_oracle_split       | G5 KID ratio (ill-conditioned) |   -1.8792 | nan      | nan      | -1.8792 | False      | False              | floor     |
| A          | c1_copy_train         | G5 KID ratio (ill-conditioned) |   38.6518 | nan      | nan      | -1.8792 | False      | True               | also      |
| A          | c3_coarse_only        | G5 KID ratio (ill-conditioned) |  -10.7094 | nan      | nan      | -1.8792 | False      | False              | same      |
| A          | c4_prototype_collapse | G5 KID ratio (ill-conditioned) | -314.138  | nan      | nan      | -1.8792 | False      | False              | same      |
| A          | c5_majority_only      | G5 KID ratio (ill-conditioned) |   -0.8433 | nan      | nan      | -1.8792 | False      | True               | also      |
| A          | c6_degraded           | G5 KID ratio (ill-conditioned) | -171.708  | nan      | nan      | -1.8792 | False      | False              | same      |
| A          | c2_oracle_split       | G5 coverage ratio              |    1.0423 | nan      | nan      |  1.0423 | False      | False              | floor     |
| A          | c1_copy_train         | G5 coverage ratio              |    0.8719 | nan      | nan      |  1.0423 | False      | False              | same      |
| A          | c3_coarse_only        | G5 coverage ratio              |    0.8803 | nan      | nan      |  1.0423 | False      | False              | same      |
| A          | c4_prototype_collapse | G5 coverage ratio              |    0.147  | nan      | nan      |  1.0423 | True       | True               | ok        |
| A          | c5_majority_only      | G5 coverage ratio              |    0.9477 | nan      | nan      |  1.0423 | False      | False              | same      |
| A          | c6_degraded           | G5 coverage ratio              |    0      | nan      | nan      |  1.0423 | False      | True               | also      |
| A          | c2_oracle_split       | G7 copies                      |    0.1351 | nan      | nan      |  0.1351 | False      | False              | floor     |
| A          | c1_copy_train         | G7 copies                      |    1      | nan      | nan      |  0.1351 | True       | True               | ok        |
| A          | c3_coarse_only        | G7 copies                      |    1      | nan      | nan      |  0.1351 | False      | True               | also      |
| A          | c4_prototype_collapse | G7 copies                      |    1      | nan      | nan      |  0.1351 | False      | True               | also      |
| A          | c5_majority_only      | G7 copies                      |    1      | nan      | nan      |  0.1351 | False      | True               | also      |
| A          | c6_degraded           | G7 copies                      |    0.4072 | nan      | nan      |  0.1351 | False      | True               | also      |
| B          | c2_oracle_real        | B1 hit rate                    |    0.9996 |   0.9994 |   0.9998 |  0.9996 | False      | False              | floor     |
| B          | c1b_copy_support      | B1 hit rate                    |    0.8504 |   0.8403 |   0.8615 |  0.9996 | False      | True               | also      |
| B          | c1b_retrieve_nn       | B1 hit rate                    |    0.9207 |   0.9136 |   0.928  |  0.9996 | False      | True               | also      |
| B          | c4_prototype_collapse | B1 hit rate                    |    1      |   1      |   1      |  0.9996 | False      | False              | same      |
| B          | c2_oracle_real        | B2 KID vs T                    |    0.0006 |  -0.0027 |   0.0038 |  0.0006 | False      | False              | floor     |
| B          | c1b_copy_support      | B2 KID vs T                    |    0.5033 |   0.4913 |   0.5158 |  0.0006 | True       | True               | ok        |
| B          | c1b_retrieve_nn       | B2 KID vs T                    |    0.0919 |   0.0853 |   0.0993 |  0.0006 | True       | True               | ok        |
| B          | c4_prototype_collapse | B2 KID vs T                    |    2.0007 |   1.9674 |   2.0331 |  0.0006 | False      | True               | also      |
| B          | c2_oracle_real        | B2 coverage vs T               |    0.684  |   0.6755 |   0.6931 |  0.684  | False      | False              | floor     |
| B          | c1b_copy_support      | B2 coverage vs T               |    0.3096 |   0.2966 |   0.3227 |  0.684  | False      | True               | also      |
| B          | c1b_retrieve_nn       | B2 coverage vs T               |    0.699  |   0.6907 |   0.7076 |  0.684  | False      | False              | same      |
| B          | c4_prototype_collapse | B2 coverage vs T               |    0.1492 |   0.1429 |   0.1551 |  0.684  | False      | True               | also      |
| B          | c2_oracle_real        | B3 ratio to S                  |    0.9998 |   0.9903 |   1.0086 |  0.9998 | False      | False              | floor     |
| B          | c1b_copy_support      | B3 ratio to S                  |    3.8083 |   3.7779 |   3.8387 |  0.9998 | True       | True               | ok        |
| B          | c1b_retrieve_nn       | B3 ratio to S                  |    1.1074 |   1.0969 |   1.1174 |  0.9998 | True       | True               | ok        |
| B          | c4_prototype_collapse | B3 ratio to S                  |    1.1494 |   1.124  |   1.1787 |  0.9998 | False      | True               | also      |
| B          | c2_oracle_real        | B4 diversity ratio             |    1.032  |   1.0236 |   1.0397 |  1.032  | False      | False              | floor     |
| B          | c1b_copy_support      | B4 diversity ratio             |    0.966  |   0.9518 |   0.9796 |  1.032  | False      | True               | also      |
| B          | c1b_retrieve_nn       | B4 diversity ratio             |    0.8788 |   0.8709 |   0.8862 |  1.032  | False      | True               | also      |
| B          | c4_prototype_collapse | B4 diversity ratio             |    0      |   0      |   0      |  1.032  | True       | True               | ok        |
