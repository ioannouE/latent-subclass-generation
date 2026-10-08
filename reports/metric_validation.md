# Metric validation (controls x metrics)

Provenance: git `f9f4e60b16d2e9a412c28856208e8f822bb431eb-dirty`, seed 0, config `configs/metric_validation.yaml`. conflicting images (same photo under different fine labels) are left out of every split: {'train': 59, 'val': 8, 'test': 87} (train / val / test).

Floor = c2_oracle_split (A), c2_oracle_real (B, set single_test_k5). Verdicts: ok = expected and clearly worse than the floor; MISSED = expected, not worse; also = worse though not expected; same = unchanged.

## Expected failures that were missed: 0

none

## All controls

| use_case   | control               | metric                         |     value |       lo |       hi |   floor | expected   | worse_than_floor   | verdict   |
|:-----------|:----------------------|:-------------------------------|----------:|---------:|---------:|--------:|:-----------|:-------------------|:----------|
| A          | c2_oracle_split       | G1 FID                         |    4.9742 | nan      | nan      |  4.9742 | False      | False              | floor     |
| A          | c1_copy_train         | G1 FID                         |    3.7779 | nan      | nan      |  4.9742 | False      | False              | same      |
| A          | c3_coarse_only        | G1 FID                         |    3.8602 | nan      | nan      |  4.9742 | False      | False              | same      |
| A          | c4_prototype_collapse | G1 FID                         |   39.8363 | nan      | nan      |  4.9742 | False      | True               | also      |
| A          | c5_majority_only      | G1 FID                         |   10.6212 | nan      | nan      |  4.9742 | False      | True               | also      |
| A          | c6_degraded           | G1 FID                         |  103.857  | nan      | nan      |  4.9742 | True       | True               | ok        |
| A          | c2_oracle_split       | G1 KID                         |   -0      | nan      | nan      | -0      | False      | False              | floor     |
| A          | c1_copy_train         | G1 KID                         |    0      | nan      | nan      | -0      | False      | False              | same      |
| A          | c3_coarse_only        | G1 KID                         |    0      | nan      | nan      | -0      | False      | False              | same      |
| A          | c4_prototype_collapse | G1 KID                         |    0.0026 | nan      | nan      | -0      | False      | True               | also      |
| A          | c5_majority_only      | G1 KID                         |    0.0011 | nan      | nan      | -0      | False      | True               | also      |
| A          | c6_degraded           | G1 KID                         |    0.0939 | nan      | nan      | -0      | True       | True               | ok        |
| A          | c2_oracle_split       | G1 recall                      |    0.7986 | nan      | nan      |  0.7986 | False      | False              | floor     |
| A          | c1_copy_train         | G1 recall                      |    0.7101 | nan      | nan      |  0.7986 | False      | False              | same      |
| A          | c3_coarse_only        | G1 recall                      |    0.7038 | nan      | nan      |  0.7986 | False      | False              | same      |
| A          | c4_prototype_collapse | G1 recall                      |    0.0001 | nan      | nan      |  0.7986 | True       | True               | ok        |
| A          | c5_majority_only      | G1 recall                      |    0.4842 | nan      | nan      |  0.7986 | False      | True               | also      |
| A          | c6_degraded           | G1 recall                      |    0.0094 | nan      | nan      |  0.7986 | False      | True               | also      |
| A          | c2_oracle_split       | G2 make acc                    |    0.8774 |   0.8625 |   0.8911 |  0.8774 | False      | False              | floor     |
| A          | c1_copy_train         | G2 make acc                    |    0.9993 |   0.9988 |   0.9997 |  0.8774 | False      | False              | same      |
| A          | c3_coarse_only        | G2 make acc                    |    0.9994 |   0.9989 |   0.9998 |  0.8774 | False      | False              | same      |
| A          | c4_prototype_collapse | G2 make acc                    |    1      |   1      |   1      |  0.8774 | False      | False              | same      |
| A          | c5_majority_only      | G2 make acc                    |    0.9998 |   0.9994 |   1      |  0.8774 | False      | False              | same      |
| A          | c6_degraded           | G2 make acc                    |    0.1212 |   0.1136 |   0.1297 |  0.8774 | False      | True               | also      |
| A          | c2_oracle_split       | G2 prevalence TV               |    0.0261 | nan      | nan      |  0.0261 | False      | False              | floor     |
| A          | c1_copy_train         | G2 prevalence TV               |    0.0008 | nan      | nan      |  0.0261 | False      | False              | same      |
| A          | c3_coarse_only        | G2 prevalence TV               |    0.001  | nan      | nan      |  0.0261 | False      | False              | same      |
| A          | c4_prototype_collapse | G2 prevalence TV               |    0      | nan      | nan      |  0.0261 | False      | False              | same      |
| A          | c5_majority_only      | G2 prevalence TV               |    0.0004 | nan      | nan      |  0.0261 | False      | False              | same      |
| A          | c6_degraded           | G2 prevalence TV               |    0.5476 | nan      | nan      |  0.0261 | False      | True               | also      |
| A          | c2_oracle_split       | G3 TV                          |    0.0192 | nan      | nan      |  0.0192 | False      | False              | floor     |
| A          | c1_copy_train         | G3 TV                          |    0.0287 | nan      | nan      |  0.0192 | False      | True               | also      |
| A          | c3_coarse_only        | G3 TV                          |    0.0483 | nan      | nan      |  0.0192 | True       | True               | ok        |
| A          | c4_prototype_collapse | G3 TV                          |    0.0281 | nan      | nan      |  0.0192 | False      | True               | also      |
| A          | c5_majority_only      | G3 TV                          |    0.4193 | nan      | nan      |  0.0192 | True       | True               | ok        |
| A          | c6_degraded           | G3 TV                          |    0.1569 | nan      | nan      |  0.0192 | False      | True               | also      |
| A          | c2_oracle_split       | G3 KL                          |    0.0024 | nan      | nan      |  0.0024 | False      | False              | floor     |
| A          | c1_copy_train         | G3 KL                          |    0.0041 | nan      | nan      |  0.0024 | False      | True               | also      |
| A          | c3_coarse_only        | G3 KL                          |    0.0138 | nan      | nan      |  0.0024 | True       | True               | ok        |
| A          | c4_prototype_collapse | G3 KL                          |    0.0043 | nan      | nan      |  0.0024 | False      | True               | also      |
| A          | c5_majority_only      | G3 KL                          |    1.3587 | nan      | nan      |  0.0024 | True       | True               | ok        |
| A          | c6_degraded           | G3 KL                          |    0.1694 | nan      | nan      |  0.0024 | False      | True               | also      |
| A          | c2_oracle_split       | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | floor     |
| A          | c1_copy_train         | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c3_coarse_only        | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c4_prototype_collapse | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c5_majority_only      | G4 covered                     |    0.5539 | nan      | nan      |  1      | True       | True               | ok        |
| A          | c6_degraded           | G4 covered                     |    0.2077 | nan      | nan      |  1      | False      | True               | also      |
| A          | c2_oracle_split       | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | floor     |
| A          | c1_copy_train         | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c3_coarse_only        | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c4_prototype_collapse | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c5_majority_only      | G4 rare covered                |    0.2143 | nan      | nan      |  1      | True       | True               | ok        |
| A          | c6_degraded           | G4 rare covered                |    0.1063 | nan      | nan      |  1      | False      | True               | also      |
| A          | c2_oracle_split       | G5 excess KID                  |    0.0185 |   0.0039 |   0.0351 |  0.0185 | False      | False              | floor     |
| A          | c1_copy_train         | G5 excess KID                  |    0.0678 |   0.0622 |   0.0755 |  0.0185 | False      | True               | also      |
| A          | c3_coarse_only        | G5 excess KID                  |    0.0656 |   0.0606 |   0.0734 |  0.0185 | False      | True               | also      |
| A          | c4_prototype_collapse | G5 excess KID                  |    1.9904 |   1.989  |   1.9933 |  0.0185 | True       | True               | ok        |
| A          | c5_majority_only      | G5 excess KID                  |    0.0589 |   0.054  |   0.0652 |  0.0185 | False      | True               | also      |
| A          | c6_degraded           | G5 excess KID                  |    4.5318 |   4.1979 |   4.4672 |  0.0185 | False      | True               | also      |
| A          | c2_oracle_split       | G5 KID ratio (ill-conditioned) |    0.5282 | nan      | nan      |  0.5282 | False      | False              | floor     |
| A          | c1_copy_train         | G5 KID ratio (ill-conditioned) |    2.7103 | nan      | nan      |  0.5282 | False      | True               | also      |
| A          | c3_coarse_only        | G5 KID ratio (ill-conditioned) |   -1.0299 | nan      | nan      |  0.5282 | False      | False              | same      |
| A          | c4_prototype_collapse | G5 KID ratio (ill-conditioned) |  -26.1581 | nan      | nan      |  0.5282 | False      | False              | same      |
| A          | c5_majority_only      | G5 KID ratio (ill-conditioned) |    1.5383 | nan      | nan      |  0.5282 | False      | True               | also      |
| A          | c6_degraded           | G5 KID ratio (ill-conditioned) | 1025      | nan      | nan      |  0.5282 | False      | True               | also      |
| A          | c2_oracle_split       | G5 coverage ratio              |    1.0328 | nan      | nan      |  1.0328 | False      | False              | floor     |
| A          | c1_copy_train         | G5 coverage ratio              |    0.886  | nan      | nan      |  1.0328 | False      | False              | same      |
| A          | c3_coarse_only        | G5 coverage ratio              |    0.8931 | nan      | nan      |  1.0328 | False      | False              | same      |
| A          | c4_prototype_collapse | G5 coverage ratio              |    0.1476 | nan      | nan      |  1.0328 | True       | True               | ok        |
| A          | c5_majority_only      | G5 coverage ratio              |    0.9493 | nan      | nan      |  1.0328 | False      | False              | same      |
| A          | c6_degraded           | G5 coverage ratio              |    0      | nan      | nan      |  1.0328 | False      | True               | also      |
| A          | c2_oracle_split       | G7 copies                      |    0.1356 | nan      | nan      |  0.1356 | False      | False              | floor     |
| A          | c1_copy_train         | G7 copies                      |    1      | nan      | nan      |  0.1356 | True       | True               | ok        |
| A          | c3_coarse_only        | G7 copies                      |    1      | nan      | nan      |  0.1356 | False      | True               | also      |
| A          | c4_prototype_collapse | G7 copies                      |    1      | nan      | nan      |  0.1356 | False      | True               | also      |
| A          | c5_majority_only      | G7 copies                      |    1      | nan      | nan      |  0.1356 | False      | True               | also      |
| A          | c6_degraded           | G7 copies                      |    0.3982 | nan      | nan      |  0.1356 | False      | True               | also      |
| B          | c2_oracle_real        | B1 hit rate                    |    0.9997 |   0.9995 |   0.9999 |  0.9997 | False      | False              | floor     |
| B          | c1b_copy_support      | B1 hit rate                    |    0.8771 |   0.8679 |   0.8866 |  0.9997 | False      | True               | also      |
| B          | c1b_retrieve_nn       | B1 hit rate                    |    0.9204 |   0.914  |   0.9267 |  0.9997 | False      | True               | also      |
| B          | c4_prototype_collapse | B1 hit rate                    |    1      |   1      |   1      |  0.9997 | False      | False              | same      |
| B          | c2_oracle_real        | B2 KID vs T                    |   -0.0002 |  -0.0035 |   0.0032 | -0.0002 | False      | False              | floor     |
| B          | c1b_copy_support      | B2 KID vs T                    |    0.4775 |   0.4662 |   0.4874 | -0.0002 | True       | True               | ok        |
| B          | c1b_retrieve_nn       | B2 KID vs T                    |    0.093  |   0.0856 |   0.0998 | -0.0002 | True       | True               | ok        |
| B          | c4_prototype_collapse | B2 KID vs T                    |    1.9889 |   1.9592 |   2.019  | -0.0002 | False      | True               | also      |
| B          | c2_oracle_real        | B2 coverage vs T               |    0.6777 |   0.6675 |   0.6874 |  0.6777 | False      | False              | floor     |
| B          | c1b_copy_support      | B2 coverage vs T               |    0.3174 |   0.3062 |   0.329  |  0.6777 | False      | True               | also      |
| B          | c1b_retrieve_nn       | B2 coverage vs T               |    0.6971 |   0.6895 |   0.7049 |  0.6777 | False      | False              | same      |
| B          | c4_prototype_collapse | B2 coverage vs T               |    0.1485 |   0.1427 |   0.1542 |  0.6777 | False      | True               | also      |
| B          | c2_oracle_real        | B3 ratio to S                  |    1.0045 |   0.9956 |   1.0136 |  1.0045 | False      | False              | floor     |
| B          | c1b_copy_support      | B3 ratio to S                  |    3.7882 |   3.7609 |   3.8146 |  1.0045 | True       | True               | ok        |
| B          | c1b_retrieve_nn       | B3 ratio to S                  |    1.1087 |   1.0985 |   1.1194 |  1.0045 | True       | True               | ok        |
| B          | c4_prototype_collapse | B3 ratio to S                  |    1.1789 |   1.1507 |   1.2074 |  1.0045 | False      | True               | also      |
| B          | c2_oracle_real        | B4 diversity ratio             |    1.0284 |   1.0208 |   1.036  |  1.0284 | False      | False              | floor     |
| B          | c1b_copy_support      | B4 diversity ratio             |    0.9443 |   0.9317 |   0.9584 |  1.0284 | False      | True               | also      |
| B          | c1b_retrieve_nn       | B4 diversity ratio             |    0.8738 |   0.8656 |   0.8831 |  1.0284 | False      | True               | also      |
| B          | c4_prototype_collapse | B4 diversity ratio             |    0      |   0      |   0      |  1.0284 | True       | True               | ok        |
