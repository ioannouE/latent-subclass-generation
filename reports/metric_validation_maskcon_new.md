# Metric validation (controls x metrics)

Provenance: git `1bfc31214bdc3f7253e40ff5b3fc71dde43f0dfc-dirty`, seed 0, config `/nvme/h/eioannou/code/latent-subclass-generation/configs/metric_validation.yaml`, eval config `configs/eval_generation_maskcon_new.yaml`, label scheme `maskcon_new`. conflicting images (same photo under different fine labels) are left out of every split: {'train': 98, 'val': 12, 'test': 101} (train / val / test).

Floor = c2_oracle_split (A), c2_oracle_real (B, set single_test_k5). Verdicts: ok = expected and clearly worse than the floor; MISSED = expected, not worse; also = worse though not expected; same = unchanged.

## Expected failures that were missed: 0

none

## All controls

| use_case   | control               | metric                         |    value |       lo |       hi |   floor | expected   | worse_than_floor   | verdict   |
|:-----------|:----------------------|:-------------------------------|---------:|---------:|---------:|--------:|:-----------|:-------------------|:----------|
| A          | c2_oracle_split       | G1 FID                         |   4.9897 | nan      | nan      |  4.9897 | False      | False              | floor     |
| A          | c1_copy_train         | G1 FID                         |   3.8826 | nan      | nan      |  4.9897 | False      | False              | same      |
| A          | c3_coarse_only        | G1 FID                         |   3.8477 | nan      | nan      |  4.9897 | False      | False              | same      |
| A          | c4_prototype_collapse | G1 FID                         |  39.1664 | nan      | nan      |  4.9897 | False      | True               | also      |
| A          | c5_majority_only      | G1 FID                         |  34.7541 | nan      | nan      |  4.9897 | False      | True               | also      |
| A          | c6_degraded           | G1 FID                         | 103.249  | nan      | nan      |  4.9897 | True       | True               | ok        |
| A          | c2_oracle_split       | G1 KID                         |   0      | nan      | nan      |  0      | False      | False              | floor     |
| A          | c1_copy_train         | G1 KID                         |   0      | nan      | nan      |  0      | False      | False              | same      |
| A          | c3_coarse_only        | G1 KID                         |  -0      | nan      | nan      |  0      | False      | False              | same      |
| A          | c4_prototype_collapse | G1 KID                         |   0.0024 | nan      | nan      |  0      | False      | True               | also      |
| A          | c5_majority_only      | G1 KID                         |   0.002  | nan      | nan      |  0      | False      | True               | also      |
| A          | c6_degraded           | G1 KID                         |   0.0934 | nan      | nan      |  0      | True       | True               | ok        |
| A          | c2_oracle_split       | G1 recall                      |   0.7975 | nan      | nan      |  0.7975 | False      | False              | floor     |
| A          | c1_copy_train         | G1 recall                      |   0.7068 | nan      | nan      |  0.7975 | False      | False              | same      |
| A          | c3_coarse_only        | G1 recall                      |   0.7111 | nan      | nan      |  0.7975 | False      | False              | same      |
| A          | c4_prototype_collapse | G1 recall                      |   0      | nan      | nan      |  0.7975 | True       | True               | ok        |
| A          | c5_majority_only      | G1 recall                      |   0.0447 | nan      | nan      |  0.7975 | False      | True               | also      |
| A          | c6_degraded           | G1 recall                      |   0.0102 | nan      | nan      |  0.7975 | False      | True               | also      |
| A          | c2_oracle_split       | G2 make acc                    |   0.8551 |   0.842  |   0.87   |  0.8551 | False      | False              | floor     |
| A          | c1_copy_train         | G2 make acc                    |   0.999  |   0.9984 |   0.9996 |  0.8551 | False      | False              | same      |
| A          | c3_coarse_only        | G2 make acc                    |   0.9993 |   0.9987 |   0.9997 |  0.8551 | False      | False              | same      |
| A          | c4_prototype_collapse | G2 make acc                    |   1      |   1      |   1      |  0.8551 | False      | False              | same      |
| A          | c5_majority_only      | G2 make acc                    |   1      |   1      |   1      |  0.8551 | False      | False              | same      |
| A          | c6_degraded           | G2 make acc                    |   0.3837 |   0.3725 |   0.3948 |  0.8551 | False      | True               | also      |
| A          | c2_oracle_split       | G2 prevalence TV               |   0.029  | nan      | nan      |  0.029  | False      | False              | floor     |
| A          | c1_copy_train         | G2 prevalence TV               |   0.0008 | nan      | nan      |  0.029  | False      | False              | same      |
| A          | c3_coarse_only        | G2 prevalence TV               |   0.0005 | nan      | nan      |  0.029  | False      | False              | same      |
| A          | c4_prototype_collapse | G2 prevalence TV               |   0      | nan      | nan      |  0.029  | False      | False              | same      |
| A          | c5_majority_only      | G2 prevalence TV               |   0      | nan      | nan      |  0.029  | False      | False              | same      |
| A          | c6_degraded           | G2 prevalence TV               |   0.2304 | nan      | nan      |  0.029  | False      | True               | also      |
| A          | c2_oracle_split       | G3 TV                          |   0.0493 | nan      | nan      |  0.0493 | False      | False              | floor     |
| A          | c1_copy_train         | G3 TV                          |   0.0544 | nan      | nan      |  0.0493 | False      | False              | same      |
| A          | c3_coarse_only        | G3 TV                          |   0.0724 | nan      | nan      |  0.0493 | True       | True               | ok        |
| A          | c4_prototype_collapse | G3 TV                          |   0.0598 | nan      | nan      |  0.0493 | False      | False              | same      |
| A          | c5_majority_only      | G3 TV                          |   0.9265 | nan      | nan      |  0.0493 | True       | True               | ok        |
| A          | c6_degraded           | G3 TV                          |   0.4966 | nan      | nan      |  0.0493 | False      | True               | also      |
| A          | c2_oracle_split       | G3 KL                          |   0.0075 | nan      | nan      |  0.0075 | False      | False              | floor     |
| A          | c1_copy_train         | G3 KL                          |   0.0096 | nan      | nan      |  0.0075 | False      | True               | also      |
| A          | c3_coarse_only        | G3 KL                          |   0.0168 | nan      | nan      |  0.0075 | True       | True               | ok        |
| A          | c4_prototype_collapse | G3 KL                          |   0.0112 | nan      | nan      |  0.0075 | False      | True               | also      |
| A          | c5_majority_only      | G3 KL                          |   3.2728 | nan      | nan      |  0.0075 | True       | True               | ok        |
| A          | c6_degraded           | G3 KL                          |   0.8663 | nan      | nan      |  0.0075 | False      | True               | also      |
| A          | c2_oracle_split       | G4 covered                     |   1      | nan      | nan      |  1      | False      | False              | floor     |
| A          | c1_copy_train         | G4 covered                     |   1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c3_coarse_only        | G4 covered                     |   1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c4_prototype_collapse | G4 covered                     |   1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c5_majority_only      | G4 covered                     |   0.0608 | nan      | nan      |  1      | True       | True               | ok        |
| A          | c6_degraded           | G4 covered                     |   0.1472 | nan      | nan      |  1      | False      | True               | also      |
| A          | c2_oracle_split       | G4 rare covered                |   1      | nan      | nan      |  1      | False      | False              | floor     |
| A          | c1_copy_train         | G4 rare covered                |   1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c3_coarse_only        | G4 rare covered                |   1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c4_prototype_collapse | G4 rare covered                |   1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c5_majority_only      | G4 rare covered                |   0      | nan      | nan      |  1      | True       | True               | ok        |
| A          | c6_degraded           | G4 rare covered                |   0.156  | nan      | nan      |  1      | False      | True               | also      |
| A          | c2_oracle_split       | G5 excess KID                  |   0.0052 |  -0.006  |   0.0236 |  0.0052 | False      | False              | floor     |
| A          | c1_copy_train         | G5 excess KID                  |   0.0753 |   0.0691 |   0.0826 |  0.0052 | False      | True               | also      |
| A          | c3_coarse_only        | G5 excess KID                  |   0.0703 |   0.0636 |   0.0774 |  0.0052 | False      | True               | also      |
| A          | c4_prototype_collapse | G5 excess KID                  |   2.0863 |   2.0831 |   2.0907 |  0.0052 | True       | True               | ok        |
| A          | c5_majority_only      | G5 excess KID                  |   0.0594 |   0.0566 |   0.0627 |  0.0052 | False      | True               | also      |
| A          | c6_degraded           | G5 excess KID                  |   4.6897 |   4.4076 |   4.723  |  0.0052 | False      | True               | also      |
| A          | c2_oracle_split       | G5 KID ratio (ill-conditioned) |   1.6919 | nan      | nan      |  1.6919 | False      | False              | floor     |
| A          | c1_copy_train         | G5 KID ratio (ill-conditioned) |  22.4252 | nan      | nan      |  1.6919 | False      | True               | also      |
| A          | c3_coarse_only        | G5 KID ratio (ill-conditioned) |  -4.7176 | nan      | nan      |  1.6919 | False      | False              | same      |
| A          | c4_prototype_collapse | G5 KID ratio (ill-conditioned) | 917.997  | nan      | nan      |  1.6919 | False      | True               | also      |
| A          | c5_majority_only      | G5 KID ratio (ill-conditioned) |  -2.3494 | nan      | nan      |  1.6919 | False      | False              | same      |
| A          | c6_degraded           | G5 KID ratio (ill-conditioned) | -55.9604 | nan      | nan      |  1.6919 | False      | False              | same      |
| A          | c2_oracle_split       | G5 coverage ratio              |   1.0306 | nan      | nan      |  1.0306 | False      | False              | floor     |
| A          | c1_copy_train         | G5 coverage ratio              |   0.8888 | nan      | nan      |  1.0306 | False      | False              | same      |
| A          | c3_coarse_only        | G5 coverage ratio              |   0.8936 | nan      | nan      |  1.0306 | False      | False              | same      |
| A          | c4_prototype_collapse | G5 coverage ratio              |   0.1519 | nan      | nan      |  1.0306 | True       | True               | ok        |
| A          | c5_majority_only      | G5 coverage ratio              |   1      | nan      | nan      |  1.0306 | False      | False              | same      |
| A          | c6_degraded           | G5 coverage ratio              |   0      | nan      | nan      |  1.0306 | False      | True               | also      |
| A          | c2_oracle_split       | G7 copies                      |   0.1343 | nan      | nan      |  0.1343 | False      | False              | floor     |
| A          | c1_copy_train         | G7 copies                      |   1      | nan      | nan      |  0.1343 | True       | True               | ok        |
| A          | c3_coarse_only        | G7 copies                      |   1      | nan      | nan      |  0.1343 | False      | True               | also      |
| A          | c4_prototype_collapse | G7 copies                      |   1      | nan      | nan      |  0.1343 | False      | True               | also      |
| A          | c5_majority_only      | G7 copies                      |   1      | nan      | nan      |  0.1343 | False      | True               | also      |
| A          | c6_degraded           | G7 copies                      |   0.4003 | nan      | nan      |  0.1343 | False      | True               | also      |
| B          | c2_oracle_real        | B1 hit rate                    |   1      |   1      |   1      |  1      | False      | False              | floor     |
| B          | c1b_copy_support      | B1 hit rate                    |   0.8476 |   0.8355 |   0.8592 |  1      | False      | True               | also      |
| B          | c1b_retrieve_nn       | B1 hit rate                    |   0.8643 |   0.8492 |   0.8802 |  1      | False      | True               | also      |
| B          | c4_prototype_collapse | B1 hit rate                    |   1      |   1      |   1      |  1      | False      | False              | same      |
| B          | c2_oracle_real        | B2 KID vs T                    |   0.0019 |  -0.0016 |   0.0058 |  0.0019 | False      | False              | floor     |
| B          | c1b_copy_support      | B2 KID vs T                    |   0.4832 |   0.472  |   0.4943 |  0.0019 | True       | True               | ok        |
| B          | c1b_retrieve_nn       | B2 KID vs T                    |   0.0907 |   0.0827 |   0.098  |  0.0019 | True       | True               | ok        |
| B          | c4_prototype_collapse | B2 KID vs T                    |   2.0884 |   2.0171 |   2.16   |  0.0019 | False      | True               | also      |
| B          | c2_oracle_real        | B2 coverage vs T               |   0.6874 |   0.6769 |   0.6986 |  0.6874 | False      | False              | floor     |
| B          | c1b_copy_support      | B2 coverage vs T               |   0.3263 |   0.3155 |   0.3366 |  0.6874 | False      | True               | also      |
| B          | c1b_retrieve_nn       | B2 coverage vs T               |   0.701  |   0.6905 |   0.7109 |  0.6874 | False      | False              | same      |
| B          | c4_prototype_collapse | B2 coverage vs T               |   0.152  |   0.1443 |   0.1597 |  0.6874 | False      | True               | also      |
| B          | c2_oracle_real        | B3 ratio to S                  |   1.006  |   0.9975 |   1.014  |  1.006  | False      | False              | floor     |
| B          | c1b_copy_support      | B3 ratio to S                  |   3.7281 |   3.6971 |   3.7608 |  1.006  | True       | True               | ok        |
| B          | c1b_retrieve_nn       | B3 ratio to S                  |   1.1165 |   1.1064 |   1.1263 |  1.006  | True       | True               | ok        |
| B          | c4_prototype_collapse | B3 ratio to S                  |   1.1531 |   1.1227 |   1.1834 |  1.006  | False      | True               | also      |
| B          | c2_oracle_real        | B4 diversity ratio             |   1.0247 |   1.0163 |   1.0335 |  1.0247 | False      | False              | floor     |
| B          | c1b_copy_support      | B4 diversity ratio             |   0.9408 |   0.9297 |   0.9543 |  1.0247 | False      | True               | also      |
| B          | c1b_retrieve_nn       | B4 diversity ratio             |   0.8776 |   0.8691 |   0.8859 |  1.0247 | False      | True               | also      |
| B          | c4_prototype_collapse | B4 diversity ratio             |   0      |  -0      |   0      |  1.0247 | True       | True               | ok        |
