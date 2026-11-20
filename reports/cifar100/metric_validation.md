# Metric validation (controls x metrics)

Provenance: git `1bfc31214bdc3f7253e40ff5b3fc71dde43f0dfc-dirty`, seed 0, config `/nvme/h/eioannou/code/latent-subclass-generation/configs/metric_validation.yaml`, eval config `configs/eval_generation_cifar100.yaml`, label scheme `cifar100`. conflicting images (same photo under different fine labels) are left out of every split: {'train': 68, 'val': 5, 'test': 17} (train / val / test).

Floor = c2_oracle_split (A), c2_oracle_real (B, set single_test_k5). Verdicts: ok = expected and clearly worse than the floor; MISSED = expected, not worse; also = worse though not expected; same = unchanged.

## Expected failures that were missed: 0

none

## All controls

| use_case   | control               | metric                         |     value |       lo |       hi |   floor | expected   | worse_than_floor   | verdict   |
|:-----------|:----------------------|:-------------------------------|----------:|---------:|---------:|--------:|:-----------|:-------------------|:----------|
| A          | c2_oracle_split       | G1 FID                         |   11.9378 | nan      | nan      | 11.9378 | False      | False              | floor     |
| A          | c1_copy_train         | G1 FID                         |    6.6322 | nan      | nan      | 11.9378 | False      | False              | same      |
| A          | c3_coarse_only        | G1 FID                         |    6.695  | nan      | nan      | 11.9378 | False      | False              | same      |
| A          | c4_prototype_collapse | G1 FID                         |  156.129  | nan      | nan      | 11.9378 | False      | True               | also      |
| A          | c5_majority_only      | G1 FID                         |   21.3715 | nan      | nan      | 11.9378 | False      | True               | also      |
| A          | c6_degraded           | G1 FID                         |  256.309  | nan      | nan      | 11.9378 | True       | True               | ok        |
| A          | c2_oracle_split       | G1 KID                         |   -0      | nan      | nan      | -0      | False      | False              | floor     |
| A          | c1_copy_train         | G1 KID                         |    0      | nan      | nan      | -0      | False      | False              | same      |
| A          | c3_coarse_only        | G1 KID                         |   -0      | nan      | nan      | -0      | False      | False              | same      |
| A          | c4_prototype_collapse | G1 KID                         |    0.0041 | nan      | nan      | -0      | False      | True               | also      |
| A          | c5_majority_only      | G1 KID                         |    0.0016 | nan      | nan      | -0      | False      | True               | also      |
| A          | c6_degraded           | G1 KID                         |    0.3293 | nan      | nan      | -0      | True       | True               | ok        |
| A          | c2_oracle_split       | G1 recall                      |    0.7829 | nan      | nan      |  0.7829 | False      | False              | floor     |
| A          | c1_copy_train         | G1 recall                      |    0.7604 | nan      | nan      |  0.7829 | False      | False              | same      |
| A          | c3_coarse_only        | G1 recall                      |    0.7618 | nan      | nan      |  0.7829 | False      | False              | same      |
| A          | c4_prototype_collapse | G1 recall                      |    0      | nan      | nan      |  0.7829 | True       | True               | ok        |
| A          | c5_majority_only      | G1 recall                      |    0.5259 | nan      | nan      |  0.7829 | False      | True               | also      |
| A          | c6_degraded           | G1 recall                      |    0.0013 | nan      | nan      |  0.7829 | False      | True               | also      |
| A          | c2_oracle_split       | G2 make acc                    |    0.9347 |   0.9275 |   0.9417 |  0.9347 | False      | False              | floor     |
| A          | c1_copy_train         | G2 make acc                    |    0.9995 |   0.999  |   0.9999 |  0.9347 | False      | False              | same      |
| A          | c3_coarse_only        | G2 make acc                    |    0.9991 |   0.9984 |   0.9996 |  0.9347 | False      | False              | same      |
| A          | c4_prototype_collapse | G2 make acc                    |    1      |   1      |   1      |  0.9347 | False      | False              | same      |
| A          | c5_majority_only      | G2 make acc                    |    0.9983 |   0.9974 |   0.9991 |  0.9347 | False      | False              | same      |
| A          | c6_degraded           | G2 make acc                    |    0.1583 |   0.1523 |   0.1639 |  0.9347 | False      | True               | also      |
| A          | c2_oracle_split       | G2 prevalence TV               |    0.0113 | nan      | nan      |  0.0113 | False      | False              | floor     |
| A          | c1_copy_train         | G2 prevalence TV               |    0.0003 | nan      | nan      |  0.0113 | False      | False              | same      |
| A          | c3_coarse_only        | G2 prevalence TV               |    0.0007 | nan      | nan      |  0.0113 | False      | False              | same      |
| A          | c4_prototype_collapse | G2 prevalence TV               |    0      | nan      | nan      |  0.0113 | False      | False              | same      |
| A          | c5_majority_only      | G2 prevalence TV               |    0.0007 | nan      | nan      |  0.0113 | False      | False              | same      |
| A          | c6_degraded           | G2 prevalence TV               |    0.568  | nan      | nan      |  0.0113 | False      | True               | also      |
| A          | c2_oracle_split       | G3 TV                          |    0.0232 | nan      | nan      |  0.0232 | False      | False              | floor     |
| A          | c1_copy_train         | G3 TV                          |    0.0323 | nan      | nan      |  0.0232 | False      | True               | also      |
| A          | c3_coarse_only        | G3 TV                          |    0.0363 | nan      | nan      |  0.0232 | True       | True               | ok        |
| A          | c4_prototype_collapse | G3 TV                          |    0.0408 | nan      | nan      |  0.0232 | False      | True               | also      |
| A          | c5_majority_only      | G3 TV                          |    0.799  | nan      | nan      |  0.0232 | True       | True               | ok        |
| A          | c6_degraded           | G3 TV                          |    0.3926 | nan      | nan      |  0.0232 | False      | True               | also      |
| A          | c2_oracle_split       | G3 KL                          |    0.0021 | nan      | nan      |  0.0021 | False      | False              | floor     |
| A          | c1_copy_train         | G3 KL                          |    0.003  | nan      | nan      |  0.0021 | False      | True               | also      |
| A          | c3_coarse_only        | G3 KL                          |    0.0044 | nan      | nan      |  0.0021 | True       | True               | ok        |
| A          | c4_prototype_collapse | G3 KL                          |    0.005  | nan      | nan      |  0.0021 | False      | True               | also      |
| A          | c5_majority_only      | G3 KL                          |    3.3339 | nan      | nan      |  0.0021 | True       | True               | ok        |
| A          | c6_degraded           | G3 KL                          |    0.5264 | nan      | nan      |  0.0021 | False      | True               | also      |
| A          | c2_oracle_split       | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | floor     |
| A          | c1_copy_train         | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c3_coarse_only        | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c4_prototype_collapse | G4 covered                     |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c5_majority_only      | G4 covered                     |    0.2    | nan      | nan      |  1      | True       | True               | ok        |
| A          | c6_degraded           | G4 covered                     |    0.3    | nan      | nan      |  1      | False      | True               | also      |
| A          | c2_oracle_split       | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | floor     |
| A          | c1_copy_train         | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c3_coarse_only        | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c4_prototype_collapse | G4 rare covered                |    1      | nan      | nan      |  1      | False      | False              | same      |
| A          | c5_majority_only      | G4 rare covered                |    0.2    | nan      | nan      |  1      | True       | True               | ok        |
| A          | c6_degraded           | G4 rare covered                |    0.3    | nan      | nan      |  1      | False      | True               | also      |
| A          | c2_oracle_split       | G5 excess KID                  |   -0.0041 |  -0.0105 |   0.0055 | -0.0041 | False      | False              | floor     |
| A          | c1_copy_train         | G5 excess KID                  |    0.004  |   0.002  |   0.0104 | -0.0041 | False      | False              | same      |
| A          | c3_coarse_only        | G5 excess KID                  |    0.0078 |   0.0037 |   0.0115 | -0.0041 | False      | False              | same      |
| A          | c4_prototype_collapse | G5 excess KID                  |    2.4953 |   2.4904 |   2.4963 | -0.0041 | True       | True               | ok        |
| A          | c5_majority_only      | G5 excess KID                  |    0.0068 |   0.0056 |   0.0096 | -0.0041 | False      | True               | also      |
| A          | c6_degraded           | G5 excess KID                  |    2.876  |   2.8014 |   2.9776 | -0.0041 | False      | True               | also      |
| A          | c2_oracle_split       | G5 KID ratio (ill-conditioned) |    7.6487 | nan      | nan      |  7.6487 | False      | False              | floor     |
| A          | c1_copy_train         | G5 KID ratio (ill-conditioned) |   -2.5427 | nan      | nan      |  7.6487 | False      | False              | same      |
| A          | c3_coarse_only        | G5 KID ratio (ill-conditioned) |    0.5652 | nan      | nan      |  7.6487 | False      | False              | same      |
| A          | c4_prototype_collapse | G5 KID ratio (ill-conditioned) | -153.207  | nan      | nan      |  7.6487 | False      | False              | same      |
| A          | c5_majority_only      | G5 KID ratio (ill-conditioned) |    0.2062 | nan      | nan      |  7.6487 | False      | False              | same      |
| A          | c6_degraded           | G5 KID ratio (ill-conditioned) | -174.958  | nan      | nan      |  7.6487 | False      | False              | same      |
| A          | c2_oracle_split       | G5 coverage ratio              |    1.0084 | nan      | nan      |  1.0084 | False      | False              | floor     |
| A          | c1_copy_train         | G5 coverage ratio              |    0.979  | nan      | nan      |  1.0084 | False      | False              | same      |
| A          | c3_coarse_only        | G5 coverage ratio              |    0.9762 | nan      | nan      |  1.0084 | False      | False              | same      |
| A          | c4_prototype_collapse | G5 coverage ratio              |    0.0653 | nan      | nan      |  1.0084 | True       | True               | ok        |
| A          | c5_majority_only      | G5 coverage ratio              |    0.9904 | nan      | nan      |  1.0084 | False      | False              | same      |
| A          | c6_degraded           | G5 coverage ratio              |    0.0116 | nan      | nan      |  1.0084 | False      | True               | also      |
| A          | c2_oracle_split       | G7 copies                      |    0.7141 | nan      | nan      |  0.7141 | False      | False              | floor     |
| A          | c1_copy_train         | G7 copies                      |    1      | nan      | nan      |  0.7141 | True       | True               | ok        |
| A          | c3_coarse_only        | G7 copies                      |    1      | nan      | nan      |  0.7141 | False      | True               | also      |
| A          | c4_prototype_collapse | G7 copies                      |    1      | nan      | nan      |  0.7141 | False      | True               | also      |
| A          | c5_majority_only      | G7 copies                      |    1      | nan      | nan      |  0.7141 | False      | True               | also      |
| A          | c6_degraded           | G7 copies                      |    0.0033 | nan      | nan      |  0.7141 | False      | False              | same      |
| B          | c2_oracle_real        | B1 hit rate                    |    0.9991 |   0.9985 |   0.9996 |  0.9991 | False      | False              | floor     |
| B          | c1b_copy_support      | B1 hit rate                    |    0.8518 |   0.84   |   0.8638 |  0.9991 | False      | True               | also      |
| B          | c1b_retrieve_nn       | B1 hit rate                    |    0.9485 |   0.9398 |   0.9571 |  0.9991 | False      | True               | also      |
| B          | c4_prototype_collapse | B1 hit rate                    |    1      |   1      |   1      |  0.9991 | False      | False              | same      |
| B          | c2_oracle_real        | B2 KID vs T                    |    0.003  |  -0.0017 |   0.0079 |  0.003  | False      | False              | floor     |
| B          | c1b_copy_support      | B2 KID vs T                    |    0.4996 |   0.4846 |   0.5174 |  0.003  | True       | True               | ok        |
| B          | c1b_retrieve_nn       | B2 KID vs T                    |    0.2883 |   0.2702 |   0.3073 |  0.003  | True       | True               | ok        |
| B          | c4_prototype_collapse | B2 KID vs T                    |    2.4942 |   2.4464 |   2.5418 |  0.003  | False      | True               | also      |
| B          | c2_oracle_real        | B2 coverage vs T               |    0.3857 |   0.3786 |   0.3937 |  0.3857 | False      | False              | floor     |
| B          | c1b_copy_support      | B2 coverage vs T               |    0.2724 |   0.2641 |   0.2801 |  0.3857 | False      | True               | also      |
| B          | c1b_retrieve_nn       | B2 coverage vs T               |    0.4086 |   0.401  |   0.4169 |  0.3857 | False      | False              | same      |
| B          | c4_prototype_collapse | B2 coverage vs T               |    0.0603 |   0.0564 |   0.0647 |  0.3857 | False      | True               | also      |
| B          | c2_oracle_real        | B3 ratio to S                  |    1.0031 |   0.9994 |   1.0069 |  1.0031 | False      | False              | floor     |
| B          | c1b_copy_support      | B3 ratio to S                  |    2.1141 |   2.0996 |   2.1282 |  1.0031 | True       | True               | ok        |
| B          | c1b_retrieve_nn       | B3 ratio to S                  |    1.1572 |   1.1482 |   1.1669 |  1.0031 | True       | True               | ok        |
| B          | c4_prototype_collapse | B3 ratio to S                  |    1.1083 |   1.0942 |   1.1225 |  1.0031 | False      | True               | also      |
| B          | c2_oracle_real        | B4 diversity ratio             |    1.0017 |   0.9947 |   1.009  |  1.0017 | False      | False              | floor     |
| B          | c1b_copy_support      | B4 diversity ratio             |    0.8131 |   0.8047 |   0.822  |  1.0017 | False      | True               | also      |
| B          | c1b_retrieve_nn       | B4 diversity ratio             |    0.7919 |   0.7825 |   0.8003 |  1.0017 | False      | True               | also      |
| B          | c4_prototype_collapse | B4 diversity ratio             |    0      |  -0      |   0      |  1.0017 | True       | True               | ok        |
