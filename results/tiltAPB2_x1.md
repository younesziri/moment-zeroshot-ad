Produced by `scripts/run_case_study.py --factor 1` (test part: 78767 points, 826 labelled anomalous).

## Scores

Random rows: mean ± sd over 1000 draws; UCR top-1: draws that hit.

| Method | Adj. best F1 (paper) | Adj. best F1, rank scores | Best F1, point-wise | AUC-PR | AUC-ROC | UCR top-1 |
|---|---|---|---|---|---|---|
| MOMENT-1-large | 0.922 | 0.684 | 0.050 | 0.014 | 0.524 | no |
| k-NN (fit part) | 0.738 | 0.676 | 0.057 | 0.022 | 0.711 | no |
| Random uniform | 0.679 ± 0.025 | 0.679 ± 0.023 | 0.022 ± 0.001 | 0.011 ± 0.000 | 0.500 ± 0.010 | 9/1000 |
| Random squared Gaussian | 0.945 ± 0.047 | 0.680 ± 0.032 | 0.022 ± 0.001 | 0.011 ± 0.000 | 0.500 ± 0.010 | 13/1000 |

## Empirical p-values

p against random uniform / random squared Gaussian: (1 + #draws scoring at least as well) / (1 + 1000).

| Method | Adj. best F1 (paper) | Adj. best F1, rank scores | Best F1, point-wise | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|
| MOMENT-1-large | 0.006 / 0.741 | 0.006 / 0.011 | 0.001 / 0.001 | 0.001 / 0.001 | 0.005 / 0.005 |
| k-NN (fit part) | 0.006 / 1.000 | 0.828 / 0.856 | 0.001 / 0.001 | 0.001 / 0.001 | 0.001 / 0.001 |

## Alerting at τ = max score on the calibration part (no labels)

Detected and false alarms use the UCR tolerance (±100 original points). Random rows: mean ± sd; Detected: draws that hit; Delay: mean over those draws.

| Method | Detected | False-alarm events | Delay (points) | Precision | Recall |
|---|---|---|---|---|---|
| MOMENT-1-large | no | 12 | – | 0.000 | 0.000 |
| k-NN (fit part) | yes | 276 | 25.000 | 0.031 | 0.199 |
| Random uniform | 112/1000 | 10.996 ± 11.642 | 343.223 ± 301.752 | 0.008 ± 0.034 | 0.000 ± 0.000 |
| Random squared Gaussian | 112/1000 | 10.800 ± 11.838 | 383.179 ± 292.100 | 0.010 ± 0.053 | 0.000 ± 0.000 |
