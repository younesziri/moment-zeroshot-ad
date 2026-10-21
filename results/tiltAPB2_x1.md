Produced by `scripts/run_case_study.py --factor 1` (test part: 78767 points, 826 labelled anomalous).

## Scores

Random rows: mean ± sd over 1000 draws; UCR top-1: draws that hit.

| Method | Adj. best F1 (paper) | Adj. best F1, rank scores | Best F1, point-wise | AUC-PR | AUC-ROC | UCR top-1 |
|---|---|---|---|---|---|---|
| MOMENT-1-large | 0.922 | 0.684 | 0.050 | 0.014 | 0.524 | no |
| k-NN (fit part) | 0.738 | 0.676 | 0.057 | 0.022 | 0.711 | no |
| MOMENT-1-large, circular shift | 0.834 ± 0.138 | 0.651 ± 0.075 | 0.025 ± 0.004 | 0.011 ± 0.001 | 0.499 ± 0.036 | 11/1000 |
| k-NN (fit part), circular shift | 0.486 ± 0.403 | 0.392 ± 0.295 | 0.033 ± 0.012 | 0.013 ± 0.006 | 0.499 ± 0.212 | 11/1000 |
| Random uniform | 0.679 ± 0.025 | 0.679 ± 0.023 | 0.022 ± 0.001 | 0.011 ± 0.000 | 0.500 ± 0.010 | 9/1000 |
| Random squared Gaussian | 0.945 ± 0.047 | 0.680 ± 0.032 | 0.022 ± 0.001 | 0.011 ± 0.000 | 0.500 ± 0.010 | 13/1000 |

## Empirical p-values

p against random uniform / random squared Gaussian / the method's own scores circularly shifted (the reference null: same values, same clustering, only the alignment with the anomaly is broken): (1 + #draws scoring at least as well) / (1 + 1000).

| Method | Adj. best F1 (paper) | Adj. best F1, rank scores | Best F1, point-wise | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|
| MOMENT-1-large | 0.006 / 0.741 / 0.379 | 0.006 / 0.011 / 0.011 | 0.001 / 0.001 / 0.002 | 0.001 / 0.001 / 0.005 | 0.005 / 0.005 / 0.234 |
| k-NN (fit part) | 0.006 / 1.000 / 0.437 | 0.828 / 0.856 / 0.436 | 0.001 / 0.001 / 0.027 | 0.001 / 0.001 / 0.063 | 0.001 / 0.001 / 0.079 |

## Alerting at τ = max score on the calibration part (no labels)

Detected and false alarms use the UCR tolerance (±100 original points). Random rows: mean ± sd; Detected: draws that hit; Delay: mean over those draws.

| Method | Detected | False-alarm events | Delay (points) | Precision | Recall |
|---|---|---|---|---|---|
| MOMENT-1-large | no | 12 | – | 0.000 | 0.000 |
| k-NN (fit part) | yes | 276 | 25.000 | 0.031 | 0.199 |
| MOMENT-1-large, circular shift | 133/1000 | 11.858 ± 0.374 | 329.353 ± 293.746 | 0.009 ± 0.031 | 0.000 ± 0.001 |
| k-NN (fit part), circular shift | 585/1000 | 282.421 ± 3.605 | 57.157 ± 198.396 | 0.010 ± 0.011 | 0.065 ± 0.069 |
| Random uniform | 112/1000 | 10.996 ± 11.642 | 343.223 ± 301.752 | 0.008 ± 0.034 | 0.000 ± 0.000 |
| Random squared Gaussian | 112/1000 | 10.800 ± 11.838 | 383.179 ± 292.100 | 0.010 ± 0.053 | 0.000 ± 0.000 |
