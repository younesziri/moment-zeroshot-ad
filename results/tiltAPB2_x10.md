Produced by `scripts/run_case_study.py --factor 10` (test part: 7877 points, 83 labelled anomalous).

## Scores

Random rows: mean ± sd over 1000 draws; UCR top-1: draws that hit.

| Method | Adj. best F1 (paper) | Adj. best F1, rank scores | Best F1, point-wise | AUC-PR | AUC-ROC | UCR top-1 |
|---|---|---|---|---|---|---|
| MOMENT-1-large | 0.960 | 0.689 | 0.068 | 0.027 | 0.691 | no |
| k-NN (fit part) | 0.559 | 0.514 | 0.057 | 0.019 | 0.615 | no |
| Random uniform | 0.581 ± 0.127 | 0.581 ± 0.127 | 0.029 ± 0.007 | 0.012 ± 0.002 | 0.501 ± 0.033 | 8/1000 |
| Random squared Gaussian | 0.696 ± 0.195 | 0.583 ± 0.135 | 0.029 ± 0.008 | 0.012 ± 0.002 | 0.500 ± 0.032 | 16/1000 |

## Empirical p-values

p against random uniform / random squared Gaussian: (1 + #draws scoring at least as well) / (1 + 1000).

| Method | Adj. best F1 (paper) | Adj. best F1, rank scores | Best F1, point-wise | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|
| MOMENT-1-large | 0.005 / 0.079 | 0.005 / 0.013 | 0.001 / 0.002 | 0.002 / 0.003 | 0.001 / 0.001 |
| k-NN (fit part) | 0.576 / 0.726 | 0.597 / 0.605 | 0.006 / 0.011 | 0.009 / 0.017 | 0.001 / 0.001 |

## Alerting at τ = max score on the calibration part (no labels)

Detected and false alarms use the UCR tolerance (±100 original points). Random rows: mean ± sd; Detected: draws that hit; Delay: mean over those draws.

| Method | Detected | False-alarm events | Delay (points) | Precision | Recall |
|---|---|---|---|---|---|
| MOMENT-1-large | no | 1 | – | 0.000 | 0.000 |
| k-NN (fit part) | no | 13 | – | 0.000 | 0.000 |
| Random uniform | 125/1000 | 10.993 ± 11.078 | 428.000 ± 308.642 | 0.010 ± 0.045 | 0.001 ± 0.004 |
| Random squared Gaussian | 131/1000 | 11.434 ± 11.782 | 406.718 ± 301.910 | 0.010 ± 0.044 | 0.002 ± 0.005 |
