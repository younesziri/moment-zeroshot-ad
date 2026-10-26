Produced by `scripts/analyse_248.py` at commit 7d46888b401f92ac5729d543d0d784a8c02a351c.

## Coverage

- Series with results: 248; the paper's per-part downsampling rule would differ from ours (one factor per series): 121; calibration part shorter than 512: 130; no autocorrelation peak (L = 512): 0.
- Failures (13):

| id | configuration | detector | error |
|---|---|---|---|
| 15 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |
| 56 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |
| 57 | consistent | k-NN | ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 5, n_samples_fit = 4, n_samples = 7 |
| 74 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |
| 75 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |
| 104 | consistent | k-NN | ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 5, n_samples_fit = 4, n_samples = 7 |
| 123 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |
| 164 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |
| 165 | consistent | k-NN | ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 5, n_samples_fit = 4, n_samples = 7 |
| 182 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |
| 183 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |
| 207 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |
| 208 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |

- Alerting missing, other metrics kept (D10) (18):

| id | detector | alerting_missing |
|---|---|---|
| 15 | MOMENT | ValueError: not enough history (429 points) to score a part of 72 |
| 15 | MOMENT masked | ValueError: not enough history (429 points) to score a part of 72 |
| 56 | MOMENT | ValueError: not enough history (429 points) to score a part of 72 |
| 56 | MOMENT masked | ValueError: not enough history (429 points) to score a part of 72 |
| 74 | MOMENT | ValueError: not enough history (343 points) to score a part of 58 |
| 74 | MOMENT masked | ValueError: not enough history (343 points) to score a part of 58 |
| 75 | MOMENT | ValueError: not enough history (343 points) to score a part of 58 |
| 75 | MOMENT masked | ValueError: not enough history (343 points) to score a part of 58 |
| 123 | MOMENT | ValueError: not enough history (429 points) to score a part of 72 |
| 123 | MOMENT masked | ValueError: not enough history (429 points) to score a part of 72 |
| 164 | MOMENT | ValueError: not enough history (429 points) to score a part of 72 |
| 164 | MOMENT masked | ValueError: not enough history (429 points) to score a part of 72 |
| 182 | MOMENT | ValueError: not enough history (343 points) to score a part of 58 |
| 182 | MOMENT masked | ValueError: not enough history (343 points) to score a part of 58 |
| 183 | MOMENT | ValueError: not enough history (343 points) to score a part of 58 |
| 183 | MOMENT masked | ValueError: not enough history (343 points) to score a part of 58 |
| 207 | MOMENT | ValueError: not enough history (272 points) to score a part of 46 |
| 207 | MOMENT masked | ValueError: not enough history (272 points) to score a part of 46 |

## Family 1 (Q1–Q2), fixed sequence at α = 0.05

Significant = clustered bootstrap 95% interval excludes 0 in MOMENT's favour and the two-sided p < 0.05. The one-sided p is reported, not used for decisions.

| test | n | mean | CI clustered | CI plain | p | p one-sided (in favour) | significant | confirmatory | hits | expected |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. VUS-PR, MOMENT vs chance | 248 | 0.019 | [0.0121, 0.0266] | [0.0149, 0.0243] | < 0.001 | < 0.001 | True | yes | – | – |
| 2. VUS-PR, MOMENT vs k-NN (difference of excesses) | 235 | -0.074 | [-0.1420, -0.0137] | [-0.0961, -0.0529] | 0.096 | 0.955 | False | no | – | – |
| 3. top-1, MOMENT vs chance (|Z|/n) | 248 | 0.178 | [0.0819, 0.2933] | [0.1291, 0.2289] | < 0.001 | – | True | no (sequence stopped earlier) | 50 | 5.890 |
| 4. top-1, MOMENT vs k-NN (McNemar) | 235 | -0.119 | [-0.2560, -0.0040] | [-0.1830, -0.0553] | < 0.001 | – | False | no (sequence stopped earlier) | 49 vs 77 | – |

## Family 2 (Q3), masked vs unmasked, fixed sequence at α = 0.05

Two-sided: significant = clustered interval excludes 0 (either direction) and p < 0.05.

| test | n | mean | CI clustered | CI plain | p | significant | confirmatory | hits |
|---|---|---|---|---|---|---|---|---|
| 1. VUS-PR, masked vs unmasked (difference of excesses) | 248 | 0.003 | [-0.0118, 0.0299] | [-0.0046, 0.0113] | 0.006 | False | no | – |
| 2. top-1, masked vs unmasked (McNemar) | 248 | -0.081 | [-0.1564, -0.0079] | [-0.1250, -0.0363] | < 0.001 | True | no (sequence stopped earlier) | 30 vs 50 |

## Per detector ("consistent")

Means over series; chance = mean over circular shifts (exact |Z|/n for top-1); share_p_le_005 = share of series beating their own null at p ≤ 0.05 (5% expected by chance); alerting at τ = max calibration score.

All series each detector ran on:

| detector | n | vus_pr | vus_pr_chance | vus_pr_excess | share_p_le_005 | top1 | top1_chance | adj_best_f1 | adj_best_f1_chance | adj_best_f1_excess | detection_rate | false_alarms_per_10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOMENT | 248 | 0.033 | 0.014 | 0.019 | 0.440 | 0.202 | 0.024 | 0.602 | 0.362 | 0.240 | 0.477 | 12.920 |
| MOMENT masked | 248 | 0.037 | 0.014 | 0.022 | 0.375 | 0.121 | 0.024 | 0.561 | 0.347 | 0.214 | 0.402 | 14.107 |
| Matrix profile | 248 | 0.334 | 0.028 | 0.305 | 0.573 | 0.375 | 0.024 | 0.413 | 0.097 | 0.316 | 0.698 | 7.860 |
| Matrix profile, self-join | 248 | 0.392 | 0.029 | 0.363 | 0.641 | 0.427 | 0.024 | 0.464 | 0.098 | 0.365 | 0.641 | 5.823 |
| Med-Mad | 248 | 0.024 | 0.016 | 0.009 | 0.121 | 0.085 | 0.024 | 0.337 | 0.286 | 0.051 | 0.355 | 13.963 |
| Mu-Sigma | 248 | 0.025 | 0.014 | 0.011 | 0.319 | 0.202 | 0.024 | 0.437 | 0.353 | 0.083 | 0.403 | 11.089 |
| Null validity (D4) | 248 | 0.013 | 0.014 | -0.001 | 0.036 | 0.020 | 0.024 | 0.343 | 0.357 | -0.014 | – | – |
| k-NN | 235 | 0.111 | 0.017 | 0.093 | 0.417 | 0.328 | 0.024 | 0.575 | 0.277 | 0.298 | 0.519 | 20.539 |

Only the 235 series where all 7 detectors ran:

| detector | n | vus_pr | vus_pr_chance | vus_pr_excess | share_p_le_005 | top1 | top1_chance | adj_best_f1 | adj_best_f1_chance | adj_best_f1_excess | detection_rate | false_alarms_per_10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOMENT | 235 | 0.034 | 0.014 | 0.020 | 0.447 | 0.209 | 0.024 | 0.611 | 0.367 | 0.244 | 0.485 | 12.913 |
| MOMENT masked | 235 | 0.038 | 0.015 | 0.023 | 0.387 | 0.128 | 0.024 | 0.574 | 0.351 | 0.223 | 0.404 | 14.174 |
| Matrix profile | 235 | 0.333 | 0.029 | 0.304 | 0.574 | 0.370 | 0.024 | 0.409 | 0.098 | 0.311 | 0.694 | 7.818 |
| Matrix profile, self-join | 235 | 0.386 | 0.029 | 0.357 | 0.626 | 0.421 | 0.024 | 0.453 | 0.099 | 0.354 | 0.626 | 5.900 |
| Med-Mad | 235 | 0.025 | 0.016 | 0.009 | 0.119 | 0.085 | 0.024 | 0.336 | 0.286 | 0.050 | 0.345 | 13.719 |
| Mu-Sigma | 235 | 0.024 | 0.014 | 0.010 | 0.311 | 0.191 | 0.024 | 0.435 | 0.356 | 0.079 | 0.383 | 10.786 |
| Null validity (D4) | 235 | 0.014 | 0.014 | -0.001 | 0.038 | 0.021 | 0.024 | 0.349 | 0.365 | -0.015 | – | – |
| k-NN | 235 | 0.111 | 0.017 | 0.093 | 0.417 | 0.328 | 0.024 | 0.575 | 0.277 | 0.298 | 0.519 | 20.539 |

Random baseline (D1, descriptive, all series): vus_pr 0.013, top1 0.027, adj_best_f1 0.363, detected 0.234, false_alarms_per_10k 10.980

## Null validity check (D4)

Share of series with p ≤ 0.05 (VUS-PR): 0.036 (n = 248); histogram in `figures/248_null_validity.png`.

## Reproduction ("paper-faithful")

**adj_best_f1**, MOMENT and k-NN (rows with a NaN dropped, as the authors' pipeline does; ranks and wins/losses between the two):

| detector | n | mean | median | std | Table 7 | mean rank | wins/losses |
|---|---|---|---|---|---|---|---|
| MOMENT | 238 | 0.586 | 0.678 | 0.378 | 0.585 | 1.481 | 108/99 |
| k-NN | 238 | 0.552 | 0.547 | 0.396 | 0.554 | 1.519 | 99/108 |


**adj_best_f1, not in the paper**: the course baselines and the random baseline (D1), same series; ranks and wins/losses among the three course baselines:

| detector | n | mean | median | std | mean rank | wins/losses |
|---|---|---|---|---|---|---|
| Mu-Sigma | 238 | 0.419 | 0.331 | 0.369 | 1.716 | 299/164 |
| Med-Mad | 238 | 0.333 | 0.230 | 0.318 | 2.145 | 197/266 |
| Matrix profile | 238 | 0.324 | 0.115 | 0.371 | 2.139 | 202/268 |
| Random (D1) | 238 | 0.376 | 0.357 | 0.268 | – | – |


**vus_roc**, MOMENT and k-NN (rows with a NaN dropped, as the authors' pipeline does; ranks and wins/losses between the two):

| detector | n | mean | median | std | Table 7 | mean rank | wins/losses |
|---|---|---|---|---|---|---|---|
| MOMENT | 238 | 0.671 | 0.678 | 0.134 | 0.670 | 1.588 | 98/140 |
| k-NN | 238 | 0.712 | 0.722 | 0.157 | 0.706 | 1.412 | 140/98 |


**vus_roc, not in the paper**: the course baselines and the random baseline (D1), same series; ranks and wins/losses among the three course baselines:

| detector | n | mean | median | std | mean rank | wins/losses |
|---|---|---|---|---|---|---|
| Mu-Sigma | 238 | 0.633 | 0.630 | 0.119 | 2.155 | 201/275 |
| Med-Mad | 238 | 0.636 | 0.618 | 0.117 | 2.206 | 189/287 |
| Matrix profile | 238 | 0.722 | 0.858 | 0.302 | 1.639 | 324/152 |
| Random (D1) | 238 | 0.595 | 0.603 | 0.031 | – | – |


Sensitivity, adjusted best F1 over all series with unscorable ones at 0: MOMENT 0.562 (n = 248), k-NN 0.530 (n = 248)

Series with nothing to score in the TSB-UAD test part (10, scored as the authors' code does): 72: anomaly entirely in the unscored tail; 79: no labelled point in the TSB-UAD file; 85: anomaly entirely in the unscored tail; 108: no labelled point in the TSB-UAD file; 180: anomaly entirely in the unscored tail; 187: no labelled point in the TSB-UAD file; 193: anomaly entirely in the unscored tail; 203: anomaly in the training part (pre-correction file name); 229: anomaly entirely in the unscored tail; 232: anomaly entirely in the unscored tail.

Series whose anomaly is partly in the unscored tail (3): 61 (44 of 60 labelled points or blocks scored), 105 (44 of 60 labelled points or blocks scored), 169 (44 of 60 labelled points or blocks scored).

k-NN's fit split (5/7) and test part downsampled differently under the per-part rule: 81 series.

tiltAPB2 (ID 198), MOMENT: adjusted best F1 0.960 (Table 22: 0.96), VUS-ROC 0.747 (Table 22: 0.75).

## Exploratory (unadjusted p-values; hypotheses, not findings)

| metric | n | mean excess | p (group Wilcoxon, unadjusted) |
|---|---|---|---|
| auc_pr | 248 | 0.031 | < 0.001 |
| auc_roc | 248 | 0.108 | < 0.001 |
| adj_best_f1_ranks | 248 | 0.162 | < 0.001 |

i.i.d. null vs circular shift, MOMENT, series where they disagree at p ≤ 0.05: {'vus_pr': 55, 'auc_pr': 34}

MOMENT's VUS-PR excess against anomaly length: Spearman rho 0.095, p 0.134; against series length: rho -0.200, p 0.002; DISTORTED vs not: Mann-Whitney p 0.996.

MOMENT's mean VUS-PR excess per source group (D5):

| group | count | mean |
|---|---|---|
| BIDMC | 3 | 0.036 |
| CHARIS | 8 | 0.002 |
| CIMISAirTemperature | 13 | 0.039 |
| ECG | 17 | 0.026 |
| EPG | 25 | 0.038 |
| Fantasia | 1 | 0.002 |
| GPMarkerLFMz | 11 | 0.009 |
| InternalBleeding | 26 | 0.030 |
| MesoplodonDensirostris | 3 | 0.009 |
| PowerDemand | 11 | 0.008 |
| STAFFIIIDatabase | 9 | 0.007 |
| TkeepMARS | 11 | 0.033 |
| WalkingAceleration | 4 | 0.028 |
| apneaecg | 9 | 0.001 |
| gait | 14 | 0.006 |
| ltstdbs | 6 | 0.077 |
| mitlongtermecg | 17 | 0.007 |
| parkm | 2 | 0.026 |
| qtdbSel | 4 | 0.006 |
| resperation | 17 | -0.000 |
| sddb | 9 | 0.014 |
| selmECG | 4 | 0.003 |
| sm | 4 | 0.040 |
| taichidbSMaster | 3 | 0.000 |
| tilt | 14 | 0.007 |
| weallwalk | 3 | 0.040 |
