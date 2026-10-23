Produced by `scripts/analyse_248.py` at commit 9484a6781dfb2446d3727d4a9d10ac1110b18049 (with uncommitted changes).

## Coverage

- Series with results: 248; per-part downsampling rule differs from ours: 112; calibration part shorter than 512: 121; no autocorrelation peak (L = 512): 0.
- Failures (31):

| id | configuration | detector | error |
|---|---|---|---|
| 15 | consistent | MOMENT | ValueError: not enough history (429 points) to score a part of 72 |
| 15 | consistent | MOMENT masked | ValueError: not enough history (429 points) to score a part of 72 |
| 15 | consistent | k-NN | ValueError: not enough history (429 points) to score a part of 72 |
| 56 | consistent | MOMENT | ValueError: not enough history (429 points) to score a part of 72 |
| 56 | consistent | MOMENT masked | ValueError: not enough history (429 points) to score a part of 72 |
| 56 | consistent | k-NN | ValueError: not enough history (429 points) to score a part of 72 |
| 57 | consistent | k-NN | ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 5, n_samples_fit = 4, n_samples = 1 |
| 74 | consistent | MOMENT | ValueError: not enough history (343 points) to score a part of 58 |
| 74 | consistent | MOMENT masked | ValueError: not enough history (343 points) to score a part of 58 |
| 74 | consistent | k-NN | ValueError: not enough history (343 points) to score a part of 58 |
| 75 | consistent | MOMENT | ValueError: not enough history (343 points) to score a part of 58 |
| 75 | consistent | MOMENT masked | ValueError: not enough history (343 points) to score a part of 58 |
| 75 | consistent | k-NN | ValueError: not enough history (343 points) to score a part of 58 |
| 104 | consistent | k-NN | ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 5, n_samples_fit = 4, n_samples = 1 |
| 123 | consistent | MOMENT | ValueError: not enough history (429 points) to score a part of 72 |
| 123 | consistent | MOMENT masked | ValueError: not enough history (429 points) to score a part of 72 |
| 123 | consistent | k-NN | ValueError: not enough history (429 points) to score a part of 72 |
| 164 | consistent | MOMENT | ValueError: not enough history (429 points) to score a part of 72 |
| 164 | consistent | MOMENT masked | ValueError: not enough history (429 points) to score a part of 72 |
| 164 | consistent | k-NN | ValueError: not enough history (429 points) to score a part of 72 |
| 165 | consistent | k-NN | ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 5, n_samples_fit = 4, n_samples = 1 |
| 182 | consistent | MOMENT | ValueError: not enough history (343 points) to score a part of 58 |
| 182 | consistent | MOMENT masked | ValueError: not enough history (343 points) to score a part of 58 |
| 182 | consistent | k-NN | ValueError: not enough history (343 points) to score a part of 58 |
| 183 | consistent | MOMENT | ValueError: not enough history (343 points) to score a part of 58 |
| 183 | consistent | MOMENT masked | ValueError: not enough history (343 points) to score a part of 58 |
| 183 | consistent | k-NN | ValueError: not enough history (343 points) to score a part of 58 |
| 207 | consistent | MOMENT | ValueError: not enough history (272 points) to score a part of 46 |
| 207 | consistent | MOMENT masked | ValueError: not enough history (272 points) to score a part of 46 |
| 207 | consistent | k-NN | ValueError: not enough history (272 points) to score a part of 46 |
| 208 | consistent | k-NN | ValueError: window shape cannot be larger than input array shape |

## Family 1 (Q1–Q2), fixed sequence at α = 0.05

Significant = clustered bootstrap 95% interval excludes 0 in MOMENT's favour and p < 0.05.

| test | n | mean | CI clustered | CI plain | p | significant | confirmatory | hits | expected |
|---|---|---|---|---|---|---|---|---|---|
| 1. VUS-PR, MOMENT vs chance | 239 | 0.019 | [0.0121, 0.0268] | [0.0149, 0.0244] | 0.000 | True | yes | – | – |
| 2. VUS-PR, MOMENT vs k-NN (difference of excesses) | 235 | -0.074 | [-0.1420, -0.0137] | [-0.0961, -0.0529] | 0.096 | False | no | – | – |
| 3. top-1, MOMENT vs chance (|Z|/n) | 239 | 0.181 | [0.0847, 0.2961] | [0.1316, 0.2315] | 0.000 | True | no (sequence stopped earlier) | 49 | 5.770 |
| 4. top-1, MOMENT vs k-NN (McNemar) | 235 | -0.119 | [-0.2560, -0.0040] | [-0.1830, -0.0553] | 0.000 | False | no (sequence stopped earlier) | 49 vs 77 | – |

## Family 2 (Q3), masked vs unmasked, fixed sequence at α = 0.05

Two-sided: significant = clustered interval excludes 0 (either direction) and p < 0.05.

| test | n | mean | CI clustered | CI plain | p | significant | confirmatory | hits |
|---|---|---|---|---|---|---|---|---|
| 1. VUS-PR, masked vs unmasked (difference of excesses) | 239 | 0.004 | [-0.0119, 0.0320] | [-0.0042, 0.0125] | 0.011 | False | no | – |
| 2. top-1, masked vs unmasked (McNemar) | 239 | -0.079 | [-0.1539, -0.0050] | [-0.1255, -0.0377] | 0.001 | True | no (sequence stopped earlier) | 30 vs 49 |

## Per detector ("consistent")

Means over series; chance = mean over circular shifts (exact |Z|/n for top-1); share_p_le_005 = share of series beating their own null at p ≤ 0.05 (5% expected by chance); alerting at τ = max calibration score.

| detector | n | vus_pr | vus_pr_chance | vus_pr_excess | share_p_le_005 | top1 | top1_chance | adj_best_f1 | adj_best_f1_chance | detection_rate | false_alarms_per_10k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOMENT | 239 | 0.034 | 0.014 | 0.019 | 0.439 | 0.205 | 0.024 | 0.603 | 0.363 | 0.477 | 12.920 |
| MOMENT masked | 239 | 0.038 | 0.015 | 0.023 | 0.381 | 0.126 | 0.024 | 0.565 | 0.347 | 0.402 | 14.107 |
| Matrix profile | 248 | 0.334 | 0.028 | 0.305 | 0.573 | 0.375 | 0.024 | 0.413 | 0.097 | 0.698 | 7.860 |
| Matrix profile, self-join | 248 | 0.392 | 0.029 | 0.363 | 0.641 | 0.427 | 0.024 | 0.464 | 0.098 | 0.641 | 5.823 |
| Med-Mad | 248 | 0.024 | 0.016 | 0.009 | 0.121 | 0.085 | 0.024 | 0.337 | 0.286 | 0.355 | 13.963 |
| Mu-Sigma | 248 | 0.025 | 0.014 | 0.011 | 0.319 | 0.202 | 0.024 | 0.437 | 0.353 | 0.403 | 11.089 |
| Null validity (D4) | 248 | 0.013 | 0.014 | -0.001 | 0.036 | 0.020 | 0.024 | 0.343 | 0.357 | – | – |
| k-NN | 235 | 0.111 | 0.017 | 0.093 | 0.417 | 0.328 | 0.024 | 0.575 | 0.277 | 0.519 | 20.539 |

Random baseline (D1, descriptive): vus_pr 0.013, top1 0.027, adj_best_f1 0.363, detected 0.234, false_alarms_per_10k 10.980

## Null validity check (D4)

Share of series with p ≤ 0.05 (VUS-PR): 0.036 (n = 248); histogram in `figures/248_null_validity.png`.

## Reproduction ("paper-faithful")

**adj_best_f1** (rows with a NaN dropped, as the authors' pipeline does):

| detector | n | mean | median | std | Table 7 | mean rank | wins/losses |
|---|---|---|---|---|---|---|---|
| MOMENT | 238 | 0.586 | 0.678 | 0.378 | 0.585 | 2.214 | 641/267 |
| k-NN | 238 | 0.552 | 0.547 | 0.396 | 0.554 | 2.416 | 588/310 |
| Mu-Sigma | 238 | 0.419 | 0.331 | 0.369 | – | 3.006 | 458/461 |
| Med-Mad | 238 | 0.333 | 0.230 | 0.318 | – | 3.723 | 294/638 |
| Matrix profile | 238 | 0.324 | 0.115 | 0.371 | – | 3.641 | 316/621 |
| Random (D1) | 238 | 0.376 | 0.357 | 0.268 | – | – | – |


**vus_roc** (rows with a NaN dropped, as the authors' pipeline does):

| detector | n | mean | median | std | Table 7 | mean rank | wins/losses |
|---|---|---|---|---|---|---|---|
| MOMENT | 238 | 0.671 | 0.678 | 0.134 | 0.670 | 2.878 | 505/447 |
| k-NN | 238 | 0.712 | 0.722 | 0.157 | 0.706 | 2.613 | 568/384 |
| Mu-Sigma | 238 | 0.633 | 0.630 | 0.119 | – | 3.538 | 348/604 |
| Med-Mad | 238 | 0.636 | 0.618 | 0.117 | – | 3.613 | 330/622 |
| Matrix profile | 238 | 0.722 | 0.858 | 0.302 | – | 2.357 | 629/323 |
| Random (D1) | 238 | 0.595 | 0.603 | 0.031 | – | – | – |


Sensitivity, adjusted best F1 over all series with unscorable ones at 0: MOMENT 0.562 (n = 248), k-NN 0.530 (n = 248)

Series with no labelled anomaly in the TSB-UAD test part (scored as the authors' code does): [72, 79, 85, 108, 180, 187, 193, 203, 229, 232]

tiltAPB2 (ID 198), MOMENT: adjusted best F1 0.960 (Table 22: 0.96), VUS-ROC 0.747 (Table 22: 0.75).

## Exploratory (unadjusted p-values; hypotheses, not findings)

| metric | n | mean excess | p (group Wilcoxon, unadjusted) |
|---|---|---|---|
| auc_pr | 239 | 0.032 | 0.000 |
| auc_roc | 239 | 0.108 | 0.000 |
| adj_best_f1_ranks | 239 | 0.165 | 0.000 |

i.i.d. null vs circular shift, MOMENT, series where they disagree at p ≤ 0.05: {'vus_pr': 54, 'auc_pr': 34}

MOMENT's VUS-PR excess against series properties: anomaly length (Spearman rho, p): 0.080, 0.219; series length (Spearman rho, p): -0.213, 0.001; DISTORTED vs not (Mann-Whitney p): 0.970

MOMENT's mean VUS-PR excess per source group (D5):

| group | count | mean |
|---|---|---|
| BIDMC | 3 | 0.036 |
| CHARIS | 7 | 0.003 |
| CIMISAirTemperature | 13 | 0.039 |
| ECG | 15 | 0.020 |
| EPG | 25 | 0.038 |
| Fantasia | 1 | 0.002 |
| GPMarkerLFMz | 11 | 0.009 |
| InternalBleeding | 26 | 0.030 |
| MesoplodonDensirostris | 3 | 0.009 |
| PowerDemand | 11 | 0.008 |
| STAFFIIIDatabase | 9 | 0.007 |
| TkeepMARS | 11 | 0.033 |
| WalkingAceleration | 4 | 0.028 |
| apneaecg | 7 | 0.002 |
| gait | 14 | 0.006 |
| ltstdbs | 6 | 0.077 |
| mitlongtermecg | 17 | 0.007 |
| parkm | 2 | 0.026 |
| resperation | 17 | -0.000 |
| sddb | 9 | 0.014 |
| selmECG | 4 | 0.003 |
| sm | 4 | 0.040 |
| taichidbSMaster | 3 | 0.000 |
| tilt | 14 | 0.007 |
| weallwalk | 3 | 0.040 |
