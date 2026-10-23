Produced by `scripts/time_vus.py` at commit 9484a6781dfb2446d3727d4a9d10ac1110b18049 (with uncommitted changes).

Timing of `generate_curve` (one VUS call), random scores, 100-point anomaly:

| Test length | Window | Seconds |
|---|---|---|
| 2000 | 16 | 0.01 |
| 2000 | 128 | 0.11 |
| 2000 | 512 | 0.50 |
| 5000 | 16 | 0.02 |
| 5000 | 128 | 0.14 |
| 5000 | 512 | 0.59 |
| 10000 | 16 | 0.02 |
| 10000 | 128 | 0.16 |
| 10000 | 512 | 0.73 |

Fitted model: t = 1.25e-04 · n^0.26 · w^1.00 seconds.

Test lengths of the 248 series under "consistent": median 7000, max 70764, total 2616615.

Window L of the 248 series (`results/periods.csv`): median 26, quartiles 16–78; at the lower bound 16: 74; at the upper bound 512: 6, of which no peak found: 0.

Projected VUS time for the whole study (consistent), with each series' own L:

One VUS pass over the 248 series: 22 s.

- 100 shifts for all 7 detectors: 4.4 h
- 100 shifts for the 3 confirmatory detectors, 20 for the other 4: 2.5 h
- 50 shifts for all 7 detectors: 2.3 h
- 20 shifts for all 7 detectors: 1.0 h

Calls per series: 7 detectors × (1 + shifts) + 10 random draws; one process, no parallelism.
