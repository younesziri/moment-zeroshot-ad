Produced by `scripts/check_labels.py` at commit b3e112fa4e4ef42a2478d7cf9aed16728b3ab144 (with uncommitted changes).

TSB-UAD KDD21 files: 250; in the study (IDs 1–250 except 32 and 140): 248.

| Check (248 series) | Count |
|---|---|
| label column = file name, archive convention (1-based inclusive) | 0 |
| label column = file name, 0-based [begin, end) | 248 |
| matches neither | 0 |
| no labelled point | 3 |
| more than one labelled segment | 0 |
| missing values | 0 |
| not exactly 2 columns | 0 |
| file name differs from the original archive's | 1 |
| labelled anomaly starts inside the training part | 1 |
| values identical to the archive's | 243 |

One-point anomalies (begin = end in the file name):

| ID | File | Labelled points | First | Last |
|---|---|---|---|---|
| 79 | 079_UCR_Anomaly_DISTORTEDresperation2_30000_168250_168250.out | 0 | nan | nan |
| 108 | 108_UCR_Anomaly_NOISEresperation2_30000_168250_168250.out | 0 | nan | nan |
| 187 | 187_UCR_Anomaly_resperation2_30000_168250_168250.out | 0 | nan | nan |

Series whose values differ from the archive's:

| ID | File | Difference |
|---|---|---|
| 233 | 233_UCR_Anomaly_mit14157longtermecg_18913_24500_24501.out | 3 values differ, max difference 3.85e-34 |
| 234 | 234_UCR_Anomaly_mit14157longtermecg_18913_24600_24601.out | 5 values differ, max difference 3.85e-34 |
| 236 | 236_UCR_Anomaly_mit14157longtermecg_19313_46350_46390.out | 1 values differ, max difference 3.85e-34 |
| 237 | 237_UCR_Anomaly_mit14157longtermecg_19313_89560_90370.out | 6 values differ, max difference 3.85e-34 |
| 238 | 238_UCR_Anomaly_mit14157longtermecg_21311_72600_72780.out | 5 values differ, max difference 3.85e-34 |

Series matching neither convention, named differently from the archive, or with the anomaly
inside the training part:

| ID | TSB-UAD file | Archive file | Labelled [first, last] | Training end | Anomaly in training part |
|---|---|---|---|---|---|
| 203 | 203_UCR_Anomaly_CHARISfive_11812_10995_11028.out | 203_UCR_Anomaly_CHARISfive_10500_10995_11028 | [10995, 11027] | 11812 | yes |
