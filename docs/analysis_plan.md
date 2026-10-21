# Analysis plan: the 248-series study

Planned 19/10/2026   before any run on the 248 series.

The study answers three questions, in this order:

- **Q1.** Is MOMENT-1-large, zero-shot, better than chance on the UCR archive, when chance keeps
  each detector's own scores?
- **Q2.** Is it better than k-NN, the paper's baseline?
- **Q3.** Does masked scoring change MOMENT's results?

It also checks the paper's numbers (Table 7) under the paper's protocol. The case study on one series
(`results/tiltAPB2_x1.md`, `results/tiltAPB2_x10.md`) motivated these choices.


## 1. Series and data

- **Series**: UCR Anomaly Archive IDs 1–250 except 32 and 140 (the paper doesn't say why they are excluded.), so 248 series, as in the 
  authors' code.
  *Why: same series as Table 7, so the reproduction is comparable.*
- **Primary data: TSB-UAD-Public, folder KDD21** (the authors' exact files, `.out`: value, label),
  about 241 MB, downloaded into `data/TSB-UAD-Public/` (gitignored) by a script that records the
  archive's SHA-256. The training end comes from the file name; the anomaly depends on the
  configuration (see Labels below).
  *Why: the reproduction must use the same data as Table 7.*
- **Files**: the original archive, `UCR_TimeSeriesAnomalyDatasets2021.zip` from cs.ucr.edu 
  (SHA-256 `ac4b991c701e620ae9cc5ebd57ae45593a36cc9c0b6ed5e3c4b7e466cf4783d4`, in `data/ucr/`), used
  for the sanity checks below.

- **Anomaly and training ranges, from the file name** `NNN_UCR_Anomaly_<name>_<X>_<begin>_<end>.txt`.
  The archive's own slides use 1-based, inclusive MATLAB indexing: points 1..X are training data and
  the anomaly is `T(begin:end)`. In 0-based indices: train = [0, X), anomaly = [begin − 1, end − 1],
  length end − begin + 1. Three series have begin = end (IDs 79, 108, 187): one-point anomalies,
  which confirms the inclusive reading.
- **Labels**: TSB-UAD's label column appears to use 0-based [begin, end) (tiltAPB2: [124159, 124984],
  826 points), whereas the file name read 1-based inclusive gives [124158, 124984] (827 points).
  - Checked on every series before any run, especially the one-point anomalies (IDs 79, 108, 187),
    where [begin, end) would be empty; if TSB-UAD's labels are empty there, this is reported, and
    those series cannot be scored under "paper-faithful" (listed as failed there).
  - Rule: "consistent" uses the archive's definition (file name, 1-based inclusive); "paper-faithful"
    uses TSB-UAD's labels as they are (shifted by the dropped first point, as the research code reads
    them). Every mismatch between the two is listed.
  - *Why: the archive's authors define the anomaly; the paper's numbers were computed on TSB-UAD's
    labels.*


- **Known archive fix**: the archive's slides say IDs 202 and 203 were mislabeled and
  renamed. My copy has the corrected names .
  If TSB-UAD has the old names, the paper's results on these two series used wrong labels, I'll use them for the faithfull run 

### Source groups 

Series built from the same source are not independent, so the confidence intervals are also
computed by resampling groups (section 6). Rule: take the name stem, strip "DISTORTED", "NOISE" and
all digits, then merge by eye the stems that are variants of the same data according to the
archive's slides, and the five candidate merges (gait + gaitHunt, insectEPG + LabCmacEPG,
CHARISfive + CHARISten, tiltAPB + tiltmtable, PowerDemand + Italianpowerdemand). 


## 2. Model and detectors

- **MOMENT-1-large**, zero-shot, `model.eval()`, positional embedding on, nothing masked, score =
  (x − x̂)², windows of 512 (`src/zsad/moment.py`, `src/zsad/data.py::windows`). Hugging Face
  revision `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc`.
  *Why: the only released checkpoint that reconstructs (`results/moment_diagnostics.md`).*
- **Main comparator: k-NN** as in the authors' code: 512-point windows, stride 1, on the fit part;
  5 neighbours; reconstruction = mean of the neighbours; score = squared error.
  *Why: the paper's own baseline, close to MOMENT on Table 7.*
- **Secondary detectors**, reported but not in the confirmatory tests: the course baselines
  (Lecture 5, pp. 64 and 73): Where does MOMENT sit among simple methods?.
  - **One length L for all three**: the dominant period of the train part, after the series'
    downsampling. L = the lag of the first local maximum of the train part's autocorrelation that
    follows its first negative value, clipped to [16, 512]. If the autocorrelation never becomes
    negative or has no later maximum, L = 512 and the series is listed. *Why: one rule, adapted to
    each series' own rhythm, with nothing tuned on the labels.*
  - **Mu-Sigma**: s[n] = |x[n] − μₙ| / σₙ, with μₙ and σₙ the mean and standard deviation of the L
    points before n (trailing window, x[n] excluded). The first test points use the end of the
    train part as history.
  - **Med-Mad**: s[n] = |x[n] − medₙ| / MADₙ, same trailing window, MAD unscaled. A zero σₙ or MADₙ
    (constant window, e.g. quantised data) is replaced by 10⁻⁸ × the train part's standard deviation.
  - **Matrix profile** (Yeh et al. 2016), with `stumpy` : AB-join of every
    test subsequence of length L against the fit part, z-normalised Euclidean distance to its nearest
    fit subsequence; a point's score is the largest value among the test subsequences that contain
    it. A self-join (as in the course's definition) is exploratory (section 7).

  The archive was designed so that anomalies are not, in general, extreme values, so the two point
  detectors are expected to be weak; they are sanity baselines.  
- **Pre-specified comparison: masked vs unmasked MOMENT.** Four passes per window; pass j (0–3)
  hides patches j, j + 4, j + 8, … (`mask` = 0 on their time steps, `input_mask` all ones). Each point
  is reconstructed by the pass that hid its patch: score = (x − x̂_masked)². Same windows, metrics,
  null and tests as unmasked. *Why: pretraining computes the loss on masked patches only (paper,
  §3.3), so predicting hidden patches is the task the model was trained for.*

## 3. Two configurations

- **"Consistent" (main analysis).**
  - One downsampling factor per series, decided on the test part with the paper's rule (×10
    max-pooling if len // 10 > 2560), then applied to every part.
  - Scaling with the whole train part's mean and std, as in the tutorial (`prepare`).   
  - fit = first 6/7 of the train part, calibration = the rest; windows as in the tutorial (the last
    one aligned to the end).
  - *Why: every part of a series at the same resolution, so k-NN compares like with like and the
    alert threshold τ applies to scores of the same kind.*
- **"Paper-faithful" (reproduction and sensitivity).** Replicates the research code
  (`moment-research`, commit 3ab637e, `_read_and_process_KDD21`), each rule documented:
  - the file is read without `header=None`, so the first point is dropped, and the training end
    from the file name is applied to the shortened series;
  - scaler fit on each config's own train split, applied to the whole series, then downsampling:
    the first 6/7 of the train part for MOMENT (zero-shot config), the first 5/7 for k-NN (its config);
  - per-part downsampling decision (×10 if len // 10 > 2560, each part on its own);
  - windows of 512 with stride 512 and `(len − 512) // 512 + 1` of them: the tail shorter than 512
    points is not scored;
  - k-NN fit = first 5/7 of the train part (its own config);
  - calibration = the rest of the train part after each detector's fit part (6/7 for MOMENT, 5/7 for
    k-NN); alerting is not in the paper, so this part is ours.
  - *Why: the closest we can get to how Table 7 was computed, with public weights.*
- **MOMENT's scores**: MOMENT runs in both configurations; the scores differ (scaler, first point,
  unscored tail) and the difference is reported.
- **Report** how many series get a different downsampling decision in the two configurations.
- **Short parts.** From the file names, 50 of the 248 series have a calibration part shorter than
  512 points at full resolution (no fit part is that short). The paper's anomaly-detection code does
  not pad: its dataset returns nothing for a series shorter than one window. Rule for parts shorter
  than one window:
  - "consistent": the part is scored with the preceding anomaly-free points as left context (the
    last 512 points before its end are scored, and only the part's own scores are kept). Caveat: for
    k-NN those context points belong to its training data, which may bias its calibration scores
    down; reported.
  - "paper-faithful": as the research code, no padding: a part shorter than one window gives no
    window and is not scored (`anomaly_detection_datasets.py`, `__getitem__` returns `None`; its
    padding options are only used by the classification and forecasting datasets). The paper's text
    mentions left zero-padding of short inputs (§3.1), but its anomaly-detection code does not do it.

  The number of series concerned is reported for each configuration. A series whose test part is
  shorter than 512 points after processing is excluded, with the reason listed.

## 4. Reproduction ("paper-faithful")

- **Anchors** (Table 7, means over 248 series): adjusted best F1, MOMENT₀ 0.585 and k-NN 0.554;
  VUS-ROC, MOMENT₀ 0.670 and k-NN 0.706. tiltAPB2 against Table 22's 0.96 (adjusted best F1).
- **Summaries in Table 7's format**: mean, median, standard deviation, mean rank, wins/losses. Mean,
  median and standard deviation compare directly with Table 7; ranks and wins/losses depend on which
  methods are ranked (seven in the paper, our five detectors here), so they are reported but not
  compared with the paper's.

- Table 7 reports no null and no significance test: that is what this study adds.

## 5. Metrics ("consistent")

All from `src/zsad/metrics.py`, computed on the test part at the configuration's resolution.

- **Co-primary**:
  - **VUS-PR** (volume under the range-based precision–recall surface; the VUS library, Paparrizos
    et al. 2022), with its sliding-window parameter set to L (section 2); the library version is
    recorded. *Why: threshold-free and range-aware, and the course recommends precision–recall
    curves, in their range-based version, when events are rare (Lecture 5, pp. 90 and 96).*
    - **Dependency check** (21/10/2026, dry-run resolution with `uv pip compile`, Python 3.11,
      macOS 15 arm64; nothing installed): `vus` 0.0.6 (latest on PyPI) resolves together with the
      project's pins (numpy 1.25.2, torch 2.14.1, transformers 4.33.3, momentfm 0.1.4), but brings 84
      packages in total, including tensorflow 2.17.1, tsfresh, arch and tslearn. Importing it and
      computing VUS-PR are untested. Not added until the author approves.
  - **UCR top-1 accuracy**: the share of series whose highest score lies within ±100 original points
    of the anomaly (±10 points after ×10 max-pooling). *Why: the archive's own scoring, one anomaly
    per series, with a tolerance margin (course, p. 89).*
- **Reproduction only**: adjusted best F1, also reported as its excess over its null, to quantify how
  much chance it credits (course, p. 91).
- **Secondary**, at τ = max score on the calibration part (no labels; course, p. 91):
  - **event detection rate**: the share of series with an alert within ±100 original points of the
    anomaly;
  - **false alarms per 10,000 points**: false-alarm events (alert runs entirely outside the tolerance
    range) per 10,000 original test points, so that full-resolution and downsampled series compare.
- **Every other metric** (point-wise best F1, AUC-PR, AUC-ROC, adjusted best F1 on rank scores,
  precision, recall, delay) goes into the per-series CSV and into no claim (section 7).
- **Implementation requirement**: a vectorised adjusted best F1. With exactly one contiguous anomaly
  of length L, at threshold t: TP = L if the anomaly's maximum score is ≥ t, else 0; FP = number of
  normal points with score ≥ t; F1 with momentfm's epsilons. Tested equal (`==`) to momentfm's
  `adjbestf1` on many random cases, including one-point anomalies and ties.

## 6. Null and tests (all under "consistent")

- **Null** (the course's comparison with a random detector, p. 89): each detector's own test scores,
  circularly shifted, labels fixed.
  - Valid offsets: o in [1, n − 1] with circular distance min(o, n − o) > L + W, where L is the
    anomaly length and W = 512 points at the series' resolution.
  - 100 offsets drawn without replacement from the valid ones, seed = series ID. If fewer than 100
    are valid, all valid offsets are used; if none, the series is excluded from the null-based tests
    and listed.
  - Chance level = mean metric over the shifts; per-series p = (1 + #shifts at least as good) /
    (1 + #shifts), i.e. / 101 with 100 shifts.
  - *Why: keeps the score distribution and its clustering; only the alignment with the anomaly is
    broken (the tiltAPB2 case study showed iid noise is too harsh or too lenient depending on the
    metric).*
- **Main statistic**: perseries difference (detector − it's chance level). Across series:
  - **effect size** = excess = metric − chance level;
  - every comparison between detectors uses their excesses over their own chance levels.
- **VUS-PR**:
  - the mean excess with a 95% bootstrap confidence interval (10,000 resamples), clustered by
    resampling the source groups of section 1; the plain version (resampling series) also reported;
  - a two-sided Wilcoxon signedrank test on group level means: the excesses (or differences of
    excesses) are averaged within each source group, then tested across the 26 groups (zero
    differences dropped).
- **Top-1**:
  - vs chance: observed hits against expected hits, the sum over series of each series' hit rate
    under its shifts, with an exact Poisson-binomial test;
  - MOMENT vs k-NN: an exact McNemar test on the paired hits;
  - a clustered bootstrap confidence interval of the accuracy difference (detector − chance, or
    MOMENT − k-NN).
- **Significant** means both: the clustered bootstrap 95% interval excludes 0 in the detector's
  favour, and the paired test above gives p < 0.05. *Why: related series would otherwise count as
  independent evidence.*       

- **MOMENT vs k-NN**: per series difference of each detector's excess over its own chance level,
  for every metric. *Why: one rule; under circular shifts the two chance levels are only roughly
  equal (on tiltAPB2 at full resolution, shifted AUC-PR averaged 0.011 for MOMENT, 0.013 for k-NN).*
- **Fixed-sequence testing at α = 0.05**, all under "consistent":
  1. VUS-PR, MOMENT vs chance;
  2. VUS-PR, MOMENT vs k-NN;
  3. top-1, MOMENT vs chance
  4. top-1, MOMENT vs k-NN.

  A test counts as confirmatory only if all earlier ones were significant. Every result is reported
  either way.
- **Masked vs unmasked**, its own fixed sequence at α = 0.05:
  1. VUS-PR, masked vs unmasked;
  2. top-1, masked vs unmasked;

  with the same primaries and tests, compared, like every pair of detectors, by their excess over
  their own chance levels.
- **Test families**: Q1–Q2 (the four tests above) and Q3 (masked vs unmasked) are two separate
  families, each controlled at α = 0.05 by its own fixed sequence; the error rate is not controlled
  across the two families together.
  - Clustered intervals with 26 groups (one of them a single series) are approximate: with few
    clusters, bootstrap intervals can be too narrow.
  - The bootstrap interval and the group-level Wilcoxon test estimate slightly different things: the
    interval is for the mean excess over series (each series weighs the same, resampled by group);
    the Wilcoxon test is about the group means (each group weighs the same). Requiring both is
    deliberate.

- **Secondary**: for each detector, the fraction of series where it beats its own null at
  p ≤ 0.05, against the 5% expected by chance 
- **Null validity check**: the same test on a detector that cannot know where the anomaly is:
  MOMENT's scores taken from a different series, with a fixed pairing. Its per-series p-values should
  be roughly uniform; the fraction ≤ 0.05 and a histogram are reported.
- **Number of shifts for VUS-PR**: VUS is much slower than the other metrics. Before fixing it, VUS-PR
  is timed on tiltAPB2 and the projected total runtime is reported; 100 shifts is the default.

## 7. Exploratorry analyses


- AUC-PR
- AUC-ROC
- Adjusted best F1 on rank scores 
- The iid random null alongside the circular shift: the number of series where they disagree at
  p ≤ 0.05.\
- MOMENT's excess on VUS-PR against anomaly length, series length, domain (from the file name) and
  Distorted vs not
- Full resolution on 50 series chosen with a fixed seed.
- Matrix profile with a self-join 



## Deviations from this plan

### D1, 21/10/2026, before any run: a standard random baseline (descriptive)

- **What**: i.i.d. uniform random scores in [0, 1), one per point, on the calibration and test
  parts; 10 draws per series (seed = 1000 × series ID + draw number). Scored with every metric of
  sections 4 and 5, in both configurations, at each configuration's resolution; τ = max of its own
  calibration scores, as for the detectors.
- **Reported**: as a "Random" row next to the detectors in every summary, including the Table 7
  format under "paper-faithful": per series, the mean over the 10 draws; across series, mean,
  median and standard deviation.
- **Not used**: in any test, in ranks or wins/losses, or as a null. The circular shift of each
  detector's own scores (section 6) stays the reference for "better than chance". It is also distinct
  from the exploratory i.i.d. null of section 7, which is a per-series test.
- **Caveat**: on shape-dependent metrics (adjusted best F1), the value depends on the distribution of
  the random scores (the tiltAPB2 case study); uniform is the convention, not a neutral choice.
- *Why: the usual reference in the literature (e.g. Kim et al. 2022): what a detector with no
  information scores on each metric, which puts numbers like Table 7's 0.585 in context.*

### D2, 21/10/2026, before any run: exact top-1 chance level

- **What**: for top-1, the chance level of a series is the exact value |Z|/n, where Z is the set of
  test positions within the UCR tolerance of the anomaly and n the test length (at the
  configuration's resolution). It is the same for every detector. The Poisson-binomial test uses
  these exact pᵢ; MOMENT vs k-NN on top-1 uses an exact McNemar test on the raw hits; no Monte Carlo
  for top-1. Replaces the shift-based top-1 chance level of section 6.
- *Why: over all n rotations of a score series, its highest point lands on every position once, so
  the probability of a hit is exactly |Z|/n, whatever the detector (assuming a unique maximum; ties
  are resolved by the first index, as in `np.argmax`). This also removes the 0-hit problem of a
  Monte Carlo estimate, and since the chance level is shared, raw hits and excesses give the same
  comparison.*

### D3, 21/10/2026,  the circular-shift null uses all non-zero offsets

- **What**: the offsets are drawn from all of 1, …, n − 1 (the standard rotation test): 100 drawn
  uniformly without replacement, seed = series ID. Replaces the exclusion of offsets within the
  anomaly length plus one window (section 6).
- *Why: excluding small offsets removes exactly the shifts that keep a hitting detector's high scores
  near the anomaly, so the chance level then depends on whether the detector hit, and per-series
  p-values can become anti-conservative. The rotation test is valid only over the full set of
  rotations.*

### D4, 21/10/2026 pairing for the null validity check

- **What**: each series is paired with one series from a different source group (a fixed pairing,
  seeded). Its scores for the target series are a contiguous segment of the paired series' MOMENT
  test scores, of the target's test length, starting at the beginning of the paired test part and
  wrapping around if it is shorter.
- *Why: the scores then come from a real detector, with realistic shape and clustering, but cannot
  know where the target's anomaly is.*

### D5, 21/10/2026,: "domain" in the exploratory list

- **What**: "domain (from the file name)" in section 7 is replaced by the 26 source groups of
  section 1, with descriptive per-group means only (no test).
- *Why: the file names give a source, not a domain; a hand-made domain mapping would be a choice
  made without a rule.*

### D6, 22/10/2026: tiltAPB2 case study recomputed

- **What**: the tiltAPB2 case study's circular-shift p-values are recomputed with all non-zero
  offsets (D3), and the README's numbers updated. The case study keeps 1000 shifts, drawn without
  replacement, seed = 198 (its series ID).
- *Why: the case study used the offset exclusion that D3 removes.*

### D7, 22/10/2026: one data source per configuration

- "Consistent" uses the UCR archive for everything (values; anomaly and training end from the file
  name, 1-based inclusive); "paper-faithful" uses TSB-UAD for everything. Replaces the label rule of
  section 1.
- Checked (`results/label_check.md`): values identical in 243 series; in IDs 233, 234, 236–238 a few
  values differ, maximum difference 3.85e-34. TSB-UAD labels follow 0-based [begin, end); IDs 79,
  108, 187 are unlabelled; ID 203 keeps the pre-correction training end (11812), which puts its
  anomaly in the training part.

### D8, 22/10/2026: series without a scorable anomaly (IDs 79, 108, 187, 203)

- "Paper-faithful": scored as the authors' code scores them (adjusted best F1 = 0, VUS = NaN, row
  dropped by their `.dropna()`); IDs listed.
- Means under "paper-faithful": (1) as their pipeline computes them, NaN rows dropped, n printed: the
  Table 7 comparison; (2) adjusted best F1 only, sensitivity line: mean over all 248 with these
  series at 0.
- "Consistent": all four scored with the archive's labels.
- ID 202: same name in both sources, values identical, scored normally.

### D9, 22/10/2026: VUS implementation and window

- VUS from the authors' own code, copied unchanged with its licence into
  `src/zsad/vus_moment_research.py` (`generate_curve`; adapted by them from TheDatumOrg/VUS); no
  `vus` package.
- Window: the reproduction anchors use theirs 
