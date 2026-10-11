# MOMENT zero-shot anomaly detection

Mini-project for *Apprentissage pour les séries temporelles* (MVA): zero-shot anomaly
detection with MOMENT (Goswami et al., ICML 2024), evaluated with and without point-adjust.

## Question
Does the zero-shot anomaly-detection advantage reported for MOMENT (Goswami et al., ICML 2024)
hold under an evaluation calibrated against chance? The paper's headline metric (adjusted best F1,
with point-adjust and a threshold chosen with the labels) can reward luck, so this project asks
which conclusions survive stricter evaluation.

## Approach
1. **Reproduce** the paper's zero-shot results with the released weights.
2. **Re-evaluate the same scores**: random-score baseline, point-wise and range-aware metrics,
   a threshold fixed without labels, simple baselines (rolling z-score, k-NN), and paired
   statistics across series.
3. **Extend**: masked scoring at inference (planned).

## Status
Work in progress, MVA course project (due Dec 2026).
## Setup

```bash
uv sync                                   # Python 3.11 + pinned dependencies from uv.lock
uv run python scripts/check_setup.py      # feasibility check (downloads ~2 GB of weights)
```

- **Python 3.11 only.** `momentfm` 0.1.4 (latest on PyPI) pins `numpy==1.25.2`, which has
  no wheels for Python 3.12+, and `transformers==4.33.3`.
- **Always call `model.eval()` after `from_pretrained`.** huggingface-hub 0.24 (pinned by
  momentfm 0.1.4) skips it when loading safetensors weights, so dropout stays on and the
  reconstruction, and therefore the anomaly score, changes on every call. The official
  tutorial doesn't call it; the authors' research code does.
- **Don't upgrade torch casually.** transformers 4.33.3 calls a torch internal
  (`_register_pytree_node`) that recent torch only keeps as deprecated; the lockfile pins a
  combination that works.

## Feasibility (MacBook Air M2, 8 GB, macOS 15.7)

Measured with `scripts/check_setup.py`: reconstruction of the full tiltAPB2 test part
(154 windows of 512 points, no downsampling), model in eval mode. Peak memory is the
process peak in a single run going from small to large.

| Model | Parameters | Apple GPU (MPS) | CPU | Peak memory |
|---|---|---|---|---|
| MOMENT-1-small | 35.3M | 0.5 s | 0.9 s | 0.45 GB |
| MOMENT-1-base | 109.6M | 1.4 s | 2.8 s | 1.36 GB |
| MOMENT-1-large | 341.2M | 4.1 s | 9.5 s | 1.68 GB |

Outputs are deterministic, and MPS agrees with CPU to within 5e-6.

## Data

`data/198_UCR_Anomaly_tiltAPB2_50000_124159_124985.out`: one series from the UCR Anomaly
Archive (Wu & Keogh), copied from the official MOMENT repo. Column 0 is the value, column 1
the label. The file has no header (read it with `header=None`). Points `[0, 50000)` are the
anomaly-free training part; the anomaly is at 0-based indices `[124159, 124984]`.
