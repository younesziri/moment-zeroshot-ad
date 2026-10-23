"""Time VUS-PR and project the run time of the 248-series study 

VUS's cost depends on the series length and the window, not on the score values, so random scores
are timed on a small grid of lengths and windows; a model t = c · n^a · w^b is fitted and applied to
the 248 series under "consistent" (archive files, the paper's x10 rule), each with its pre-registered
window L (`dominant_period` of its train part). The L values are written to results/periods.csv.

Usage:   uv run python scripts/time_vus.py
Outputs: results/vus_timing.md, results/periods.csv
"""

import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from zsad.baselines import dominant_period
from zsad.data import downsample_max
from zsad.vus_moment_research import generate_curve

UCR_ZIP = Path("data/ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
EXCLUDED = {32, 140}
LENGTHS, WINDOWS = [2000, 5000, 10000], [16, 128, 512]
# VUS calls per series implied by the plan: 7 detectors with a null (MOMENT, masked MOMENT, k-NN,
# Mu-Sigma, Med-Mad, matrix profile, the null-validity detector) x (1 + shifts), plus 10 random draws.
DETECTORS_WITH_NULL, RANDOM_DRAWS = 7, 10


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return head + (" (with uncommitted changes)" if dirty else "")


def series_info() -> pd.DataFrame:
    """Per study series under "consistent": test length after the x10 rule, and the window L of its
    train part at that resolution."""
    rows = []
    with zipfile.ZipFile(UCR_ZIP) as z:
        for name in sorted(z.namelist()):
            if "/UCR_Anomaly_FullData/" not in name or not name.endswith(".txt"):
                continue
            parts = name.split("/")[-1][:-4].split("_")
            if int(parts[0]) in EXCLUDED:
                continue
            values = np.array(z.read(name).split(), dtype=float)
            train_end = int(parts[4])
            n_test = len(values) - train_end
            factor = 10 if n_test // 10 > 2560 else 1  # the paper's rule, decided on the test part
            train = values[:train_end]
            if factor > 1:
                train = downsample_max(train, np.zeros(len(train), dtype=int), factor)[0]
                n_test = n_test // 10 + 1  # downsample_max adds one block
            L, found = dominant_period(train)
            rows.append({"id": int(parts[0]), "name": parts[3], "factor": factor, "n_test": n_test,
                         "L": L, "peak_found": found})
    return pd.DataFrame(rows)


def time_vus(n: int, window: int) -> float:
    labels = np.zeros(n, dtype=int)
    labels[n // 2 : n // 2 + 100] = 1
    scores = np.random.default_rng(0).uniform(size=n)
    start = time.perf_counter()
    generate_curve(labels, scores, window)
    return time.perf_counter() - start


def main():
    grid = [(n, w, time_vus(n, w)) for n in LENGTHS for w in WINDOWS]
    X = np.column_stack([np.ones(len(grid)), np.log([g[0] for g in grid]), np.log([g[1] for g in grid])])
    coef = np.linalg.lstsq(X, np.log([g[2] for g in grid]), rcond=None)[0]

    def predict(n, w):
        return np.exp(coef[0] + coef[1] * np.log(n) + coef[2] * np.log(w))

    info = series_info()
    info.to_csv("results/periods.csv", index=False)
    n, L = info["n_test"].to_numpy(), info["L"].to_numpy()
    lines = [f"Produced by `scripts/time_vus.py` at commit {commit()}.", "",
             "Timing of `generate_curve` (one VUS call), random scores, 100-point anomaly:", "",
             "| Test length | Window | Seconds |", "|---|---|---|",
             *[f"| {a} | {b} | {c:.2f} |" for a, b, c in grid], "",
             f"Fitted model: t = {np.exp(coef[0]):.2e} · n^{coef[1]:.2f} · w^{coef[2]:.2f} seconds.", "",
             f"Test lengths of the {len(n)} series under \"consistent\": median {np.median(n):.0f}, "
             f"max {n.max()}, total {n.sum()}.", "",
             f"Window L of the {len(L)} series (`results/periods.csv`): median {np.median(L):.0f}, quartiles "
             f"{np.percentile(L, 25):.0f}–{np.percentile(L, 75):.0f}; at the lower bound 16: {(L == 16).sum()}; "
             f"at the upper bound 512: {(L == 512).sum()}, of which no peak found: {(~info['peak_found']).sum()}.", "",
             "Projected VUS time for the whole study (consistent), with each series' own L:", ""]
    one_pass = predict(n, L).sum()
    for label, calls in [("100 shifts for all 7 detectors", DETECTORS_WITH_NULL * 101 + RANDOM_DRAWS),
                         ("100 shifts for the 3 confirmatory detectors, 20 for the other 4", 3 * 101 + 4 * 21 + RANDOM_DRAWS),
                         ("50 shifts for all 7 detectors", DETECTORS_WITH_NULL * 51 + RANDOM_DRAWS),
                         ("20 shifts for all 7 detectors", DETECTORS_WITH_NULL * 21 + RANDOM_DRAWS)]:
        lines.append(f"- {label}: {one_pass * calls / 3600:.1f} h")
    lines.insert(len(lines) - 4, f"One VUS pass over the 248 series: {one_pass:.0f} s.")
    lines.insert(len(lines) - 4, "")
    lines += ["", f"Calls per series: {DETECTORS_WITH_NULL} detectors × (1 + shifts) + {RANDOM_DRAWS} random draws; "
              "one process, no parallelism."]
    Path("results").mkdir(exist_ok=True)
    Path("results/vus_timing.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
