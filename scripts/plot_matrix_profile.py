"""Matrix profile (AB-join against the fit part) on three series, "consistent" configuration.

Series chosen by a fixed rule, not by eye: tiltAPB2 (ID 198), the series at the median of the matrix
profile's VUS-PR excess (results/248/per_series.csv), and one series drawn at random (seed 0) from the
others. No label is used to compute the scores; labels only shade the anomaly in the plot.

Usage:  uv run python scripts/plot_matrix_profile.py   (after scripts/analyse_248.py)
Output: figures/248_matrix_profile.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zsad.baselines import dominant_period, matrix_profile
from zsad.ucr import STUDY_IDS, archive_members, consistent_parts, load_archive


def main():
    per_series = pd.read_csv("results/248/per_series.csv")
    mp = per_series[(per_series["detector"] == "Matrix profile") & per_series["vus_pr_excess"].notna()]
    median_id = int(mp.iloc[(mp["vus_pr_excess"] - mp["vus_pr_excess"].median()).abs().argmin()]["id"])
    others = [i for i in STUDY_IDS if i not in (198, median_id)]
    ids = [198, median_id, int(np.random.default_rng(0).choice(others))]

    members = archive_members()
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 2.6 * len(ids)), facecolor="white")
    for ax, i in zip(axes, ids):
        s = load_archive(i, members)
        p = consistent_parts(s)
        L, _ = dominant_period(np.r_[p["fit"], p["cal"]])
        score = matrix_profile(p["fit"], p["test"], L)
        anomaly = np.flatnonzero(p["labels"])
        t = np.arange(len(p["test"]))
        ax.plot(t, p["test"], color="#8a8984", linewidth=0.6, label="series (scaled)")
        ax.axvspan(anomaly[0], anomaly[-1] + 1, color="#eb6834", alpha=0.3, linewidth=0, label="labelled anomaly")
        twin = ax.twinx()
        twin.plot(t, score, color="#2a78d6", linewidth=0.8, label="matrix profile score")
        excess = mp.loc[mp["id"] == i, "vus_pr_excess"]
        ax.set_title(f"ID {i} ({s['name']}), ×{p['factor']}, L = {L}; VUS-PR excess "
                     f"{excess.iloc[0]:.3f}" if len(excess) else f"ID {i} ({s['name']}), ×{p['factor']}, L = {L}",
                     loc="left", fontsize=9)
        ax.spines[["top"]].set_visible(False)
        twin.spines[["top"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    twin.legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("test-part index (at the series' resolution)")
    fig.suptitle("Matrix profile (AB-join against the fit part): tiltAPB2, median series, random series",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/248_matrix_profile.png", dpi=150)
    print("series:", ids)


if __name__ == "__main__":
    main()
