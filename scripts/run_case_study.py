"""tiltAPB2 case study: the series of the authors' anomaly-detection tutorial, re-evaluated.

Scores: MOMENT-1-large zero-shot (eval mode), k-NN (the paper's baseline), and random scores
(uniform and squared Gaussian, N_DRAWS each) as the chance level. Data handling follows the
tutorial (see zsad/data.py); --factor 10 adds the paper's x10 max-pooling (Table 22).

Usage:   uv run python scripts/run_case_study.py [--factor 10]
Outputs: results/tiltAPB2_x{factor}.csv (one row per run), results/tiltAPB2_x{factor}.md (tables),
         figures/tiltAPB2_x{factor}.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from zsad.baselines import knn_scores, random_scores
from zsad.data import prepare
from zsad.metrics import adjusted_best_f1, alert_metrics, best_f1, near_anomaly, p_value, ucr_top1
from zsad.moment import load_moment, moment_scores, pick_device

DATA = Path("data/198_UCR_Anomaly_tiltAPB2_50000_124159_124985.out")
N_DRAWS = 1000
TOLERANCE = 100  # UCR scoring: an alert within 100 original points of the anomaly counts
RANDOM = {"uniform": "Random uniform", "squared_gaussian": "Random squared Gaussian"}
SCORE_METRICS = {"adj_best_f1": "Adj. best F1 (paper)", "adj_best_f1_ranks": "Adj. best F1, rank scores",
                 "best_f1": "Best F1, point-wise", "auc_pr": "AUC-PR", "auc_roc": "AUC-ROC"}
ALERT_METRICS = {"detected": "Detected", "false_alarms": "False-alarm events",
                 "delay": "Delay (points)", "precision": "Precision", "recall": "Recall"}


def evaluate(cal, test, labels, near, factor) -> dict:
    """All metrics for one method. tau (alert threshold) = max score on the calibration part."""
    return {
        "adj_best_f1": adjusted_best_f1(labels, test),
        "adj_best_f1_ranks": adjusted_best_f1(labels, rankdata(test)),  # same ranking, new scale
        "best_f1": best_f1(labels, test),
        "auc_pr": average_precision_score(labels, test),
        "auc_roc": roc_auc_score(labels, test),
        "ucr_top1": ucr_top1(test, near),
        **alert_metrics(test, labels, cal.max(), near, factor),
    }


def summarise(v: pd.Series) -> str:
    """One table cell: a method's value, or mean ± sd (hits / draws for yes-no) over random draws."""
    if len(v) == 1:
        x = v.iloc[0]
        if isinstance(x, (bool, np.bool_)):
            return "yes" if x else "no"
        return "–" if pd.isna(x) else str(x) if isinstance(x, (int, np.integer)) else f"{x:.3f}"
    if v.dtype == bool:
        return f"{v.sum()}/{len(v)}"
    v = v.dropna().astype(float)
    return "–" if v.empty else f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std():.3f}"


def markdown(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def tables(runs: pd.DataFrame) -> str:
    groups = list(runs.groupby("method", sort=False))
    scores = markdown([[m] + [summarise(g[c]) for c in [*SCORE_METRICS, "ucr_top1"]] for m, g in groups],
                      ["Method", *SCORE_METRICS.values(), "UCR top-1"])
    p_rows = []
    for m, g in groups:
        if len(g) == 1:  # a detector: compare with the two random baselines and its own shifted scores
            nulls = [runs.loc[runs["method"] == name] for name in [*RANDOM.values(), f"{m}, circular shift"]]
            p_rows.append([m] + [" / ".join(f"{p_value(g[c].iloc[0], n[c]):.3f}" for n in nulls)
                                 for c in SCORE_METRICS])
    p_values = markdown(p_rows, ["Method", *SCORE_METRICS.values()])
    alerts = markdown([[m] + [summarise(g[c]) for c in ALERT_METRICS] for m, g in groups],
                      ["Method", *ALERT_METRICS.values()])
    return (f"## Scores\n\nRandom rows: mean ± sd over {N_DRAWS} draws; UCR top-1: draws that hit.\n\n{scores}"
            f"\n## Empirical p-values\n\np against random uniform / random squared Gaussian / the method's own "
            f"scores circularly shifted (the reference null: same values, same clustering, only the alignment "
            f"with the anomaly is broken): (1 + #draws scoring at least as well) / (1 + {N_DRAWS}).\n\n{p_values}"
            f"\n## Alerting at τ = max score on the calibration part (no labels)\n\n"
            f"Detected and false alarms use the UCR tolerance (±{TOLERANCE} original points). "
            f"Random rows: mean ± sd; Detected: draws that hit; Delay: mean over those draws.\n\n{alerts}")


def plot(d, score, reconstruction, tau, factor, path):
    anomaly = np.flatnonzero(d["labels"])
    zoom = slice(max(anomaly[0] - 3000 // factor, 0), anomaly[-1] + 3000 // factor)
    t = np.arange(len(d["test"]))
    resolution = "full resolution" if factor == 1 else f"×{factor} max-pooled"
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), facecolor="white")
    for ax in (top, bottom):
        ax.axvspan(anomaly[0], anomaly[-1], color="#8a8984", alpha=0.2, linewidth=0, label="labelled anomaly")
        ax.spines[["top", "right"]].set_visible(False)
    top.plot(t[zoom], d["test"][zoom], color="#2a78d6", linewidth=1, label=f"series ({resolution}, z-scored)")
    top.plot(t[zoom], reconstruction[zoom], color="#eb6834", linewidth=1, label="MOMENT-1-large reconstruction")
    top.legend(frameon=False, fontsize=8, loc="upper left")
    bottom.plot(t, score, color="#2a78d6", linewidth=0.7, label="score (x − x̂)²")
    bottom.axhline(tau, color="black", linestyle="--", linewidth=1, label="τ = max score on calibration part")
    bottom.legend(frameon=False, fontsize=8, loc="upper right")
    bottom.set_xlabel(f"test-part index ({resolution})")
    fig.suptitle(f"tiltAPB2 test part, {resolution}: MOMENT-1-large zero-shot", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=1, help="1 = tutorial (full resolution), 10 = paper")
    factor = parser.parse_args().factor

    d = prepare(DATA, factor)
    labels = d["labels"]
    near = near_anomaly(labels, TOLERANCE // factor)
    device = pick_device()

    model = load_moment("large", device)
    cal_score, _ = moment_scores(model, d["cal"], device)
    score, reconstruction = moment_scores(model, d["test"], device)
    detectors = {"MOMENT-1-large": (cal_score, score),
                 "k-NN (fit part)": (knn_scores(d["fit"], d["cal"]), knn_scores(d["fit"], d["test"]))}
    rows = [{"method": name, **evaluate(cal, test, labels, near, factor)} for name, (cal, test) in detectors.items()]
    # Chance with the detector's own scores: slide the whole score series in time (wrapping around),
    # which keeps the values and their clustering and only breaks the alignment with the anomaly.
    # All non-zero offsets (the standard rotation test, analysis plan D3): excluding small ones would
    # remove exactly the shifts that keep a hitting detector near the anomaly. Seed = series ID.
    offsets = np.random.default_rng(198).choice(np.arange(1, len(labels)), size=N_DRAWS, replace=False)
    for name, (cal, test) in detectors.items():
        for seed, offset in enumerate(offsets):
            rows.append({"method": f"{name}, circular shift", "seed": seed,
                         **evaluate(cal, np.roll(test, offset), labels, near, factor)})
    n_cal = len(d["cal"])
    for kind, name in RANDOM.items():
        for seed in range(N_DRAWS):
            r = random_scores(n_cal + len(d["test"]), kind, seed)
            rows.append({"method": name, "seed": seed, **evaluate(r[:n_cal], r[n_cal:], labels, near, factor)})
    runs = pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    runs.to_csv(f"results/tiltAPB2_x{factor}.csv", index=False)
    text = (f"Produced by `scripts/run_case_study.py --factor {factor}` "
            f"(test part: {len(labels)} points, {labels.sum()} labelled anomalous).\n\n" + tables(runs))
    Path(f"results/tiltAPB2_x{factor}.md").write_text(text)
    plot(d, score, reconstruction, cal_score.max(), factor, f"figures/tiltAPB2_x{factor}.png")
    print(text)


if __name__ == "__main__":
    main()
