"""Analysis of the 248-series run (docs/analysis_plan.md, sections 4-7, deviations D1-D9).

Reads results/248/raw/*.json (from scripts/run_248.py) and writes:
  results/248/per_series.csv      one row per series and detector ("consistent"): observed metrics,
                                  chance level (mean over circular shifts; exact |Z|/n for top-1),
                                  excess and per-series p-value;
  results/248/paper_faithful.csv  one row per series and detector ("paper-faithful");
  results/248/summary.md          the plan's tables; figures/248_null_validity.png.

Usage:  uv run python scripts/analyse_248.py
"""

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr

from zsad.stats import bootstrap_ci, group_wilcoxon, mcnemar_exact, poisson_binomial_test

RAW, OUT = Path("results/248/raw"), Path("results/248")
SHIFT_METRICS = ["vus_pr", "vus_roc", "adj_best_f1", "adj_best_f1_ranks", "best_f1", "auc_pr", "auc_roc"]
ALPHA = 0.05


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return head + (" (with uncommitted changes)" if dirty else "")


def detector_row(base: dict, name: str, result: dict, top1_chance: float) -> dict:
    row = {**base, "detector": name}
    if "error" in result:
        return {**row, "error": result["error"]}
    row.update(result["observed"])
    for m in SHIFT_METRICS:
        null = np.asarray(result["shifts"][m], dtype=float)
        obs = row[m]
        row[f"{m}_chance"] = null.mean()
        row[f"{m}_excess"] = obs - null.mean()
        row[f"{m}_p"] = (1 + np.sum(null >= obs)) / (1 + len(null))
    row["top1_chance"] = top1_chance
    row["top1_excess"] = float(row["top1"]) - top1_chance
    return row


def load():
    consistent, paper, failures = [], [], []
    for f in sorted(RAW.glob("*.json")):
        r = json.loads(f.read_text())
        base = {"id": r["id"], "name": r["name"], "group": r["group"], "distorted": r["distorted"]}
        c = r["consistent"]
        if "error" in c:
            failures.append((r["id"], "consistent", "all", c["error"]))
        else:
            meta = c["meta"]
            base_c = {**base, **{k: meta[k] for k in ["factor", "L", "peak_found", "n_test", "anomaly_length",
                                                         "series_length", "short_cal", "per_part_rule_differs"]}}
            for name, result in {**c["detectors"], **({"Null validity (D4)": r["null_validity"]}
                                                      if "null_validity" in r else {})}.items():
                row = detector_row(base_c, name, result, meta["top1_chance"])
                if name == "MOMENT" and "error" not in row:  # exploratory: the i.i.d. random null
                    for mt in ["vus_pr", "auc_pr"]:
                        null = np.asarray(c["iid_null"][mt], dtype=float)
                        row[f"{mt}_p_iid"] = (1 + np.sum(null >= row[mt])) / (1 + len(null))
                consistent.append(row)
                if "error" in row:
                    failures.append((r["id"], "consistent", name, row["error"]))
            rnd = pd.DataFrame(c["random"]).mean(numeric_only=True).to_dict()
            consistent.append({**base_c, "detector": "Random (D1)", **rnd})
        p = r.get("paper_faithful", {})
        if "error" in p:
            failures.append((r["id"], "paper-faithful", "all", p["error"]))
        elif p:
            for name, result in p["detectors"].items():
                if "error" in result:
                    failures.append((r["id"], "paper-faithful", name, result["error"]))
                paper.append({**base, "detector": name, **result, "labelled_points": p["meta"]["labelled_points"]})
            paper.append({**base, "detector": "Random (D1)", **pd.DataFrame(p["random"]).mean().to_dict(),
                          "labelled_points": p["meta"]["labelled_points"]})
    return pd.DataFrame(consistent), pd.DataFrame(paper), failures


def md_table(df: pd.DataFrame) -> str:
    cells = df.astype(object).where(df.notna(), "–")
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
                              for r in cells.itertuples(index=False)]) + "\n"


def paired(c: pd.DataFrame, a: str, b: str | None, column: str) -> pd.DataFrame:
    """Per-series values of `column` for detector a, minus detector b (or alone), with groups."""
    x = c[c["detector"] == a].set_index("id")
    if b is None:
        return x[[column, "group"]].dropna().rename(columns={column: "d"})
    y = c[c["detector"] == b].set_index("id")
    d = (x[column] - y[column]).dropna()
    return pd.DataFrame({"d": d, "group": x.loc[d.index, "group"]})


def continuous_test(d: pd.DataFrame, favour: bool = True) -> dict:
    ci = bootstrap_ci(d["d"].to_numpy(), d["group"].to_numpy())
    p = group_wilcoxon(d["d"].to_numpy(), d["group"].to_numpy())
    sig = (ci[0] > 0 if favour else (ci[0] > 0 or ci[1] < 0)) and p < ALPHA
    return {"n": len(d), "mean": d["d"].mean(), "CI clustered": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            "CI plain": "[{:.4f}, {:.4f}]".format(*bootstrap_ci(d["d"].to_numpy())), "p": p, "significant": sig}


def top1_vs_chance(c: pd.DataFrame, detector: str) -> dict:
    x = c[(c["detector"] == detector) & c["top1"].notna()]
    hits, chance = x["top1"].astype(bool).to_numpy(), x["top1_chance"].to_numpy()
    d = pd.DataFrame({"d": hits - chance, "group": x["group"].to_numpy()})
    ci = bootstrap_ci(d["d"].to_numpy(), d["group"].to_numpy())
    p = poisson_binomial_test(hits, chance)
    return {"n": len(x), "mean": d["d"].mean(), "CI clustered": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            "CI plain": "[{:.4f}, {:.4f}]".format(*bootstrap_ci(d["d"].to_numpy())), "p": p,
            "significant": ci[0] > 0 and p < ALPHA, "hits": int(hits.sum()), "expected": round(chance.sum(), 2)}


def top1_paired(c: pd.DataFrame, a: str, b: str, favour: bool = True) -> dict:
    x = c[c["detector"] == a].set_index("id")["top1"].dropna()
    y = c[c["detector"] == b].set_index("id")["top1"].dropna()
    ids = x.index.intersection(y.index)
    ha, hb = x[ids].astype(bool).to_numpy(), y[ids].astype(bool).to_numpy()
    groups = c[c["detector"] == a].set_index("id").loc[ids, "group"].to_numpy()
    ci = bootstrap_ci((ha.astype(float) - hb), groups)
    p = mcnemar_exact(ha, hb)
    sig = (ci[0] > 0 if favour else (ci[0] > 0 or ci[1] < 0)) and p < ALPHA
    return {"n": len(ids), "mean": float(np.mean(ha.astype(float) - hb)), "CI clustered": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            "CI plain": "[{:.4f}, {:.4f}]".format(*bootstrap_ci(ha.astype(float) - hb)), "p": p, "significant": sig,
            "hits": f"{int(ha.sum())} vs {int(hb.sum())}"}


def fixed_sequence(steps: list[tuple[str, dict]]) -> pd.DataFrame:
    rows, open_ = [], True
    for label, r in steps:
        rows.append({"test": label, **r, "confirmatory": "yes" if open_ and r["significant"] else
                     ("no (sequence stopped earlier)" if not open_ else "no")})
        open_ = open_ and r["significant"]
    return pd.DataFrame(rows)


def reproduction(paper: pd.DataFrame) -> str:
    """Table 7's format. Their pipeline drops any row with a NaN (D8): n is printed."""
    detectors = ["MOMENT", "k-NN", "Mu-Sigma", "Med-Mad", "Matrix profile"]
    out = []
    for metric, table7 in [("adj_best_f1", {"MOMENT": 0.585, "k-NN": 0.554}), ("vus_roc", {"MOMENT": 0.670, "k-NN": 0.706})]:
        rows = []
        kept = paper.dropna(subset=["adj_best_f1", "vus_roc", "vus_pr"])
        wide = kept[kept["detector"].isin(detectors)].pivot(index="id", columns="detector", values=metric).dropna()
        ranks = wide.apply(lambda r: pd.Series(rankdata(-r.to_numpy()), index=r.index), axis=1)
        for d in detectors + ["Random (D1)"]:
            v = kept.loc[kept["detector"] == d, metric]
            row = {"detector": d, "n": len(v), "mean": v.mean(), "median": v.median(), "std": v.std(),
                   "Table 7": table7.get(d, "–")}
            if d in detectors:
                wins = sum((wide[d] > wide[o]).sum() for o in detectors if o != d)
                losses = sum((wide[d] < wide[o]).sum() for o in detectors if o != d)
                row.update({"mean rank": ranks[d].mean(), "wins/losses": f"{wins}/{losses}"})
            rows.append(row)
        out.append(f"**{metric}** (rows with a NaN dropped, as the authors' pipeline does):\n\n" + md_table(pd.DataFrame(rows)))
    adj = paper[paper["detector"].isin(["MOMENT", "k-NN"])].groupby("detector")["adj_best_f1"]
    out.append("Sensitivity, adjusted best F1 over all series with unscorable ones at 0: "
               + ", ".join(f"{d} {m:.3f} (n = {n})" for d, m, n in zip(adj.mean().index, adj.mean(), adj.count())))
    unscorable = sorted(paper.loc[paper["labelled_points"] == 0, "id"].unique().tolist())
    out.append(f"Series with no labelled anomaly in the TSB-UAD test part (scored as the authors' code does): {unscorable}")
    t = paper[(paper["id"] == 198) & (paper["detector"] == "MOMENT")]
    if len(t):
        out.append(f"tiltAPB2 (ID 198), MOMENT: adjusted best F1 {t['adj_best_f1'].iloc[0]:.3f} (Table 22: 0.96), "
                   f"VUS-ROC {t['vus_roc'].iloc[0]:.3f} (Table 22: 0.75).")
    return "\n\n".join(out) + "\n"


def main():
    c, paper, failures = load()
    OUT.mkdir(parents=True, exist_ok=True)
    c.to_csv(OUT / "per_series.csv", index=False)
    paper.to_csv(OUT / "paper_faithful.csv", index=False)
    ok = c[c.get("error", pd.Series(index=c.index, dtype=object)).isna()] if "error" in c else c
    m = ok[ok["detector"] == "MOMENT"]

    family1 = fixed_sequence([
        ("1. VUS-PR, MOMENT vs chance", continuous_test(paired(ok, "MOMENT", None, "vus_pr_excess"))),
        ("2. VUS-PR, MOMENT vs k-NN (difference of excesses)", continuous_test(paired(ok, "MOMENT", "k-NN", "vus_pr_excess"))),
        ("3. top-1, MOMENT vs chance (|Z|/n)", top1_vs_chance(ok, "MOMENT")),
        ("4. top-1, MOMENT vs k-NN (McNemar)", top1_paired(ok, "MOMENT", "k-NN")),
    ])
    family2 = fixed_sequence([
        ("1. VUS-PR, masked vs unmasked (difference of excesses)",
         continuous_test(paired(ok, "MOMENT masked", "MOMENT", "vus_pr_excess"), favour=False)),
        ("2. top-1, masked vs unmasked (McNemar)", top1_paired(ok, "MOMENT masked", "MOMENT", favour=False)),
    ])

    per_detector = ok[ok["detector"] != "Random (D1)"].groupby("detector").agg(
        n=("id", "count"), vus_pr=("vus_pr", "mean"), vus_pr_chance=("vus_pr_chance", "mean"),
        vus_pr_excess=("vus_pr_excess", "mean"), share_p_le_005=("vus_pr_p", lambda p: np.mean(p <= ALPHA)),
        top1=("top1", "mean"), top1_chance=("top1_chance", "mean"), adj_best_f1=("adj_best_f1", "mean"),
        adj_best_f1_chance=("adj_best_f1_chance", "mean"), detection_rate=("detected", "mean"),
        false_alarms_per_10k=("false_alarms_per_10k", "mean")).reset_index()
    random_row = ok[ok["detector"] == "Random (D1)"][["vus_pr", "top1", "adj_best_f1", "detected",
                                                      "false_alarms_per_10k"]].mean()

    nv = ok[ok["detector"] == "Null validity (D4)"]
    if len(nv):
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="white")
        ax.hist(nv["vus_pr_p"], bins=np.linspace(0, 1, 21), color="#2a78d6")
        ax.set_xlabel("per-series p-value, VUS-PR, scores from another series")
        ax.set_ylabel("series")
        ax.set_title("Null validity check: should be roughly flat", loc="left", fontsize=10)
        fig.tight_layout()
        Path("figures").mkdir(exist_ok=True)
        fig.savefig("figures/248_null_validity.png", dpi=150)
        plt.close(fig)

    explore = []
    for metric in ["auc_pr", "auc_roc", "adj_best_f1_ranks"]:
        d = paired(ok, "MOMENT", None, f"{metric}_excess")
        explore.append({"metric": metric, "n": len(d), "mean excess": d["d"].mean(),
                        "p (group Wilcoxon, unadjusted)": group_wilcoxon(d["d"].to_numpy(), d["group"].to_numpy())})
    disagree = {mt: int(((m[f"{mt}_p"] <= ALPHA) != (m[f"{mt}_p_iid"] <= ALPHA)).sum()) for mt in ["vus_pr", "auc_pr"]
                if f"{mt}_p_iid" in m}
    excess = m["vus_pr_excess"]
    associations = {
        "anomaly length (Spearman rho, p)": spearmanr(m["anomaly_length"], excess),
        "series length (Spearman rho, p)": spearmanr(m["series_length"], excess),
        "DISTORTED vs not (Mann-Whitney p)": mannwhitneyu(excess[m["distorted"]], excess[~m["distorted"]]).pvalue
        if m["distorted"].any() and (~m["distorted"]).any() else float("nan"),
    }
    by_group = m.groupby("group")["vus_pr_excess"].agg(["count", "mean"]).reset_index()

    meta = m.drop_duplicates("id")
    lines = [
        f"Produced by `scripts/analyse_248.py` at commit {commit()}.", "",
        "## Coverage", "",
        f"- Series with results: {c['id'].nunique()}; per-part downsampling rule differs from ours: "
        f"{int(meta['per_part_rule_differs'].sum())}; calibration part shorter than 512: {int(meta['short_cal'].sum())}; "
        f"no autocorrelation peak (L = 512): {int((~meta['peak_found'].astype(bool)).sum())}.",
        f"- Failures ({len(failures)}):", "",
        md_table(pd.DataFrame(failures, columns=["id", "configuration", "detector", "error"])) if failures else "none\n",
        "## Family 1 (Q1–Q2), fixed sequence at α = 0.05", "",
        "Significant = clustered bootstrap 95% interval excludes 0 in MOMENT's favour and p < 0.05.", "",
        md_table(family1),
        "## Family 2 (Q3), masked vs unmasked, fixed sequence at α = 0.05", "",
        "Two-sided: significant = clustered interval excludes 0 (either direction) and p < 0.05.", "",
        md_table(family2),
        "## Per detector (\"consistent\")", "",
        "Means over series; chance = mean over circular shifts (exact |Z|/n for top-1); share_p_le_005 = "
        "share of series beating their own null at p ≤ 0.05 (5% expected by chance); alerting at τ = max "
        "calibration score.", "",
        md_table(per_detector),
        "Random baseline (D1, descriptive): " + ", ".join(f"{k} {v:.3f}" for k, v in random_row.items()), "",
        "## Null validity check (D4)", "",
        f"Share of series with p ≤ 0.05 (VUS-PR): {np.mean(nv['vus_pr_p'] <= ALPHA):.3f} (n = {len(nv)}); "
        "histogram in `figures/248_null_validity.png`." if len(nv) else "Not run.", "",
        "## Reproduction (\"paper-faithful\")", "",
        reproduction(paper) if len(paper) else "Not run.\n",
        "## Exploratory (unadjusted p-values; hypotheses, not findings)", "",
        md_table(pd.DataFrame(explore)),
        f"i.i.d. null vs circular shift, MOMENT, series where they disagree at p ≤ 0.05: {disagree}", "",
        "MOMENT's VUS-PR excess against series properties: "
        + "; ".join(f"{k}: {v[0]:.3f}, {v[1]:.3f}" if isinstance(v, tuple) or hasattr(v, "statistic") else f"{k}: {v:.3f}"
                    for k, v in associations.items()), "",
        "MOMENT's mean VUS-PR excess per source group (D5):", "",
        md_table(by_group),
    ]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
