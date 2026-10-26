"""Analysis of the 248-series run (docs/analysis_plan.md, sections 4-7, deviations D1-D10).

Reads results/248/raw/*.json (from scripts/run_248.py), and results/248/raw_timestral/*.json (from
scripts/run_timestral.py, D11) when present, and writes:
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

from zsad.data import downsample_max
from zsad.stats import bootstrap_ci, group_wilcoxon, mcnemar_exact, poisson_binomial_test
from zsad.ucr import STUDY_IDS, factor_for, load_tsb, paper_parts

RAW, OUT = Path("results/248/raw"), Path("results/248")
RAW_TIMESTRAL = Path("results/248/raw_timestral")  # scripts/run_timestral.py (D11)
SHIFT_METRICS = ["vus_pr", "vus_roc", "adj_best_f1", "adj_best_f1_ranks", "best_f1", "auc_pr", "auc_roc"]
DETECTORS = ["MOMENT", "MOMENT masked", "k-NN", "Mu-Sigma", "Med-Mad", "Matrix profile", "Matrix profile, self-join"]
PAPER_DETECTORS, NOT_IN_PAPER = ["MOMENT", "k-NN"], ["Mu-Sigma", "Med-Mad", "Matrix profile"]
ALPHA = 0.05


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    # Only code counts: this script's own outputs (results/, figures/) are tracked and change on every run.
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "pyproject.toml", "uv.lock"],
                           capture_output=True, text=True).stdout.strip()
    return head + (" (with uncommitted changes)" if dirty else "")


def detector_row(base: dict, name: str, result: dict, top1_chance: float) -> dict:
    row = {**base, "detector": name}
    if "error" in result:
        return {**row, "error": result["error"]}
    row.update(result["observed"])
    row["alerting_missing"] = result.get("alerting_missing")
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
    consistent, paper, metas, failures = [], [], [], []
    for f in sorted(RAW.glob("*.json")):
        r = json.loads(f.read_text())
        base = {"id": r["id"], "name": r["name"], "group": r["group"], "distorted": r["distorted"]}
        c = r["consistent"]
        if "error" in c:
            failures.append((r["id"], "consistent", "all", c["error"]))
        else:
            meta = c["meta"]
            metas.append({"id": r["id"], **meta})
            base_c = {**base, **{k: meta[k] for k in ["factor", "L", "n_test", "anomaly_length", "series_length"]}}
            detectors = {**c["detectors"], **({"Null validity (D4)": r["null_validity"]} if "null_validity" in r else {})}
            t = RAW_TIMESTRAL / f.name  # same parts as the main run, so the same meta and top-1 chance apply
            if t.exists():
                tr = json.loads(t.read_text())["timestral"]
                if "error" in tr:
                    failures.append((r["id"], "consistent", "Timestral", tr["error"]))
                else:
                    detectors.update(tr["detectors"])
            for name, result in detectors.items():
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
            pm = {k: p["meta"].get(k) for k in ["labelled_points", "labelled_points_whole_test", "anomaly_in_train_part"]}
            for name, result in p["detectors"].items():
                if "error" in result:
                    failures.append((r["id"], "paper-faithful", name, result["error"]))
                paper.append({**base, "detector": name, **result, **pm})
            paper.append({**base, "detector": "Random (D1)", **pd.DataFrame(p["random"]).mean().to_dict(), **pm})
    return pd.DataFrame(consistent), pd.DataFrame(paper), pd.DataFrame(metas), failures


def md_table(df: pd.DataFrame) -> str:
    def cell(column, v):
        if not isinstance(v, float):
            return str(v)
        if str(column).startswith("p") and v < 0.001:
            return "< 0.001"
        return f"{v:.3f}"
    cells = df.astype(object).where(df.notna(), "–")
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(cell(c, v) for c, v in zip(df.columns, r)) + " |"
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
    out = {"n": len(d), "mean": d["d"].mean(), "CI clustered": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
           "CI plain": "[{:.4f}, {:.4f}]".format(*bootstrap_ci(d["d"].to_numpy())), "p": p}
    if favour:  # reported only; the decision uses the two-sided p
        out["p one-sided (in favour)"] = group_wilcoxon(d["d"].to_numpy(), d["group"].to_numpy(), "greater")
    return {**out, "significant": sig}


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


def per_detector(ok: pd.DataFrame) -> pd.DataFrame:
    return ok[ok["detector"] != "Random (D1)"].groupby("detector").agg(
        n=("id", "count"), vus_pr=("vus_pr", "mean"), vus_pr_chance=("vus_pr_chance", "mean"),
        vus_pr_excess=("vus_pr_excess", "mean"), share_p_le_005=("vus_pr_p", lambda p: np.mean(p <= ALPHA)),
        top1=("top1", "mean"), top1_chance=("top1_chance", "mean"), adj_best_f1=("adj_best_f1", "mean"),
        adj_best_f1_chance=("adj_best_f1_chance", "mean"), adj_best_f1_excess=("adj_best_f1_excess", "mean"),
        detection_rate=("detected", "mean"),
        false_alarms_per_10k=("false_alarms_per_10k", "mean")).reset_index()


def summary_table(kept: pd.DataFrame, detectors: list[str], metric: str, table7: dict | None) -> pd.DataFrame:
    """Table 7's columns; ranks and wins/losses among `detectors` (series where all of them ran)."""
    wide = kept[kept["detector"].isin(detectors)].pivot(index="id", columns="detector", values=metric).dropna()
    ranks = wide.apply(lambda r: pd.Series(rankdata(-r.to_numpy()), index=r.index), axis=1)
    rows = []
    for d in detectors + ["Random (D1)"] * (table7 is None):
        v = kept.loc[kept["detector"] == d, metric]
        row = {"detector": d, "n": len(v), "mean": v.mean(), "median": v.median(), "std": v.std()}
        if table7 is not None:
            row["Table 7"] = table7.get(d, "–")
        if d in detectors:
            wins = sum((wide[d] > wide[o]).sum() for o in detectors if o != d)
            losses = sum((wide[d] < wide[o]).sum() for o in detectors if o != d)
            row.update({"mean rank": ranks[d].mean(), "wins/losses": f"{wins}/{losses}"})
        rows.append(row)
    return pd.DataFrame(rows)


def paper_data_facts() -> pd.DataFrame:
    """Facts about the authors' data under their rules (no model needed): labelled test blocks in
    total and in the scored part (the tail is dropped), anomaly in the training part, and whether
    k-NN's fit split (5/7) and the test part get different downsampling decisions."""
    rows = []
    for i in STUDY_IDS:
        t = load_tsb(i)
        te, test_labels = t["train_end"], t["labels"][t["train_end"]:]
        factor = factor_for(len(test_labels))
        total = (downsample_max(test_labels, test_labels, 10)[1] if factor > 1 else test_labels).sum()
        rows.append({"id": i, "labelled_test": int(total), "labelled_scored": int(paper_parts(t, 0.6 / 0.7)["labels"].sum()),
                     "anomaly_in_train_part": bool(t["labels"][:te].any()),
                     "knn_resolution_mismatch": factor_for(int(0.5 / 0.7 * te)) != factor})
    return pd.DataFrame(rows).set_index("id")


def reproduction(paper: pd.DataFrame) -> str:
    """Table 7's format. Their pipeline drops any row with a NaN (D8): n is printed."""
    kept = paper.dropna(subset=["adj_best_f1", "vus_roc", "vus_pr"])
    out = []
    for metric, table7 in [("adj_best_f1", {"MOMENT": 0.585, "k-NN": 0.554}), ("vus_roc", {"MOMENT": 0.670, "k-NN": 0.706})]:
        out.append(f"**{metric}**, MOMENT and k-NN (rows with a NaN dropped, as the authors' pipeline does; "
                   "ranks and wins/losses between the two):\n\n" + md_table(summary_table(kept, PAPER_DETECTORS, metric, table7)))
        out.append(f"**{metric}, not in the paper**: the course baselines and the random baseline (D1), same series; "
                   "ranks and wins/losses among the three course baselines:\n\n"
                   + md_table(summary_table(kept, NOT_IN_PAPER, metric, None)))
    adj = paper[paper["detector"].isin(PAPER_DETECTORS)].groupby("detector")["adj_best_f1"]
    out.append("Sensitivity, adjusted best F1 over all series with unscorable ones at 0: "
               + ", ".join(f"{d} {m:.3f} (n = {n})" for d, m, n in zip(adj.mean().index, adj.mean(), adj.count())))
    facts = paper_data_facts()
    reasons = []
    for i, r in facts[facts["labelled_scored"] == 0].iterrows():
        if r["anomaly_in_train_part"]:
            reasons.append(f"{i}: anomaly in the training part (pre-correction file name)")
        elif r["labelled_test"] == 0:
            reasons.append(f"{i}: no labelled point in the TSB-UAD file")
        else:
            reasons.append(f"{i}: anomaly entirely in the unscored tail")
    out.append(f"Series with nothing to score in the TSB-UAD test part ({len(reasons)}, scored as the authors' code "
               "does): " + "; ".join(reasons) + ".")
    cut = facts[(facts["labelled_scored"] > 0) & (facts["labelled_scored"] < facts["labelled_test"])]
    out.append(f"Series whose anomaly is partly in the unscored tail ({len(cut)}): "
               + ", ".join(f"{i} ({r.labelled_scored} of {r.labelled_test} labelled points or blocks scored)"
                           for i, r in cut.iterrows()) + ".")
    out.append("k-NN's fit split (5/7) and test part downsampled differently under the per-part rule: "
               f"{int(facts['knn_resolution_mismatch'].sum())} series.")
    t = paper[(paper["id"] == 198) & (paper["detector"] == "MOMENT")]
    if len(t):
        out.append(f"tiltAPB2 (ID 198), MOMENT: adjusted best F1 {t['adj_best_f1'].iloc[0]:.3f} (Table 22: 0.96), "
                   f"VUS-ROC {t['vus_roc'].iloc[0]:.3f} (Table 22: 0.75).")
    return "\n\n".join(out) + "\n"


def main():
    c, paper, metas, failures = load()
    OUT.mkdir(parents=True, exist_ok=True)
    c.to_csv(OUT / "per_series.csv", index=False)
    paper.to_csv(OUT / "paper_faithful.csv", index=False)
    ok = c[c["error"].isna()] if "error" in c else c
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
    family3 = fixed_sequence([  # D11, same rules as family 1
        ("1. VUS-PR, Timestral vs chance", continuous_test(paired(ok, "Timestral", None, "vus_pr_excess"))),
        ("2. VUS-PR, Timestral vs MOMENT (difference of excesses)",
         continuous_test(paired(ok, "Timestral", "MOMENT", "vus_pr_excess"))),
        ("3. top-1, Timestral vs chance (|Z|/n)", top1_vs_chance(ok, "Timestral")),
        ("4. top-1, Timestral vs MOMENT (McNemar)", top1_paired(ok, "Timestral", "MOMENT")),
    ]) if "Timestral" in set(ok["detector"]) else None

    ran = ok[ok["detector"].isin(DETECTORS)].groupby("id")["detector"].nunique()
    common = ran[ran == len(DETECTORS)].index
    random_row = ok[ok["detector"] == "Random (D1)"][["vus_pr", "top1", "adj_best_f1", "detected",
                                                      "false_alarms_per_10k"]].mean()
    alert_missing = ok[ok["alerting_missing"].notna()][["id", "detector", "alerting_missing"]] if "alerting_missing" in ok else []

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
    rho_a, p_a = spearmanr(m["anomaly_length"], excess)
    rho_s, p_s = spearmanr(m["series_length"], excess)
    p_d = mannwhitneyu(excess[m["distorted"]], excess[~m["distorted"]]).pvalue
    by_group = m.groupby("group")["vus_pr_excess"].agg(["count", "mean"]).reset_index()

    lines = [
        f"Produced by `scripts/analyse_248.py` at commit {commit()}.", "",
        "## Coverage", "",
        f"- Series with results: {c['id'].nunique()}; the paper's per-part downsampling rule would differ from ours "
        f"(one factor per series): {int(metas['per_part_rule_differs'].sum())}; calibration part shorter than 512: "
        f"{int(metas['short_cal'].sum())}; no autocorrelation peak (L = 512): {int((~metas['peak_found']).sum())}.",
        f"- Failures ({len(failures)}):", "",
        md_table(pd.DataFrame(failures, columns=["id", "configuration", "detector", "error"])) if failures else "none\n",
        f"- Alerting missing, other metrics kept (D10) ({len(alert_missing)}):", "",
        md_table(alert_missing) if len(alert_missing) else "none\n",
        "## Family 1 (Q1–Q2), fixed sequence at α = 0.05", "",
        "Significant = clustered bootstrap 95% interval excludes 0 in MOMENT's favour and the two-sided p < 0.05. "
        "The one-sided p is reported, not used for decisions.", "",
        md_table(family1),
        "## Family 2 (Q3), masked vs unmasked, fixed sequence at α = 0.05", "",
        "Two-sided: significant = clustered interval excludes 0 (either direction) and p < 0.05.", "",
        md_table(family2),
        "## Family 3 (Q4, D11), Timestral-209M, fixed sequence at α = 0.05", "",
        "Same rules as family 1. Timestral vs k-NN and vs the matrix profile: descriptive (tables below); "
        "\"Timestral, scaled\" (|x − median| / (q0.9 − q0.1)) is exploratory.", "",
        md_table(family3) if family3 is not None else "Not run.\n",
        "## Per detector (\"consistent\")", "",
        "Means over series; chance = mean over circular shifts (exact |Z|/n for top-1); share_p_le_005 = "
        "share of series beating their own null at p ≤ 0.05 (5% expected by chance); alerting at τ = max "
        "calibration score.", "",
        "All series each detector ran on:", "",
        md_table(per_detector(ok)),
        f"Only the {len(common)} series where all {len(DETECTORS)} detectors ran:", "",
        md_table(per_detector(ok[ok["id"].isin(common)])),
        "Random baseline (D1, descriptive, all series): " + ", ".join(f"{k} {v:.3f}" for k, v in random_row.items()), "",
        "## Null validity check (D4)", "",
        f"Share of series with p ≤ 0.05 (VUS-PR): {np.mean(nv['vus_pr_p'] <= ALPHA):.3f} (n = {len(nv)}); "
        "histogram in `figures/248_null_validity.png`." if len(nv) else "Not run.", "",
        "## Reproduction (\"paper-faithful\")", "",
        reproduction(paper) if len(paper) else "Not run.\n",
        "## Exploratory (unadjusted p-values; hypotheses, not findings)", "",
        md_table(pd.DataFrame(explore)),
        f"i.i.d. null vs circular shift, MOMENT, series where they disagree at p ≤ 0.05: {disagree}", "",
        f"MOMENT's VUS-PR excess against anomaly length: Spearman rho {rho_a:.3f}, p {p_a:.3f}; against series "
        f"length: rho {rho_s:.3f}, p {p_s:.3f}; DISTORTED vs not: Mann-Whitney p {p_d:.3f}.", "",
        "MOMENT's mean VUS-PR excess per source group (D5):", "",
        md_table(by_group),
    ]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
