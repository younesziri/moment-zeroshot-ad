"""Overnight run of the 248-series study (docs/analysis_plan.md, with deviations D1-D9).

Pass 1, per series:
  - "consistent" (main analysis, UCR archive): every detector, observed metrics and circular shifts
    (D3), the random baseline (D1) and the i.i.d. random null (exploratory);
  - "paper-faithful" (reproduction, TSB-UAD with the research code's rules): observed metrics only.
Pass 2: the null validity check (D4), which needs every series' MOMENT scores from pass 1.

Resumable: one JSON file per series in results/248/raw/ (gitignored); a series already done is
skipped. A detector that fails is recorded with its error, never dropped silently.

Usage (one night, three processes in parallel, then the null validity pass once all are done):
        caffeinate -i sh -c 'for k in 0 1 2; do uv run python scripts/run_248.py --part $k/3 & done; wait;
                             uv run python scripts/run_248.py'
        uv run python scripts/run_248.py --ids 198 79 --shifts 3 --out /tmp/check  # a quick check
        caffeinate -i uv run python scripts/run_248.py --full-resolution  # exploratory, another night
Then:   uv run python scripts/analyse_248.py
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from zsad.baselines import (
    dominant_period, knn_scores, matrix_profile, matrix_profile_self, med_mad, mu_sigma, random_scores,
)
from zsad.metrics import (
    adjusted_best_f1, alert_metrics, authors_vus_window, best_f1, near_anomaly, top1_chance, ucr_top1, vus,
)
from zsad.data import split_parts
from zsad.moment import load_moment, masked_scores, moment_scores, pick_device
from zsad.ucr import (
    STUDY_IDS, archive_members, consistent_parts, factor_for, load_archive, load_tsb, null_pairing, paper_parts,
    source_group, with_context,
)

OUT = Path("results/248/raw")
TOLERANCE = 100  # UCR scoring: within 100 original points of the anomaly
RANDOM_DRAWS, IID_DRAWS = 10, 100  # D1 (descriptive); the exploratory i.i.d. null


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return head + (" (with uncommitted changes)" if dirty else "")


def observed(cal, test, labels, near, L, factor) -> dict:
    """Every metric of one score series on the test part ("consistent")."""
    vus_roc, vus_pr = vus(labels, test, L)
    m = {"vus_pr": vus_pr, "vus_roc": vus_roc, "top1": ucr_top1(test, near),
         "adj_best_f1": adjusted_best_f1(labels, test), "adj_best_f1_ranks": adjusted_best_f1(labels, rankdata(test)),
         "best_f1": best_f1(labels, test), "auc_pr": average_precision_score(labels, test),
         "auc_roc": roc_auc_score(labels, test)}
    if cal is not None:  # alerting at tau = max calibration score (no labels)
        a = alert_metrics(test, labels, cal.max(), near, factor)
        m.update(detected=a["detected"], false_alarms_per_10k=a["false_alarms"] / (len(test) * factor) * 1e4,
                 delay=a["delay"], precision=a["precision"], recall=a["recall"])
    return m


def shifted(test, labels, L, offsets) -> dict:
    """The circular-shift null (D3): the same metrics on the test scores slid by each offset."""
    ranks = rankdata(test)
    out = {k: [] for k in ["vus_pr", "vus_roc", "adj_best_f1", "adj_best_f1_ranks", "best_f1", "auc_pr", "auc_roc"]}
    for o in offsets:
        s, r = np.roll(test, o), np.roll(ranks, o)
        vus_roc, vus_pr = vus(labels, s, L)
        for k, v in [("vus_pr", vus_pr), ("vus_roc", vus_roc), ("adj_best_f1", adjusted_best_f1(labels, s)),
                     ("adj_best_f1_ranks", adjusted_best_f1(labels, r)), ("best_f1", best_f1(labels, s)),
                     ("auc_pr", average_precision_score(labels, s)), ("auc_roc", roc_auc_score(labels, s))]:
            out[k].append(v)
    return out


def run_detector(get_cal, get_test, labels, near, L, factor, offsets) -> dict:
    """A failure on the test part fails the detector. A failure on the calibration part only removes
    alerting, the one metric that uses it (analysis plan, D10). Both are recorded, never dropped."""
    out = {}
    try:
        cal = get_cal()
    except Exception as e:
        cal, out["alerting_missing"] = None, f"{type(e).__name__}: {e}"
    try:
        test = get_test()
        return {**out, "observed": observed(cal, test, labels, near, L, factor),
                "shifts": shifted(test, labels, L, offsets)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def consistent(series_id, members, model, device, n_shifts, full_resolution=False) -> tuple[dict, np.ndarray | None]:
    s = load_archive(series_id, members)
    p = consistent_parts(s)
    if full_resolution:  # exploratory (plan, section 7): no downsampling
        p = {**split_parts(s["values"], s["labels"], s["train_end"], 1), "factor": 1}
    fit, cal, test, labels, factor = p["fit"], p["cal"], p["test"], p["labels"], p["factor"]
    train = np.r_[fit, cal]
    L, peak_found = dominant_period(train)
    eps = 1e-8 * train.std()
    near = near_anomaly(labels, TOLERANCE // factor)
    offsets = np.random.default_rng(series_id).choice(np.arange(1, len(test)), size=min(n_shifts, len(test) - 1),
                                                      replace=False)
    mom = lambda x: moment_scores(model, x, device)[0]
    masked = lambda x: masked_scores(model, x, device)[0]
    moment_test = mom(test)  # computed once: used for the metrics and saved for the null validity check
    cache = {}

    def self_join():  # one profile for both parts, computed inside run_detector so a failure is recorded
        if "s" not in cache:
            cache["s"] = matrix_profile_self(np.r_[fit, cal, test], L)[len(fit):]
        return cache["s"]

    detectors = {  # (calibration scores, test scores)
        "MOMENT": (lambda: with_context(mom, fit, cal), lambda: moment_test),
        "MOMENT masked": (lambda: with_context(masked, fit, cal), lambda: masked(test)),
        "k-NN": (lambda: with_context(lambda x: knn_scores(fit, x), fit, cal), lambda: knn_scores(fit, test)),
        "Mu-Sigma": (lambda: mu_sigma(fit, cal, L, eps), lambda: mu_sigma(train, test, L, eps)),
        "Med-Mad": (lambda: med_mad(fit, cal, L, eps), lambda: med_mad(train, test, L, eps)),
        "Matrix profile": (lambda: with_context(lambda x: matrix_profile(fit, x, L), fit, cal, L),
                           lambda: matrix_profile(fit, test, L)),
        "Matrix profile, self-join": (lambda: self_join()[: len(cal)], lambda: self_join()[len(cal):]),
    }
    results = {name: run_detector(*f, labels, near, L, factor, offsets) for name, f in detectors.items()}

    random_runs, iid = [], {"vus_pr": [], "auc_pr": []}
    for d in range(IID_DRAWS):  # the first RANDOM_DRAWS draws are D1's random baseline
        r = random_scores(len(cal) + len(test), "uniform", 1000 * series_id + d)
        if d < RANDOM_DRAWS:
            random_runs.append(observed(r[: len(cal)], r[len(cal):], labels, near, L, factor))
        iid["vus_pr"].append(vus(labels, r[len(cal):], L)[1])
        iid["auc_pr"].append(average_precision_score(labels, r[len(cal):]))

    fit_end, te = int(0.6 / 0.7 * s["train_end"]), s["train_end"]
    per_part = [factor_for(fit_end), factor_for(te - fit_end), factor_for(len(s["values"]) - te)]  # the paper's rule
    meta = {"factor": factor, "L": L, "peak_found": peak_found, "n_test": len(test), "n_cal": len(cal),
            "n_fit": len(fit), "anomaly_length": int(labels.sum()), "near_points": int(near.sum()),
            "top1_chance": top1_chance(near), "series_length": len(s["values"]), "train_end": te,
            "short_cal": len(cal) < 512, "per_part_factors": per_part,
            "per_part_rule_differs": any(f != factor for f in per_part)}
    return {"meta": meta, "detectors": results, "random": random_runs, "iid_null": iid}, moment_test


def paper_faithful(series_id, model, device) -> dict:
    """Observed metrics as the authors' code computes them (D8, D9): adjusted best F1 = 0 and VUS =
    NaN when the test part has no labelled point."""
    t = load_tsb(series_id)
    p6, p5 = paper_parts(t, 0.6 / 0.7), paper_parts(t, 0.5 / 0.7)
    labels = p6["labels"]
    window = authors_vus_window(labels) if labels.any() else None
    L, _ = dominant_period(p6["fit"])
    history = np.r_[p6["fit"], p6["cal"]]
    eps = 1e-8 * history.std()

    def metrics(scores):
        if window is None:
            return {"adj_best_f1": 0.0, "vus_roc": float("nan"), "vus_pr": float("nan")}
        vus_roc, vus_pr = vus(labels, scores, window)
        return {"adj_best_f1": float(adjusted_best_f1(labels, scores)), "vus_roc": vus_roc, "vus_pr": vus_pr}

    detectors = {
        "MOMENT": lambda: moment_scores(model, p6["test"], device)[0],
        "k-NN": lambda: knn_scores(p5["fit"], p5["test"]),
        "Mu-Sigma": lambda: mu_sigma(history, p6["test"], L, eps),
        "Med-Mad": lambda: med_mad(history, p6["test"], L, eps),
        "Matrix profile": lambda: matrix_profile(p6["fit"], p6["test"], L),
    }
    results = {}
    for name, f in detectors.items():
        try:
            results[name] = metrics(f())
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {e}"}
    random_runs = [metrics(random_scores(len(labels), "uniform", 1000 * series_id + d)) for d in range(RANDOM_DRAWS)]
    meta = {"n_test_scored": len(labels), "labelled_points": int(labels.sum()), "vus_window": window, "L": L,
            "factors": [p6["fit_factor"], p6["cal_factor"], p6["test_factor"]], "train_end": t["train_end"]}
    return {"meta": meta, "detectors": results, "random": random_runs}


def save(path: Path, obj: dict):
    path.write_text(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=int, nargs="*", default=STUDY_IDS)
    parser.add_argument("--shifts", type=int, default=100)
    parser.add_argument("--out", type=Path, default=OUT, help="a separate folder for quick checks")
    parser.add_argument("--part", help="K/N: pass 1 only, on every N-th series starting at K (parallel runs)")
    parser.add_argument("--full-resolution", action="store_true",
                        help="exploratory: 50 series drawn with seed 0, no downsampling, in results/248/raw_full/")
    args = parser.parse_args()
    if args.full_resolution:
        args.ids = sorted(np.random.default_rng(0).choice(STUDY_IDS, size=50, replace=False).tolist())
        args.out = OUT.parent / "raw_full" if args.out == OUT else args.out
    if args.part:
        k, n = map(int, args.part.split("/"))
        args.ids = args.ids[k::n]
    out = args.out
    (out / "scores").mkdir(parents=True, exist_ok=True)
    members = archive_members()
    names = {i: members[i].split("/")[-1].split("_")[3] for i in STUDY_IDS}
    groups = {i: source_group(n) for i, n in names.items()}
    device = pick_device()
    model = load_moment("large", device)

    start = time.time()
    for k, i in enumerate(args.ids):  # pass 1
        path = out / f"{i:03d}.json"
        if path.exists():
            continue
        t0 = time.time()
        record = {"id": i, "name": names[i], "group": groups[i], "distorted": names[i].startswith("DISTORTED"),
                  "shifts": args.shifts, "commit": commit()}
        try:
            record["consistent"], moment_test = consistent(i, members, model, device, args.shifts, args.full_resolution)
            if moment_test is not None:
                np.save(out / "scores" / f"{i:03d}.npy", moment_test)
        except Exception as e:
            record["consistent"] = {"error": f"{type(e).__name__}: {e}"}
        try:
            if not args.full_resolution:
                record["paper_faithful"] = paper_faithful(i, model, device)
        except Exception as e:
            record["paper_faithful"] = {"error": f"{type(e).__name__}: {e}"}
        save(path, record)
        elapsed = time.time() - start
        print(f"[{k + 1}/{len(args.ids)}] series {i} ({names[i]}): {time.time() - t0:.0f} s, "
              f"elapsed {elapsed / 60:.0f} min, about {elapsed / (k + 1) * (len(args.ids) - k - 1) / 60:.0f} min left",
              flush=True)

    if args.part or args.full_resolution:  # pass 2 needs every series' MOMENT scores (main run only)
        return
    pairing = null_pairing(groups)
    for i in args.ids:  # pass 2: the null validity check (D4)
        path = out / f"{i:03d}.json"
        record = json.loads(path.read_text())
        if "null_validity" in record or "error" in record.get("consistent", {}):
            continue
        partner = out / "scores" / f"{pairing[i]:03d}.npy"
        try:
            s = load_archive(i, members)
            p = consistent_parts(s)
            labels, factor = p["labels"], p["factor"]
            L = record["consistent"]["meta"]["L"]
            near = near_anomaly(labels, TOLERANCE // factor)
            borrowed = np.resize(np.load(partner), len(labels))  # from the start, wrapping if shorter
            offsets = np.random.default_rng(i).choice(np.arange(1, len(labels)), size=min(args.shifts, len(labels) - 1),
                                                      replace=False)
            record["null_validity"] = {"partner": pairing[i], **run_detector(lambda: None, lambda: borrowed, labels, near, L,
                                                                             factor, offsets)}
        except Exception as e:
            record["null_validity"] = {"partner": pairing[i], "error": f"{type(e).__name__}: {e}"}
        save(path, record)
    print(f"done in {(time.time() - start) / 3600:.1f} h")


if __name__ == "__main__":
    main()
