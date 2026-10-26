"""Timestral on the 248 series, "consistent" configuration only (analysis plan, D11).

Same parts, window L, circular shifts (same offsets, seed = series ID) and metrics as scripts/run_248.py,
whose functions it reuses. Resumable: one JSON file per series in results/248/raw_timestral/ (gitignored).

Usage:  caffeinate -i uv run python scripts/run_timestral.py  # one process: on CPU, torch already uses every core
        uv run python scripts/run_timestral.py --ids 198 --out /tmp/check  # timing on tiltAPB2 (D11 fallback rule)
Then:   uv run python scripts/analyse_248.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_248 import TOLERANCE, commit, run_detector, save  # noqa: E402

from zsad.baselines import dominant_period  # noqa: E402
from zsad.metrics import near_anomaly  # noqa: E402
from zsad.timestral import REVISIONS, load_timestral, timestral_scores  # noqa: E402
from zsad.ucr import STUDY_IDS, archive_members, consistent_parts, load_archive, source_group  # noqa: E402

OUT = Path("results/248/raw_timestral")


def run_series(series_id, members, model, n_shifts) -> dict:
    p = consistent_parts(load_archive(series_id, members))
    fit, cal, test, labels, factor = p["fit"], p["cal"], p["test"], p["labels"], p["factor"]
    L, _ = dominant_period(np.r_[fit, cal])
    near = near_anomaly(labels, TOLERANCE // factor)
    offsets = np.random.default_rng(series_id).choice(np.arange(1, len(test)), size=min(n_shifts, len(test) - 1),
                                                      replace=False)  # the same offsets as run_248.py
    t0 = time.time()
    cache = {}

    def scores(part):  # one forecast pass gives both scores; inside run_detector so a failure is recorded
        if part not in cache:
            cache[part] = timestral_scores(model, fit, cal) if part == "cal" else timestral_scores(model, np.r_[fit, cal], test)
        return cache[part]

    detectors = {
        "Timestral": (lambda: scores("cal")[0], lambda: scores("test")[0]),
        "Timestral, scaled": (lambda: scores("cal")[1], lambda: scores("test")[1]),  # exploratory
    }
    results = {}
    for name, f in detectors.items():
        results[name] = run_detector(*f, labels, near, L, factor, offsets)
        if name == "Timestral":
            forecast_seconds = time.time() - t0  # the forecasts are made while scoring the first detector
    meta = {"factor": factor, "L": L, "n_test": len(test), "n_cal": len(cal), "forecast_seconds": forecast_seconds,
            "total_seconds": time.time() - t0}
    return {"meta": meta, "detectors": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=int, nargs="*", default=STUDY_IDS)
    parser.add_argument("--size", default="209M", choices=list(REVISIONS), help="28M is D11's fallback")
    parser.add_argument("--shifts", type=int, default=100)
    parser.add_argument("--out", type=Path, default=OUT, help="a separate folder for quick checks")
    parser.add_argument("--part", help="K/N: every N-th series starting at K (parallel runs)")
    args = parser.parse_args()
    if args.part:
        k, n = map(int, args.part.split("/"))
        args.ids = args.ids[k::n]
    args.out.mkdir(parents=True, exist_ok=True)
    members = archive_members()
    names = {i: members[i].split("/")[-1].split("_")[3] for i in STUDY_IDS}
    model = load_timestral(args.size, "cpu")  # its scaler works in float64, which MPS lacks

    start = time.time()
    for k, i in enumerate(args.ids):
        path = args.out / f"{i:03d}.json"
        if path.exists():
            continue
        record = {"id": i, "name": names[i], "group": source_group(names[i]), "model": f"Timestral-{args.size}",
                  "revision": REVISIONS[args.size], "shifts": args.shifts, "commit": commit()}
        try:
            record["timestral"] = run_series(i, members, model, args.shifts)
        except Exception as e:
            record["timestral"] = {"error": f"{type(e).__name__}: {e}"}
        save(path, record)
        elapsed = time.time() - start
        print(f"[{k + 1}/{len(args.ids)}] series {i} ({names[i]}): {record['timestral'].get('meta', {})}, "
              f"elapsed {elapsed / 60:.0f} min, about {elapsed / (k + 1) * (len(args.ids) - k - 1) / 60:.0f} min left",
              flush=True)


if __name__ == "__main__":
    main()
