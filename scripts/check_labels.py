"""Label check for the 248-series study (docs/analysis_plan.md, section 1, "Labels").

For every TSB-UAD KDD21 file, compare the labelled anomaly (label column) with the range in the file
name, read two ways:
  - "archive": the UCR archive's convention, 1-based inclusive -> 0-based [begin - 1, end - 1];
  - "[begin, end)": 0-based, end excluded -> 0-based [begin, end - 1] (empty when begin = end).
Also: empty labels, several labelled segments, missing values, file names that differ from the
original archive's, and values that differ from the archive's (the "consistent" configuration uses
the archive, "paper-faithful" uses TSB-UAD, so both sources must be compared). No detector is run.

Usage:   uv run python scripts/check_labels.py   (after scripts/download_data.py)
Outputs: results/label_check.csv (one row per file), results/label_check.md (summary)
"""

import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from zsad.metrics import segments

KDD21 = Path("data/TSB-UAD-Public/KDD21")
UCR_ZIP = Path("data/ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
EXCLUDED = {32, 140}  # as in the authors' code


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return head + (" (with uncommitted changes)" if dirty else "")


def check(path: Path) -> dict:
    parts = path.stem.split("_")
    begin, end = int(parts[5]), int(parts[6])
    df = pd.read_csv(path, header=None)
    labels = df[1].to_numpy()
    runs = segments(labels > 0)
    first, last = (runs[0][0], runs[-1][1] - 1) if runs else (None, None)
    single = len(runs) == 1
    return {
        "id": int(parts[0]), "name": parts[3], "file": path.name, "rows": len(df), "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()), "train_end": int(parts[4]), "begin": begin, "end": end,
        "labelled_points": int((labels > 0).sum()), "segments": len(runs), "first": first, "last": last,
        "matches_archive": single and (first, last) == (begin - 1, end - 1),
        "matches_begin_end": (single and (first, last) == (begin, end - 1)) or (not runs and begin == end),
        # The training part must be anomaly-free: an anomaly starting before train_end breaks the split.
        "anomaly_in_train_part": bool(runs) and first < int(parts[4]),
    }


def compare_values(tsb_path: Path, archive_text: bytes) -> str:
    tsb = pd.read_csv(tsb_path, header=None)[0].to_numpy(dtype=float)
    ucr = np.array(archive_text.split(), dtype=float)  # one value per line, or several per line
    if len(tsb) != len(ucr):
        return f"length differs ({len(tsb)} vs {len(ucr)})"
    diff = np.abs(tsb - ucr)
    return "identical" if not diff.any() else f"{(diff > 0).sum()} values differ, max difference {diff.max():.3g}"


def main():
    rows = pd.DataFrame([check(p) for p in sorted(KDD21.glob("*.out"))])
    with zipfile.ZipFile(UCR_ZIP) as z:
        members = {int(n.split("/")[-1][:3]): n for n in z.namelist()
                   if "/UCR_Anomaly_FullData/" in n and n.endswith(".txt")}
        rows["values_vs_archive"] = [compare_values(KDD21 / f, z.read(members[i]))
                                     for f, i in zip(rows["file"], rows["id"])]
    rows["archive_name"] = rows["id"].map(lambda i: members[i].split("/")[-1][:-4])
    rows["same_name_as_archive"] = rows["file"].str[:-4] == rows["archive_name"]
    rows["in_study"] = ~rows["id"].isin(EXCLUDED)

    s = rows[rows["in_study"]]
    neither = s[~s["matches_archive"] & ~s["matches_begin_end"]]
    lines = [
        f"Produced by `scripts/check_labels.py` at commit {commit()}.",
        "",
        f"TSB-UAD KDD21 files: {len(rows)}; in the study (IDs 1–250 except 32 and 140): {len(s)}.",
        "",
        "| Check (248 series) | Count |",
        "|---|---|",
        f"| label column = file name, archive convention (1-based inclusive) | {s['matches_archive'].sum()} |",
        f"| label column = file name, 0-based [begin, end) | {s['matches_begin_end'].sum()} |",
        f"| matches neither | {len(neither)} |",
        f"| no labelled point | {(s['labelled_points'] == 0).sum()} |",
        f"| more than one labelled segment | {(s['segments'] > 1).sum()} |",
        f"| missing values | {(s['missing_values'] > 0).sum()} |",
        f"| not exactly 2 columns | {(s['columns'] != 2).sum()} |",
        f"| file name differs from the original archive's | {(~s['same_name_as_archive']).sum()} |",
        f"| labelled anomaly starts inside the training part | {s['anomaly_in_train_part'].sum()} |",
        f"| values identical to the archive's | {(s['values_vs_archive'] == 'identical').sum()} |",
        "",
        "One-point anomalies (begin = end in the file name):",
        "",
        "| ID | File | Labelled points | First | Last |",
        "|---|---|---|---|---|",
        *[f"| {r.id} | {r.file} | {r.labelled_points} | {r.first} | {r.last} |"
          for r in s[s["begin"] == s["end"]].itertuples()],
        "",
        "Series whose values differ from the archive's:",
        "",
        "| ID | File | Difference |",
        "|---|---|---|",
        *[f"| {r.id} | {r.file} | {r.values_vs_archive} |" for r in s[s["values_vs_archive"] != "identical"].itertuples()],
        "",
        "Series matching neither convention, named differently from the archive, or with the anomaly",
        "inside the training part:",
        "",
        "| ID | TSB-UAD file | Archive file | Labelled [first, last] | Training end | Anomaly in training part |",
        "|---|---|---|---|---|---|",
        *[f"| {r.id} | {r.file} | {r.archive_name} | [{r.first:.0f}, {r.last:.0f}] | {r.train_end} | "
          f"{'yes' if r.anomaly_in_train_part else 'no'} |"
          for r in s[(~s["matches_archive"] & ~s["matches_begin_end"]) | ~s["same_name_as_archive"]
                     | s["anomaly_in_train_part"]].itertuples()],
    ]
    Path("results").mkdir(exist_ok=True)
    rows.to_csv("results/label_check.csv", index=False)
    Path("results/label_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
