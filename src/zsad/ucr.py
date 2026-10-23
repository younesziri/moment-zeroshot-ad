"""The 248 UCR series of the study (docs/analysis_plan.md), in its two configurations (D7):
"consistent" reads the original UCR archive; "paper-faithful" reads TSB-UAD with the research code's
rules (moment-research, commit 3ab637e, `_read_and_process_KDD21`)."""

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from zsad.data import WINDOW, downsample_max, split_parts

UCR_ZIP = Path("data/ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
KDD21 = Path("data/TSB-UAD-Public/KDD21")
STUDY_IDS = [i for i in range(1, 251) if i not in (32, 140)]  # as in the authors' code

# Stems that are variants of the same data (archive slides) or merged in the plan (section 1).
MERGES = {"TkeepFirstMARS": "TkeepMARS", "TkeepSecondMARS": "TkeepMARS", "TkeepThirdMARS": "TkeepMARS",
          "TkeepForthMARS": "TkeepMARS", "TkeepFifthMARS": "TkeepMARS", "ltstdbsAI": "ltstdbs",
          "ltstdbsAS": "ltstdbs", "ltstdbsES": "ltstdbs", "qtdbSelV": "qtdbSel", "qtdbSelMLII": "qtdbSel",
          "smML": "sm", "tiltAPB": "tilt", "tiltmtable": "tilt", "tilttable": "tilt", "gaitHunt": "gait",
          "insectEPG": "EPG", "LabCmacEPG": "EPG", "Italianpowerdemand": "PowerDemand",
          "CHARISfive": "CHARIS", "CHARISten": "CHARIS"}


def source_group(name: str) -> str:
    stem = re.sub(r"\d", "", name.replace("DISTORTED", "").replace("NOISE", ""))
    return MERGES.get(stem, stem)


def archive_members() -> dict[int, str]:
    with zipfile.ZipFile(UCR_ZIP) as z:
        return {int(n.split("/")[-1][:3]): n for n in z.namelist()
                if "/UCR_Anomaly_FullData/" in n and n.endswith(".txt")}


def load_archive(series_id: int, members: dict[int, str]) -> dict:
    """"Consistent": values from the archive file; training end and anomaly from its name, read with
    the archive's convention (1-based, inclusive)."""
    with zipfile.ZipFile(UCR_ZIP) as z:
        values = np.array(z.read(members[series_id]).split(), dtype=float)
    parts = members[series_id].split("/")[-1][:-4].split("_")
    labels = np.zeros(len(values), dtype=int)
    labels[int(parts[5]) - 1 : int(parts[6])] = 1
    return {"name": parts[3], "values": values, "labels": labels, "train_end": int(parts[4])}


def load_tsb(series_id: int) -> dict:
    """"Paper-faithful": the TSB-UAD file read as the research code reads it, without header=None,
    so the first row is lost; training end from the TSB-UAD file name; labels from its label column."""
    path = next(KDD21.glob(f"{series_id:03d}_*.out"))
    df = pd.read_csv(path, header=None).iloc[1:]
    parts = path.stem.split("_")
    return {"name": parts[3], "values": df[0].to_numpy(dtype=float), "labels": df[1].to_numpy().astype(int),
            "train_end": int(parts[4])}


def factor_for(length: int) -> int:
    """The paper's downsampling rule: x10 if length // 10 > 2560."""
    return 10 if length // 10 > 2560 else 1


def consistent_parts(s: dict) -> dict:
    """One downsampling factor per series, decided on the test part, applied to every part."""
    factor = factor_for(len(s["values"]) - s["train_end"])
    return {**split_parts(s["values"], s["labels"], s["train_end"], factor), "factor": factor}


def paper_parts(s: dict, fit_ratio: float) -> dict:
    """The research code's rules: scaler on the config's own train split (fit_ratio = 0.6/0.7 for
    the zero-shot config, 0.5/0.7 for k-NN), each part downsampled on its own decision, and only
    the first (n - 512) // 512 + 1 windows of the test part scored (the tail is dropped)."""
    values, labels, train_end = s["values"], s["labels"], s["train_end"]
    fit_end = int(fit_ratio * train_end)
    x = (values - values[:fit_end].mean()) / values[:fit_end].std()
    out = {}
    for name, part in {"fit": slice(0, fit_end), "cal": slice(fit_end, train_end),
                       "test": slice(train_end, None)}.items():
        v, y = x[part], labels[part]
        out[f"{name}_factor"] = factor_for(len(v))
        if out[f"{name}_factor"] > 1:
            v, y, _ = downsample_max(v, y, 10)
        out[name] = v
    n = len(out["test"])
    scored = ((n - WINDOW) // WINDOW + 1) * WINDOW if n >= WINDOW else 0
    out["labels"] = y[:scored]
    out["test"] = out["test"][:scored]
    return out


def with_context(score_fn, history: np.ndarray, part: np.ndarray, min_length: int = WINDOW) -> np.ndarray:
    """Score a part shorter than `min_length` with the preceding anomaly-free points as left context:
    the last `min_length` points ending at the part's end are scored, only the part's scores are kept
    (analysis plan, section 3, "consistent")."""
    if len(part) >= min_length:
        return score_fn(part)
    need = min_length - len(part)
    if len(history) < need:
        raise ValueError(f"not enough history ({len(history)} points) to score a part of {len(part)}")
    return score_fn(np.r_[history[-need:], part])[-len(part):]


def null_pairing(groups: dict[int, str], seed: int = 0) -> dict[int, int]:
    """D4: each series paired with one series from a different source group (fixed, seeded)."""
    rng = np.random.default_rng(seed)
    ids = sorted(groups)
    return {i: int(rng.choice([j for j in ids if groups[j] != groups[i]])) for i in ids}
