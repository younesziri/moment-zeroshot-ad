"""Loading a UCR series the way the authors' tutorial does (moment repo, tutorials/
anomaly_detection.ipynb and momentfm/data/anomaly_detection_dataset.py), as plain functions.

Differences from the tutorial, each deliberate:
  - the file has no header row, so we read it with header=None (the tutorial drops the first point);
  - the train part is split into fit and calibration, with the paper's ratio, so that k-NN has
    training data and the alert threshold can be set without labels;
  - optional x10 max-pooling of each split, as in the paper's research code (Table 22).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class UCRSeries:
    name: str
    values: np.ndarray  # float64, full series
    labels: np.ndarray  # int, 1 inside the anomaly
    train_end: int  # points [0, train_end) are guaranteed anomaly-free
    anomaly_start: int  # 0-based, inclusive
    anomaly_end: int  # 0-based, inclusive


def load_ucr(path: str | Path) -> UCRSeries:
    """Read a UCR Anomaly Archive file in the TSB-UAD format (value, label; no header).

    The file name encodes the split and the anomaly, e.g.
    198_UCR_Anomaly_tiltAPB2_50000_124159_124985.out -> train ends at 50000.
    The anomaly range is taken from the label column, not the name.
    """
    path = Path(path)
    df = pd.read_csv(path, header=None)  # no header row: header=None keeps the first point
    values = df[0].to_numpy(dtype=np.float64)
    labels = df[1].to_numpy().astype(int)
    parts = path.stem.split("_")
    anomaly = np.flatnonzero(labels)
    if anomaly.size == 0 or np.any(np.diff(anomaly) != 1):
        raise ValueError(f"{path.name}: expected exactly one contiguous anomaly")
    return UCRSeries(
        name=parts[3],
        values=values,
        labels=labels,
        train_end=int(parts[4]),
        anomaly_start=int(anomaly[0]),
        anomaly_end=int(anomaly[-1]),
    )


def downsample_max(values: np.ndarray, labels: np.ndarray, factor: int = 10):
    """Downsample by taking the max over blocks of `factor` points, as the authors' code does
    (moment-research, AnomalyDetectionDataset._downsample_timeseries).

    The series is left-padded with zeros to a multiple of `factor`. Like the original, the
    padding is a full block when the length is already a multiple of `factor`.

    Returns (values, labels, padding).
    """
    padding = factor - len(values) % factor
    v = np.pad(values, (padding, 0)).reshape(-1, factor).max(axis=1)
    y = np.pad(labels, (padding, 0)).reshape(-1, factor).max(axis=1)
    return v, y, padding


def prepare(path: str | Path, factor: int = 1) -> dict:
    """Scaled fit, calibration and test parts of a UCR series, plus the test labels.

    Scaling uses the mean and std of the whole train part [0, train_end), as the tutorial does.
    fit = [0, fit_end), calibration = [fit_end, train_end), test = [train_end, end), where
    fit_end = int(0.6 / (0.6 + 0.1) * train_end) is the split of the authors' zero-shot config.
    With factor > 1, each part is max-pooled separately, as in the paper's code.
    """
    s = load_ucr(path)
    return split_parts(s.values, s.labels, s.train_end, factor)


def split_parts(values: np.ndarray, labels: np.ndarray, train_end: int, factor: int) -> dict:
    """The body of `prepare`, for any series: scale with the whole train part, split into fit,
    calibration and test, max-pool each part separately when factor > 1."""
    train = values[:train_end]
    x = (values - train.mean()) / train.std()
    fit_end = int(0.6 / (0.6 + 0.1) * train_end)
    parts = {"fit": slice(0, fit_end), "cal": slice(fit_end, train_end), "test": slice(train_end, None)}
    out = {}
    for name, part in parts.items():
        v, y = x[part], labels[part]
        if factor > 1:
            v, y, _ = downsample_max(v, y, factor)
        out[name] = v
    out["labels"] = y  # test labels (fit and calibration are anomaly-free)
    return out


WINDOW = 512  # MOMENT's input length


def windows(x: np.ndarray, size: int = WINDOW) -> np.ndarray:
    """Non-overlapping windows of `size` points; the last window is aligned to the end of the
    series, so it may overlap the previous one (the tutorial's dataset does the same)."""
    if len(x) < size:
        raise ValueError(f"series of length {len(x)} is shorter than one window ({size})")
    starts = list(range(0, len(x) - size + 1, size))
    if starts[-1] + size < len(x):
        starts.append(len(x) - size)
    return np.stack([x[s : s + size] for s in starts])


def unwindow(per_window: np.ndarray, length: int) -> np.ndarray:
    """Back to one value per point, as the tutorial does: keep all full windows, then only the
    points of the last window that no earlier window covered. Unlike the tutorial's version,
    this also works when `length` is an exact multiple of the window size."""
    size = per_window.shape[1]
    n_full = length // size
    tail = length - n_full * size
    out = per_window[:n_full].ravel()
    return np.concatenate([out, per_window[-1, size - tail :]]) if tail else out
