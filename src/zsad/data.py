"""Loading UCR Anomaly Archive series and the paper's downsampling."""

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

    Returns (values, labels, padding). Downsampled point i covers original indices
    [i * factor - padding, i * factor - padding + factor - 1] of the input.
    """
    padding = factor - len(values) % factor
    v = np.pad(values, (padding, 0)).reshape(-1, factor).max(axis=1)
    y = np.pad(labels, (padding, 0)).reshape(-1, factor).max(axis=1)
    return v, y, padding


def block_to_original(i: np.ndarray | int, factor: int, padding: int, offset: int = 0):
    """Original-index range [first, last] covered by downsampled point(s) i.
    `offset` is the original index where the downsampled segment started."""
    first = np.asarray(i) * factor - padding + offset
    return first, first + factor - 1
