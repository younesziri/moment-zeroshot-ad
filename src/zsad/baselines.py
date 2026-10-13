"""Baseline anomaly scores: one score per point, higher = more anomalous."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from zsad.windows import make_windows, stitch


def random_scores(n: int, kind: str, seed: int) -> np.ndarray:
    """Scores that carry no information. Both kinds are iid, so their rankings have the same
    distribution; only their scale differs (which matters for adjusted best F1's linear grid)."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(size=n)
    if kind == "squared_gaussian":
        return rng.normal(size=n) ** 2
    raise ValueError(f"unknown kind: {kind}")


def rolling_zscore(x: np.ndarray, window: int = 512) -> np.ndarray:
    """Squared z-score of each point against the `window` points before it (causal: the current
    point is not in its own reference window). The first `window` points are NaN."""
    s = pd.Series(x)
    past = s.shift(1).rolling(window)
    z = (s - past.mean()) / past.std()
    return (z**2).to_numpy()


def knn_scores(train: np.ndarray, x: np.ndarray, window: int = 512, k: int = 5) -> np.ndarray:
    """k-NN reconstruction, as in the authors' code (moment/models/anomaly_nearest_neighbors.py):
    each test window is reconstructed as the mean of its k nearest train windows (stride 1),
    and the score is the squared error per point. Test windows are the same non-overlapping
    windows as for MOMENT."""
    train_windows = np.lib.stride_tricks.sliding_window_view(train, window)
    nn = NearestNeighbors(n_neighbors=k).fit(train_windows)
    windows, starts = make_windows(x, window)
    _, idx = nn.kneighbors(windows)
    reconstruction = train_windows[idx].mean(axis=1)
    return stitch((windows - reconstruction) ** 2, starts, len(x))
