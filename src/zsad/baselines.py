"""Baseline anomaly scores: one score per point, higher = more anomalous."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from zsad.data import unwindow, windows


def random_scores(n: int, kind: str, seed: int) -> np.ndarray:
    """Scores that carry no information. Both kinds are iid, so their rankings have the same
    distribution; only their scale differs (which matters for adjusted best F1's linear grid)."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(size=n)
    if kind == "squared_gaussian":
        return rng.normal(size=n) ** 2
    raise ValueError(f"unknown kind: {kind}")


def knn_scores(train: np.ndarray, x: np.ndarray, window: int = 512, k: int = 5) -> np.ndarray:
    """k-NN reconstruction, as in the authors' code (moment/models/anomaly_nearest_neighbors.py):
    each test window is reconstructed as the mean of its k nearest train windows (stride 1),
    and the score is the squared error per point. Test windows are the same non-overlapping
    windows as for MOMENT."""
    train_windows = np.lib.stride_tricks.sliding_window_view(train, window)
    nn = NearestNeighbors(n_neighbors=k).fit(train_windows)
    test_windows = windows(x, window)
    _, idx = nn.kneighbors(test_windows)
    reconstruction = train_windows[idx].mean(axis=1)
    return unwindow((test_windows - reconstruction) ** 2, len(x))
