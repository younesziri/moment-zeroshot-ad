"""Baseline anomaly scores: one score per point, higher = more anomalous."""

import numpy as np
import pandas as pd
import stumpy
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


def dominant_period(x: np.ndarray, low: int = 16, high: int = 512) -> tuple[int, bool]:
    """The pre-registered window L (analysis plan, section 2): the lag of the first local maximum of
    the autocorrelation after it first becomes negative, clipped to [low, high]. Returns (L, found);
    when there is no such maximum, L = high and found = False (the series is listed)."""
    x = x - x.mean()
    spectrum = np.fft.rfft(x, 2 * len(x))  # zero-padded, so the autocorrelation is not circular
    acf = np.fft.irfft(np.abs(spectrum) ** 2)[: len(x)]
    acf = acf / acf[0]
    negative = np.flatnonzero(acf < 0)
    if negative.size:
        k = np.arange(negative[0] + 1, len(acf) - 1)
        peaks = k[(acf[k] > acf[k - 1]) & (acf[k] >= acf[k + 1])]
        if peaks.size:
            return int(np.clip(peaks[0], low, high)), True
    return high, False


def _trailing(history: np.ndarray, x: np.ndarray, L: int) -> pd.core.window.Rolling:
    """Rolling window over the L points before each point of x (x[n] excluded), using `history` (the
    points just before x) for the first points."""
    return pd.Series(np.r_[history, x]).shift(1).rolling(L, min_periods=2)


def mu_sigma(history: np.ndarray, x: np.ndarray, L: int, eps: float) -> np.ndarray:
    """Course baseline (Lecture 5, p. 64): |x[n] - mean| / std over a trailing window of L points.
    A zero std (constant window) is replaced by eps (analysis plan, section 2)."""
    past = _trailing(history, x, L)
    mu, sd = past.mean().to_numpy(), past.std(ddof=0).to_numpy()
    return (np.abs(x - mu[len(history):]) / np.where(sd > 0, sd, eps)[len(history):])


def med_mad(history: np.ndarray, x: np.ndarray, L: int, eps: float) -> np.ndarray:
    """Course baseline (Lecture 5, p. 64): |x[n] - median| / MAD over a trailing window of L points
    (MAD unscaled; a zero MAD is replaced by eps)."""
    past = _trailing(history, x, L)
    med = past.median().to_numpy()
    mad = past.apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True).to_numpy()
    return (np.abs(x - med[len(history):]) / np.where(mad > 0, mad, eps)[len(history):])


def _max_over_covering(profile: np.ndarray, L: int) -> np.ndarray:
    """Per point, the largest profile value among the subsequences of length L that contain it."""
    padded = np.r_[profile, np.full(L - 1, -np.inf)]  # one value per point: n - L + 1 + L - 1
    return pd.Series(padded).rolling(L, min_periods=1).max().to_numpy()


def matrix_profile(fit: np.ndarray, x: np.ndarray, L: int) -> np.ndarray:
    """Course baseline (Lecture 5, p. 73; Yeh et al. 2016), as an AB-join: distance from each
    subsequence of x to its nearest subsequence of the fit part (z-normalised Euclidean)."""
    profile = stumpy.stump(x, L, T_B=fit, ignore_trivial=False)[:, 0].astype(float)
    return _max_over_covering(profile, L)


def matrix_profile_self(series: np.ndarray, L: int) -> np.ndarray:
    """Exploratory: the course's self-join over a whole series (exclusion zone set by stumpy)."""
    return _max_over_covering(stumpy.stump(series, L)[:, 0].astype(float), L)
