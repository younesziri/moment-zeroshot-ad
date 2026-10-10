"""Cutting a series into fixed-length windows and stitching per-window outputs back."""

import numpy as np


def window_starts(length: int, window: int = 512) -> list[int]:
    """Start indices of non-overlapping windows covering [0, length).

    The last window is aligned to the end of the series, so it may overlap the previous one.
    No window is duplicated when `length` is a multiple of `window` (the official tutorial's
    de-duplication breaks in that case).
    """
    if length < window:
        raise ValueError(f"series of length {length} is shorter than the window ({window})")
    starts = list(range(0, length - window + 1, window))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def make_windows(x: np.ndarray, window: int = 512) -> tuple[np.ndarray, list[int]]:
    starts = window_starts(len(x), window)
    return np.stack([x[s : s + window] for s in starts]), starts


def stitch(per_window: np.ndarray, starts: list[int], length: int) -> np.ndarray:
    """Inverse of make_windows. Where the last window overlaps the previous one, the earlier
    window's values are kept (as in the tutorial), so every point has exactly one value."""
    window = per_window.shape[1]
    out = np.full(length, np.nan)
    for w, s in zip(per_window, starts):
        span = slice(s, s + window)
        todo = np.isnan(out[span])
        out[span][todo] = w[todo]
    assert not np.isnan(out).any()
    return out
