"""Anomaly-detection metrics: the paper's adjusted best F1 and stricter alternatives.

All functions take one score per (downsampled) point; labels are 0/1.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each run of True values, end exclusive."""
    m = np.asarray(mask, dtype=bool).astype(int)
    edges = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """If any point of an anomaly segment is predicted, count the whole segment as predicted."""
    pred = np.asarray(pred, dtype=bool).copy()
    for start, end in segments(labels > 0):
        if pred[start:end].any():
            # momentfm's backward loop (range(i, 0, -1)) never reaches index 0, so index 0 is
            # only predicted if it was already. Copied to match momentfm exactly; it cannot
            # matter on UCR, where the anomaly never starts at the first test point.
            pred[max(start, 1) : end] = True
    return pred


def _f1_momentfm(pred: np.ndarray, labels: np.ndarray) -> float:
    # Same formula and epsilons as momentfm's f1_score, so results are bit-for-bit equal.
    actual = labels > 0
    tp = np.sum(pred & actual)
    fp = np.sum(pred & ~actual)
    fn = np.sum(~pred & actual)
    precision = tp / (tp + fp + 0.00001)
    recall = tp / (tp + fn + 0.00001)
    return 2 * precision * recall / (precision + recall + 0.00001)


def adjusted_best_f1(labels: np.ndarray, scores: np.ndarray, n_thresholds: int = 100) -> float:
    """The paper's metric (momentfm's adjbestf1): point-adjust, then the best F1 over
    `n_thresholds` thresholds spaced linearly from min to max score. Uses the labels to pick
    the threshold, and the linear grid makes it depend on the score scale, not only the ranking."""
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    return max(_f1_momentfm(point_adjust(scores >= t, labels), labels) for t in thresholds)


def best_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    """Best point-wise F1 over all thresholds, no point-adjust (still an oracle threshold)."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(f1.max())


def near_anomaly(labels: np.ndarray, tolerance: int) -> np.ndarray:
    """True within `tolerance` points of the labelled anomaly (UCR scoring, KDD Cup 2021:
    +-100 original points, i.e. +-10 points after x10 downsampling). One anomaly per series."""
    anomaly = np.flatnonzero(labels)
    near = np.zeros(len(labels), dtype=bool)
    near[max(anomaly[0] - tolerance, 0) : anomaly[-1] + tolerance + 1] = True
    return near


def ucr_top1(scores: np.ndarray, near: np.ndarray) -> bool:
    """UCR scoring: the single highest-scoring point must be near the anomaly."""
    return bool(near[np.argmax(scores)])


def top1_chance(near: np.ndarray) -> float:
    """Exact top-1 chance level |Z| / n, the same for every detector: over all n rotations of a
    score series, its maximum lands on every position once, so a hit has probability |Z| / n."""
    return float(np.mean(near))


def alert_metrics(scores: np.ndarray, labels: np.ndarray, tau: float, near: np.ndarray,
                  factor: int = 1) -> dict:
    """Alerting with a threshold `tau` fixed without labels: alert where score > tau.

    precision/recall: point-wise against the labels (no tolerance).
    detected: some alert is near the anomaly (within the UCR tolerance).
    false_alarms: runs of consecutive alerts that are entirely outside the tolerance range.
    delay: first alert near the anomaly minus the anomaly start, in original points
    (x `factor` after downsampling; negative = before the labelled start).
    """
    alert = scores > tau
    actual = labels > 0
    tp = np.sum(alert & actual)
    hits = np.flatnonzero(alert & near)
    return {
        "precision": tp / alert.sum() if alert.any() else float("nan"),
        "recall": tp / actual.sum(),
        "detected": bool(hits.size),
        "false_alarms": sum(not near[s:e].any() for s, e in segments(alert)),
        "delay": int((hits[0] - np.flatnonzero(actual)[0]) * factor) if hits.size else None,
    }


def p_value(value: float, null_values: np.ndarray) -> float:
    """Empirical p-value: share of random runs scoring at least as well, counting the method
    itself as one more draw so that p is never 0 (smallest value 1 / (N + 1))."""
    return (1 + np.sum(np.asarray(null_values) >= value)) / (1 + len(null_values))
