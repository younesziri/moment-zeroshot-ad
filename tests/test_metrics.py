import numpy as np
import pytest
from momentfm.utils.anomaly_detection_metrics import adjbestf1

from zsad.data import block_to_original
from zsad.metrics import (
    adjusted_best_f1, alert_metrics, best_f1, near_anomaly, point_adjust, segments, ucr_top1,
)


def test_segments():
    assert segments(np.array([1, 1, 0, 0, 1, 0, 1])) == [(0, 2), (4, 5), (6, 7)]
    assert segments(np.zeros(5)) == []


def test_point_adjust_hand_example():
    labels = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1])
    # First segment is hit once -> fully predicted; second segment is missed -> unchanged.
    np.testing.assert_array_equal(point_adjust(pred, labels), [0, 1, 1, 1, 0, 0, 0, 0, 1])


def test_point_adjust_index_zero_quirk():
    # momentfm never fills index 0 retroactively; we copy that.
    np.testing.assert_array_equal(point_adjust(np.array([0, 1, 0]), np.array([1, 1, 0])), [0, 1, 0])


def _labels(n, ranges):
    y = np.zeros(n, dtype=int)
    for s, e in ranges:
        y[s:e] = 1
    return y


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("ranges", [[(700, 783)], [(0, 20), (300, 340), (900, 1000)], [(999, 1000)]])
def test_adjusted_best_f1_equals_momentfm(seed, ranges):
    rng = np.random.default_rng(seed)
    labels = _labels(1000, ranges)
    for scores in [rng.uniform(size=1000), rng.normal(size=1000) ** 2,
                   rng.integers(0, 5, size=1000).astype(float)]:  # ties
        assert adjusted_best_f1(labels, scores) == adjbestf1(labels, scores)


def test_adjusted_best_f1_constant_scores():
    labels = _labels(100, [(40, 50)])
    scores = np.ones(100)
    assert adjusted_best_f1(labels, scores) == adjbestf1(labels, scores)


def test_best_f1_perfect_and_no_point_adjust():
    labels = _labels(100, [(40, 50)])
    scores = labels.astype(float)
    assert best_f1(labels, scores) == 1.0
    # One lucky point inside the anomaly: point-adjust gives 1.0 (up to momentfm's epsilons),
    # point-wise F1 stays low (1 TP among 10 positives: F1 = 2/11).
    lucky = np.zeros(100)
    lucky[45] = 1.0
    assert adjusted_best_f1(labels, lucky) > 0.999
    assert best_f1(labels, lucky) == pytest.approx(2 / 11)


def test_near_anomaly_tolerance_edges():
    # Blocks of 10, no padding: anomaly at original [1000, 1009], tolerance range [900, 1109].
    first, last = block_to_original(np.arange(200), 10, 0)
    near = near_anomaly(first, last, 1000, 1009)
    assert not near[89] and near[90]  # [890, 899] is outside, [900, 909] touches 900
    assert near[110] and not near[111]  # [1100, 1109] touches 1109, [1110, 1119] is outside
    scores = np.zeros(200)
    scores[90] = 1.0
    assert ucr_top1(scores, near)
    scores[89] = 2.0
    assert not ucr_top1(scores, near)


def test_alert_metrics_hand_example():
    first, last = block_to_original(np.arange(200), 10, 0)
    near = near_anomaly(first, last, 1000, 1009)
    labels = np.zeros(200, dtype=int)
    labels[100] = 1
    scores = np.zeros(200)
    scores[[5, 6, 30, 95, 96, 100]] = 1.0  # two far events, one early event near, one hit
    m = alert_metrics(scores, labels, tau=0.5, near=near, first=first, anomaly_start=1000)
    assert m["detected"]
    assert m["false_alarms"] == 2  # [5, 6] and [30]; [95, 96] is within the tolerance
    assert m["delay"] == 950 - 1000  # first alert near the anomaly is block 95
    assert m["precision"] == pytest.approx(1 / 6)
    assert m["recall"] == 1.0


def test_alert_metrics_no_alert():
    first, last = block_to_original(np.arange(200), 10, 0)
    near = near_anomaly(first, last, 1000, 1009)
    labels = np.zeros(200, dtype=int)
    labels[100] = 1
    m = alert_metrics(np.zeros(200), labels, tau=0.5, near=near, first=first, anomaly_start=1000)
    assert not m["detected"] and m["false_alarms"] == 0 and m["delay"] is None
    assert np.isnan(m["precision"]) and m["recall"] == 0.0
