import numpy as np
import pytest

from zsad.baselines import knn_scores, random_scores, rolling_zscore


def test_random_scores_reproducible():
    for kind in ["uniform", "squared_gaussian"]:
        a = random_scores(100, kind, seed=1)
        np.testing.assert_array_equal(a, random_scores(100, kind, seed=1))
        assert not np.array_equal(a, random_scores(100, kind, seed=2))
    with pytest.raises(ValueError):
        random_scores(10, "normal", seed=0)


def test_rolling_zscore_is_causal():
    x = np.random.default_rng(0).normal(size=2000)
    future_changed = x.copy()
    future_changed[1500:] += 100.0
    a, b = rolling_zscore(x), rolling_zscore(future_changed)
    assert np.isnan(a[:512]).all() and not np.isnan(a[512:]).any()
    np.testing.assert_array_equal(a[:1500], b[:1500])  # past scores ignore the future
    assert b[1500] > 1000  # the jump itself is scored against the unchanged past


def test_rolling_zscore_value():
    x = np.zeros(5)
    x[[0, 2]] = 1.0  # window of 3 before index 3: [1, 0, 1], mean 2/3, std (ddof 1) sqrt(1/3)
    assert rolling_zscore(x, window=3)[3] == pytest.approx((2 / 3) ** 2 * 3)


def test_knn_scores_zero_on_seen_pattern_and_high_on_spike():
    pattern = np.random.default_rng(0).normal(size=50)
    train = np.tile(pattern, 40)  # 2000 points: every phase of the pattern appears many times
    x = np.tile(pattern, 22)[7:1107]  # 1100 points, same periodic signal, shifted phase
    np.testing.assert_allclose(knn_scores(train, x), 0.0, atol=1e-20)
    x_spike = x.copy()
    x_spike[800] += 10.0
    scores = knn_scores(train, x_spike)
    assert np.argmax(scores) == 800
