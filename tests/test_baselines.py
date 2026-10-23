import numpy as np
import pytest

from zsad.baselines import knn_scores, random_scores


def test_random_scores_reproducible():
    for kind in ["uniform", "squared_gaussian"]:
        a = random_scores(100, kind, seed=1)
        np.testing.assert_array_equal(a, random_scores(100, kind, seed=1))
        assert not np.array_equal(a, random_scores(100, kind, seed=2))
    with pytest.raises(ValueError):
        random_scores(10, "normal", seed=0)


def test_knn_scores_zero_on_seen_pattern_and_high_on_spike():
    pattern = np.random.default_rng(0).normal(size=50)
    train = np.tile(pattern, 40)  # 2000 points: every phase of the pattern appears many times
    x = np.tile(pattern, 22)[7:1107]  # 1100 points, same periodic signal, shifted phase
    np.testing.assert_allclose(knn_scores(train, x), 0.0, atol=1e-20)
    x_spike = x.copy()
    x_spike[800] += 10.0
    scores = knn_scores(train, x_spike)
    assert np.argmax(scores) == 800


def test_dominant_period():
    from zsad.baselines import dominant_period

    t = np.arange(5000)
    assert dominant_period(np.sin(2 * np.pi * t / 50)) == (50, True)
    assert dominant_period(np.sin(2 * np.pi * t / 2000))[0] == 512  # clipped to the upper bound
    assert dominant_period(np.sin(2 * np.pi * t / 5)) == (16, True)  # clipped to the lower bound
    assert dominant_period(np.arange(1000.0)) == (512, False)  # a trend: no peak, listed
