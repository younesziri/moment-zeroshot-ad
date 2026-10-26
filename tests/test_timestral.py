from pathlib import Path

import numpy as np
import pytest

CACHED = (Path.home() / ".cache/huggingface/hub/models--mistralai--Timestral-5M").exists()
pytestmark = pytest.mark.skipif(not CACHED, reason="Timestral-5M weights not downloaded")


@pytest.fixture(scope="module")
def model():
    from zsad.timestral import load_timestral

    return load_timestral("5M", "cpu")


def test_shape_and_determinism(model):
    from zsad.timestral import timestral_scores

    t = np.arange(3000)
    x = np.sin(t / 10.0) + np.random.default_rng(0).normal(scale=0.1, size=3000)
    score, scaled = timestral_scores(model, x[:1000], x[1000:])  # 2000 points: not a multiple of 64
    assert score.shape == scaled.shape == (2000,)
    np.testing.assert_array_equal(score, timestral_scores(model, x[:1000], x[1000:])[0])
    assert np.median(score) < 0.1  # a noisy sine is easy to forecast


def test_causal(model):
    # Changing the series from point 1100 on can't change the scores of blocks that end before it.
    from zsad.timestral import timestral_scores

    x = np.sin(np.arange(1600) / 10.0)
    changed = x.copy()
    changed[1100:] += 5.0
    a = timestral_scores(model, x[:500], x[500:])[0]
    b = timestral_scores(model, changed[:500], changed[500:])[0]
    np.testing.assert_array_equal(a[:576], b[:576])  # 9 full blocks of 64 end at 1076
    assert not np.allclose(a[600:], b[600:])


def test_anomaly_scores_higher(model):
    from zsad.timestral import timestral_scores

    x = np.sin(np.arange(3000) / 10.0)
    x[2500:2520] += 3.0
    score = timestral_scores(model, x[:1000], x[1000:])[0]
    assert np.argmax(score) in range(1500, 1520)
