from pathlib import Path

import numpy as np
import pytest

CACHED = (Path.home() / ".cache/huggingface/hub/models--AutonLab--MOMENT-1-small").exists()
pytestmark = pytest.mark.skipif(not CACHED, reason="MOMENT-1-small weights not downloaded")


def test_moment_eval_mode_and_deterministic():
    from zsad.moment import load_moment, moment_scores

    model = load_moment("small", "cpu")
    assert not model.training  # dropout off
    x = np.sin(np.arange(1100) / 10.0) + np.random.default_rng(0).normal(scale=0.1, size=1100)
    scores, reconstruction = moment_scores(model, x, "cpu")
    assert scores.shape == reconstruction.shape == (1100,)
    np.testing.assert_array_equal(scores, moment_scores(model, x, "cpu")[0])
    np.testing.assert_allclose(scores, (x - reconstruction) ** 2)
