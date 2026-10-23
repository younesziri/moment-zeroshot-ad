from pathlib import Path

import numpy as np
import pytest

CACHED = (Path.home() / ".cache/huggingface/hub/models--AutonLab--MOMENT-1-small").exists()
pytestmark = pytest.mark.skipif(not CACHED, reason="MOMENT-1-small weights not downloaded")


def test_moment_eval_mode_and_deterministic():
    from zsad.moment import load_moment, moment_scores

    model = load_moment("small", "cpu")
    assert not model.training  # dropout off
    assert model.patch_embedding.add_positional_embedding  # the released weights need it
    x = np.sin(np.arange(1100) / 10.0) + np.random.default_rng(0).normal(scale=0.1, size=1100)
    scores, reconstruction = moment_scores(model, x, "cpu")
    assert scores.shape == reconstruction.shape == (1100,)
    np.testing.assert_array_equal(scores, moment_scores(model, x, "cpu")[0])
    np.testing.assert_allclose(scores, (x - reconstruction) ** 2)


def test_positional_embedding_off_takes_effect():
    from zsad.moment import load_moment, moment_scores

    off = load_moment("small", "cpu", positional_embedding=False)
    assert not off.patch_embedding.add_positional_embedding
    assert not hasattr(off.patch_embedding, "position_embedding")  # the layer is not even built
    x = np.sin(np.arange(1100) / 10.0)
    on_reconstruction = moment_scores(load_moment("small", "cpu"), x, "cpu")[1]
    assert not np.allclose(moment_scores(off, x, "cpu")[1], on_reconstruction)


def test_checkpoint_loads_strictly():
    # huggingface-hub 0.24 loads with strict=False by default, which would silently leave any
    # mismatched layer at its random initialisation.
    from momentfm import MOMENTPipeline

    MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small", strict=True,
                                   model_kwargs={"task_name": "reconstruction"})


def test_masked_scores_deterministic_and_shaped():
    from zsad.moment import load_moment, masked_scores

    model = load_moment("small", "cpu")
    x = np.sin(np.arange(1100) / 10.0)
    scores, reconstruction = masked_scores(model, x, "cpu")
    assert scores.shape == reconstruction.shape == (1100,)
    np.testing.assert_array_equal(scores, masked_scores(model, x, "cpu")[0])
