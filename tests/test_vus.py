import numpy as np
import pytest

from zsad.vus_moment_research import generate_curve


def _vus_pr(labels, scores, window):
    return generate_curve(labels, scores, window)[7]  # outputs: ..., VUS-ROC, VUS-PR


def test_vus_pr_ranks_a_good_detector_above_random():
    labels = np.r_[np.zeros(2000), np.ones(300), np.zeros(700)].astype(int)
    rng = np.random.default_rng(0)
    good = _vus_pr(labels, labels + 0.01 * rng.uniform(size=3000), 600)
    random = _vus_pr(labels, rng.uniform(size=3000), 600)
    assert 0 <= random < good <= 1


def test_vus_raises_without_anomaly():
    # The authors' wrapper turns this error into NaN; we let it surface (analysis plan, D9).
    with pytest.raises(IndexError):
        generate_curve(np.zeros(3000, dtype=int), np.random.default_rng(0).uniform(size=3000), 600)
