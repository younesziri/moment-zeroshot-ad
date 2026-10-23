from itertools import product

import numpy as np
import pytest

from zsad.stats import bootstrap_ci, group_wilcoxon, mcnemar_exact, poisson_binomial_test


def test_poisson_binomial_matches_brute_force():
    p = np.array([0.1, 0.5, 0.3, 0.8])
    pmf = np.zeros(5)
    for outcome in product([0, 1], repeat=4):  # enumerate every outcome of the 4 series
        pmf[sum(outcome)] += np.prod([q if o else 1 - q for q, o in zip(p, outcome)])
    for k in range(5):
        expected = min(1.0, 2 * min(pmf[: k + 1].sum(), pmf[k:].sum()))
        assert poisson_binomial_test(np.r_[np.ones(k), np.zeros(4 - k)], p) == pytest.approx(expected)


def test_mcnemar_counts_only_disagreements():
    a = np.array([1, 1, 1, 0, 0, 1])
    b = np.array([1, 0, 0, 0, 0, 1])  # a alone hits twice, b alone never
    assert mcnemar_exact(a, b) == pytest.approx(0.5)  # binomial(2, 0.5): P(2 or 0) = 0.5
    assert mcnemar_exact(a, a) == 1.0


def test_bootstrap_ci_contains_mean_and_clustering_widens_it():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 10)
    values = rng.normal(size=10)[groups] + 0.1 * rng.normal(size=100)  # strong group effect
    plain, clustered = bootstrap_ci(values, n=2000), bootstrap_ci(values, groups, n=2000)
    assert plain[0] < values.mean() < plain[1]
    assert clustered[1] - clustered[0] > plain[1] - plain[0]


def test_group_wilcoxon_uses_group_means():
    groups = np.repeat(np.arange(8), 3)
    values = np.repeat(np.arange(1.0, 9.0), 3)  # all group means positive
    assert group_wilcoxon(values, groups) < 0.05
    assert group_wilcoxon(np.zeros(24), groups) == 1.0
