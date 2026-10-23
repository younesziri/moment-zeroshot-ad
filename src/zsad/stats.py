"""Statistics across series (analysis plan, section 6). Decisions use two-sided tests."""

import numpy as np
from scipy.stats import binomtest, wilcoxon


def bootstrap_ci(values: np.ndarray, groups: np.ndarray | None = None, n: int = 10_000,
                 seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the mean over series. With `groups`, whole source groups are
    resampled (clustered bootstrap), so related series move together."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if groups is None:
        means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    else:
        groups = np.asarray(groups)
        members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        means = []
        for _ in range(n):
            picked = rng.integers(0, len(members), len(members))
            means.append(values[np.concatenate([members[i] for i in picked])].mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def group_wilcoxon(values: np.ndarray, groups: np.ndarray, alternative: str = "two-sided") -> float:
    """Wilcoxon signed-rank test on the group means (each source group counts once). The plan's
    decisions use the two-sided test; "greater" gives the one-sided p for a positive mean."""
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    means = np.array([values[groups == g].mean() for g in np.unique(groups)])
    if not np.any(means != 0):
        return 1.0
    return float(wilcoxon(means, zero_method="wilcox", alternative=alternative).pvalue)


def poisson_binomial_test(hits: np.ndarray, probabilities: np.ndarray) -> float:
    """Exact test of the number of top-1 hits against independent series with hit probabilities
    |Z|/n (D2): two-sided, twice the smaller tail, capped at 1."""
    pmf = np.array([1.0])
    for p in probabilities:  # distribution of the sum, one series at a time
        pmf = np.r_[pmf * (1 - p), 0.0] + np.r_[0.0, pmf * p]
    k = int(np.sum(hits))
    return float(min(1.0, 2 * min(pmf[: k + 1].sum(), pmf[k:].sum())))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar test on paired yes/no outcomes (top-1 hits of two detectors on the same series)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if only_a + only_b == 0:
        return 1.0
    return float(binomtest(only_a, only_a + only_b, 0.5).pvalue)
