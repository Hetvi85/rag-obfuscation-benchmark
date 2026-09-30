"""Metrics and statistics: Recall@k, bootstrap CIs, monotonicity tests."""
import numpy as np


def recall_at_k(ranks, k):
    """ranks: array of the retrieved rank of the adversarial passage per query
    (rank 1 = top; np.inf if not retrieved). Returns Recall@k = P(rank <= k)."""
    ranks = np.asarray(ranks, dtype=float)
    return float(np.mean(ranks <= k))


def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a per-query statistic."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = [stat(values[rng.integers(0, n, n)]) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(values)), float(lo), float(hi)


def spearman_trend(rhos, values):
    """Spearman rho between obfuscation intensity and a metric (monotonicity test
    for H1/H2). Returns (coefficient, n)."""
    rhos = np.asarray(rhos, dtype=float)
    values = np.asarray(values, dtype=float)
    def rank(a):
        order = a.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        return r
    rr, rv = rank(rhos), rank(values)
    if rr.std() == 0 or rv.std() == 0:
        return 0.0, len(rhos)
    return float(np.corrcoef(rr, rv)[0, 1]), len(rhos)


def interior_optimum(rhos, asr):
    """Locate rho* = argmax ASR and flag whether it is interior (not 0 or 1)."""
    rhos = np.asarray(rhos, dtype=float)
    asr = np.asarray(asr, dtype=float)
    i = int(np.argmax(asr))
    return float(rhos[i]), bool(0 < rhos[i] < 1), float(asr[i])
