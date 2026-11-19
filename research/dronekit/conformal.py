"""Conformal intervals for per-image object counts, calibrated by video sequence.

Point count yhat from detections at a dev-tuned threshold; score s(y, yhat) measures the count error.

Exchangeability: frames of one video are not exchangeable, but whole sequences of a split can be treated as exchangeable (each is a
separate flight). So the calibration unit is the sequence. Three methods:
- `sequence` (primary): each sequence contributes one score, its `seq_q` within-sequence quantile (default 1 - alpha). If a test sequence
  is exchangeable with the calibration sequences, then with probability >= 1 - alpha at least a `seq_q` share of its frames are covered.
  Rigorous and conservative: it needs ceil((K+1)(1-alpha)) <= K, so at least 9 sequences for alpha 0.1 and 4 for alpha 0.2.
- `pooled` (secondary): every sequence gets the same total weight, plus one unit of weight at +inf, and the threshold is the weighted
  1 - alpha quantile (hierarchical conformal prediction, Lee, Barber and Willett). Targets the coverage of a random frame of a new sequence;
  the worst-case bound is 1 - 2 alpha, not 1 - alpha.
- `image` (baseline): every frame exchangeable. Expected to under-cover.
"""
from __future__ import annotations

import numpy as np

SCORES = ("abs", "sqrt")      # |y - yhat| and |y - yhat| / sqrt(yhat + 1)
METHODS = ("sequence", "pooled", "image")


def score(y, yhat, kind="sqrt"):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = np.abs(y - yhat)
    return e if kind == "abs" else e / np.sqrt(yhat + 1)


def interval(yhat, q, kind="sqrt"):
    """Integer count interval [lo, hi] holding every y with score(y, yhat) <= q. Lower end is at least 0."""
    yhat = np.asarray(yhat, float)
    r = q if kind == "abs" else q * np.sqrt(yhat + 1)
    return np.maximum(np.ceil(yhat - r - 1e-9), 0), np.floor(yhat + r + 1e-9)


def conformal_quantile(scores, alpha):
    """The ceil((n+1)(1-alpha))-th smallest score; +inf when n is too small for that rank to exist."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[k - 1]) if 1 <= k <= n else float("inf")


def sequence_scores(s, groups, seq_q):
    """One score per sequence: the `seq_q` quantile of its frame scores (upper order statistic, so never interpolated down)."""
    s, groups = np.asarray(s, float), np.asarray(groups)
    return np.array([np.quantile(s[groups == g], seq_q, method="higher") for g in np.unique(groups)], float)


def pooled_quantile(s, groups, alpha):
    """Weighted 1 - alpha quantile: each of K sequences has total weight 1/(K+1), split over its frames; +inf holds the last 1/(K+1)."""
    s, g = np.asarray(s, float), np.asarray(groups)
    uniq, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / (cnt[inv] * (len(uniq) + 1))
    o = np.argsort(s)
    c = np.cumsum(w[o])
    i = np.searchsorted(c, (1 - alpha) - 1e-12)
    return float(s[o[i]]) if i < len(s) else float("inf")


def calibrate(s, groups=None, alpha=0.1, method="sequence", seq_q=None):
    """Threshold q for one of METHODS."""
    if method == "sequence":
        return conformal_quantile(sequence_scores(s, groups, 1 - alpha if seq_q is None else seq_q), alpha)
    if method == "pooled":
        return pooled_quantile(s, groups, alpha)
    return conformal_quantile(s, alpha)


def mondrian(s, strata, groups=None, alpha=0.1, method="sequence", seq_q=None):
    """{stratum: q}, calibrated separately in each stratum. A stratum whose calibration data cannot support alpha gets +inf, which
    is reported as an uninformative interval rather than hidden by borrowing from other strata."""
    s, strata = np.asarray(s, float), np.asarray(strata)
    g = None if groups is None else np.asarray(groups)
    out = {}
    for k in np.unique(strata):
        m = strata == k
        out[k.item() if hasattr(k, "item") else k] = calibrate(s[m], None if g is None else g[m], alpha, method, seq_q)
    return out


def poisson_interval(yhat, alpha=0.1):
    """Baseline: central Poisson interval with mean yhat (no coverage guarantee)."""
    from scipy.stats import poisson
    yhat = np.asarray(yhat, float)
    mu = np.maximum(yhat, 1e-9)
    return poisson.ppf(alpha / 2, mu), poisson.ppf(1 - alpha / 2, mu)


def residual_range(yhat, cal_y, cal_yhat, lo=0.1, hi=0.9, eps=1.0):
    """Baseline: the 10th-90th percentile of relative residuals on calibration records, applied to yhat (no coverage guarantee)."""
    rel = (np.asarray(cal_y, float) - np.asarray(cal_yhat, float)) / (np.asarray(cal_yhat, float) + eps)
    a, b = np.quantile(rel, [lo, hi])
    yhat = np.asarray(yhat, float)
    return np.maximum(np.ceil(yhat + a * (yhat + eps) - 1e-9), 0), np.floor(yhat + b * (yhat + eps) + 1e-9)


def coverage(y, lo, hi):
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi):
    """Mean of hi - lo. Infinite intervals make it infinite, on purpose."""
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))


def sequence_coverage(y, lo, hi, groups):
    """Coverage per sequence: {sequence: coverage}. Pooled coverage can hide sequences that fail."""
    cov = (np.asarray(y, float) >= lo) & (np.asarray(y, float) <= hi)
    g = np.asarray(groups)
    return {k: float(cov[g == k].mean()) for k in np.unique(g)}


def bootstrap_ci(values, groups, stat=np.mean, n=2000, seed=0, level=0.95):
    """Percentile interval from resampling whole sequences."""
    v, g = np.asarray(values, float), np.asarray(groups)
    uniq = np.unique(g)
    parts = [v[g == u] for u in uniq]
    rng = np.random.default_rng(seed)
    b = [stat(np.concatenate([parts[i] for i in rng.integers(0, len(uniq), len(uniq))])) for _ in range(n)]
    return tuple(np.quantile(b, [(1 - level) / 2, (1 + level) / 2]))


def few_shot_draws(frames_by_seq, k, n_draws=20, seed=0):
    """Draws of about k labeled target frames taken from whole sequences, in random sequence order: sequences are added until at
    least k frames are held, and only the first k frames of the last sequence are used. Returns one list of (sequence, n frames)
    per draw. Whole sequences are then excluded from evaluation in that draw."""
    rng = np.random.default_rng(seed)
    seqs = sorted(frames_by_seq)
    out = []
    for _ in range(n_draws):
        pick, left = [], k
        for i in rng.permutation(len(seqs)):
            if left <= 0:
                break
            s = seqs[i]
            t = min(left, frames_by_seq[s])
            pick.append((s, t))
            left -= t
        out.append(pick)
    return out
