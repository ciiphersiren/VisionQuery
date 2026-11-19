import numpy as np

from dronekit import conformal as C


def test_quantile_rank_and_small_n():
    assert C.conformal_quantile(np.arange(1, 10), 0.1) == 9          # ceil(10 * 0.9) = 9th of 9
    assert C.conformal_quantile(np.arange(1, 9), 0.1) == np.inf      # n = 8: rank 9 does not exist


def test_interval_is_exactly_the_scores_within_q():
    for kind in C.SCORES:
        for yhat in (0, 3, 40):
            lo, hi = C.interval(np.array([yhat]), 1.3, kind)
            for y in range(0, 80):
                assert (lo[0] <= y <= hi[0]) == (C.score(y, yhat, kind) <= 1.3 + 1e-9)


def test_image_level_coverage_on_exchangeable_data():
    # target met within sampling error on iid data, over many calibration/test splits
    rng = np.random.default_rng(0)
    for alpha in (0.1, 0.2):
        cov = []
        for _ in range(300):
            lam = rng.uniform(1, 40, 700)
            y, yhat = rng.poisson(lam), rng.poisson(lam)
            s = C.score(y, yhat)
            q = C.calibrate(s[:200], alpha=alpha, method="image")
            lo, hi = C.interval(yhat[200:], q)
            cov.append(C.coverage(y[200:], lo, hi))
        assert abs(np.mean(cov) - (1 - alpha)) < 0.015
        assert np.mean(cov) >= 1 - alpha - 0.005                     # marginally valid, integer scores only make it wider


def _seq_data(rng, n_seq, n_frames=30, sd=1.0):
    # each sequence has its own bias: frames of a sequence are correlated, sequences are exchangeable
    y, yhat, g = [], [], []
    for i in range(n_seq):
        b = rng.normal(0, sd)
        lam = rng.uniform(5, 40, n_frames)
        t = rng.poisson(lam)
        y.append(t); yhat.append(np.maximum(np.round(t * np.exp(b * 0.3) + rng.normal(0, 1, n_frames)), 0)); g.append(np.full(n_frames, i))
    return np.concatenate(y), np.concatenate(yhat), np.concatenate(g)


def test_sequence_calibration_guarantee_and_image_baseline_undercovers():
    # sequence guarantee: with probability >= 1 - alpha a new sequence has >= 1 - alpha of its frames covered
    rng = np.random.default_rng(1)
    alpha, hit, img_cov, seq_cov = 0.1, [], [], []
    for _ in range(200):
        y, yhat, g = _seq_data(rng, 30)
        tst = g >= 19                                                # 19 calibration sequences are the fewest that allow alpha 0.1
        s = C.score(y, yhat)
        q = C.calibrate(s[~tst], g[~tst], alpha)
        lo, hi = C.interval(yhat[tst], q)
        hit += [v >= 1 - alpha for v in C.sequence_coverage(y[tst], lo, hi, g[tst]).values()]
        seq_cov.append(C.coverage(y[tst], lo, hi))
        qi = C.calibrate(s[~tst], alpha=alpha, method="image")
        lo, hi = C.interval(yhat[tst], qi)
        img_cov.append(np.mean([v >= 1 - alpha for v in C.sequence_coverage(y[tst], lo, hi, g[tst]).values()]))
    assert np.mean(hit) >= 1 - alpha - 0.02
    assert np.mean(img_cov) < np.mean(hit)                           # frame-level calibration does not hold per sequence


def test_mondrian_covers_each_stratum():
    rng = np.random.default_rng(2)
    n = 6000
    k = rng.integers(0, 2, n)
    y = rng.poisson(10, n)
    yhat = np.maximum(y + np.round(rng.normal(0, np.where(k == 1, 6, 1))), 0)
    s = C.score(y, yhat, "abs")
    q = C.mondrian(s[:3000], k[:3000], alpha=0.1, method="image")
    assert q[1] > q[0]
    for j in (0, 1):
        m = k[3000:] == j
        lo, hi = C.interval(yhat[3000:][m], q[j], "abs")
        assert C.coverage(y[3000:][m], lo, hi) >= 0.87              # stratum 0's errors are nearly all 0 or 1, so it covers well above target


def test_pooled_targets_frame_coverage_of_a_new_sequence():
    rng = np.random.default_rng(3)
    pooled, seq, width_p, width_s = [], [], [], []
    for _ in range(200):
        y, yhat, g = _seq_data(rng, 30)
        tst = g >= 19
        s = C.score(y, yhat)
        for m, cov, wid in (("pooled", pooled, width_p), ("sequence", seq, width_s)):
            lo, hi = C.interval(yhat[tst], C.calibrate(s[~tst], g[~tst], 0.1, m))
            cov.append(C.coverage(y[tst], lo, hi)); wid.append(C.mean_width(lo, hi))
    assert np.mean(pooled) >= 0.85 and np.mean(seq) >= np.mean(pooled) and np.mean(width_s) >= np.mean(width_p)
    assert C.pooled_quantile([1, 2, 3], [0, 0, 0], 0.1) == np.inf    # one sequence: +inf holds half the weight


def test_few_shot_draws_use_whole_sequences_first():
    draws = C.few_shot_draws({"a": 4, "b": 10, "c": 3}, 5, n_draws=30)
    for d in draws:
        assert sum(n for _, n in d) == 5 and len({s for s, _ in d}) == len(d)
    assert len({tuple(d) for d in draws}) > 1


def test_baselines_and_bootstrap():
    lo, hi = C.poisson_interval(np.array([0.0, 10.0]))
    assert lo[1] <= 10 <= hi[1] and hi[0] == 0
    lo, hi = C.residual_range(np.array([10.0]), [12, 8, 10, 9, 11], [10, 10, 10, 10, 10])
    assert lo[0] <= 10 <= hi[0]
    a, b = C.bootstrap_ci([1, 1, 3, 3], [0, 0, 1, 1], n=200)
    assert 1 <= a <= b <= 3
