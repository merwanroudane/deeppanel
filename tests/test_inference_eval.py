import numpy as np

from deeppanel import diebold_mariano, fluctuation_test, conformal_radii, coverage, rmse
from deeppanel.evaluation.fluctuation import gr_critical_value
from deeppanel.inference.conformal import _conformal_quantile
from deeppanel.surrogate import svd_reduce, max_pool, mean_pool


def test_dm_detects_more_accurate_model():
    rng = np.random.default_rng(0)
    e1 = rng.normal(0, 1.0, 200)     # more accurate
    e2 = rng.normal(0, 1.6, 200)     # less accurate
    r = diebold_mariano(e1, e2, horizon=1, alternative="less")
    assert r.p_value < 0.05 and r.statistic < 0


def test_dm_symmetry_pvalue_bounds():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 100)
    r = diebold_mariano(e, e.copy() + 1e-9, horizon=1)
    assert 0 <= r.p_value <= 1


def test_fluctuation_runs_and_cv_monotone():
    rng = np.random.default_rng(0)
    e1 = rng.normal(0, 1, 150)
    e2 = rng.normal(0, 1.3, 150)
    fr = fluctuation_test(e1, e2, window=0.3, horizon=1)
    assert np.isfinite(fr.statistic) and fr.crit_value > 0
    # critical value decreases in mu
    assert gr_critical_value(0.1, 0.05) > gr_critical_value(0.9, 0.05)


def test_conformal_quantile_formula():
    # for m scores and alpha, rank = ceil((m+1)(1-alpha))
    s = np.arange(1, 11).astype(float)  # 1..10
    q = _conformal_quantile(s, alpha=0.1)   # ceil(11*0.9)=10 -> s[9]=10
    assert q == 10.0
    q2 = _conformal_quantile(s, alpha=0.5)  # ceil(11*0.5)=6 -> s[5]=6
    assert q2 == 6.0


def test_conformal_coverage_marginal():
    rng = np.random.default_rng(0)
    scores = np.abs(rng.normal(size=500))
    groups = rng.integers(0, 3, size=500)
    cal = conformal_radii(scores, groups, alpha=0.1)
    # new draw, same distribution -> coverage near nominal
    new = np.abs(rng.normal(size=2000))
    g = rng.integers(0, 3, size=2000)
    hi = np.array([cal.radius(gg) for gg in g])
    lo = -hi
    cov = coverage(new, lo, hi)      # |normal| >= 0 > lo, so this is upper coverage
    assert cov > 0.85


def test_svd_and_pooling():
    X = np.random.default_rng(0).normal(size=(60, 20))
    XV, V = svd_reduce(X, 5)
    assert XV.shape == (60, 5) and V.shape == (20, 5)
    assert np.allclose(max_pool([np.array([1.0, 0]), np.array([0.0, 2])]), [1.0, 2.0])
    assert mean_pool([1, 2, 3]) == 2.0


def test_rmse_metric():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert abs(rmse([0, 0], [1, 1]) - 1.0) < 1e-12
