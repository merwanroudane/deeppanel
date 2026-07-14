import numpy as np
import pytest

from deeppanel import DeepPooledPanel, LDPM, TrainConfig, surrogate_residuals
from deeppanel.benchmarks import ARBenchmark, PanelVAR, LinearPooledPanel, DeepTimeSeries
from deeppanel.datasets import simulate_pooled_panel, simulate_ldpm

FAST = TrainConfig(max_epochs=120, patience=20, lr=0.01, batch_size=64, seed=0)


def _panel():
    return simulate_pooled_panel(N=10, T=40, p=3, idio_scale=0.05, noise_sd=0.2,
                                 horizon=1, seed=0).panel


def test_deep_pooled_fit_predict_forecast():
    p = _panel()
    m = DeepPooledPanel(horizon=1, depth=2, width=16, config=FAST, seed=0).fit(p)
    feats, targ, uidx, _ = p.direct_pairs(1)
    pred = m.predict(feats, units=uidx)
    assert pred.shape == targ.shape
    # network beats the unconditional mean
    assert np.mean((targ - pred) ** 2) < np.var(targ)
    fc = m.forecast(p)
    assert fc.shape == (p.N,) and np.all(np.isfinite(fc))


def test_deep_pooled_idiosyncratic_predict_needs_units():
    p = _panel()
    m = DeepPooledPanel(horizon=1, depth=2, width=12, idiosyncratic=True,
                        config=FAST, seed=0).fit(p)
    with pytest.raises(ValueError):
        m.predict(p.direct_pairs(1)[0])  # no units -> error
    fc = m.forecast(p)
    assert fc.shape == (p.N,)


def test_partial_and_poolability_api():
    p = _panel()
    m = DeepPooledPanel(horizon=1, depth=2, width=16, config=FAST, seed=0).fit(p)
    d = m.partial_derivatives(p.X.reshape(-1, p.p))
    assert d.shape == (p.N * p.T, p.p)
    pt = m.poolability_test(p)
    assert np.isfinite(pt.statistic) and 0 <= pt.p_value <= 1


@pytest.mark.parametrize("Est", [ARBenchmark, PanelVAR, LinearPooledPanel, DeepTimeSeries])
def test_benchmarks_forecast_shape(Est):
    p = _panel()
    if Est is DeepTimeSeries:
        est = Est(horizon=1, depth=2, width=8, config=FAST, seed=0)
    elif Est is ARBenchmark:
        est = Est(1)
    elif Est is PanelVAR:
        est = Est(max_lags=2)
    else:
        est = Est(horizon=1)
    est.fit(p)
    try:
        f = est.forecast(p)
    except TypeError:
        f = est.forecast(1)
    assert f.shape == (p.N,) and np.all(np.isfinite(f))


def test_ldpm_group_recovery_and_conformal():
    s = simulate_ldpm(N=15, T=50, p=6, p_prime=4, n_groups=3, rho=0.6, seed=1)
    eps = surrogate_residuals(s.y_surrogate, s.X, n_lags=1, depth=2, width=10,
                              config=TrainConfig(max_epochs=120, seed=0))
    ld = LDPM(n_groups=3, d_h=6, depth=2, width=12, lam=0.05, alpha=0.1,
              warmup=80, epochs=120, lr=0.01, seed=0)
    ld.fit(s.y[:, :-1], s.X[:, :-1], surrogate_resid=eps[:, :-1], calibrate=True)
    assert ld.groups.shape == (s.N,)
    yhat, lo, hi = ld.forecast_interval(s.X[:, -1, :], surrogate_resid_last=eps[:, -1])
    assert yhat.shape == (s.N,) and np.all(hi >= lo)

    # group recovery clearly beats chance (1/3)
    from itertools import permutations
    K = 3
    acc = max(np.mean([p[e] for e in ld.groups] == s.groups) for p in permutations(range(K)))
    assert acc > 0.5
