import numpy as np
import pytest

from deeppanel import PanelData


def test_shapes_and_props():
    y = np.random.randn(5, 8)
    X = np.random.randn(5, 8, 3)
    p = PanelData(y, X)
    assert (p.N, p.T, p.p) == (5, 8, 3)
    assert len(p.feature_names) == 3


def test_single_regressor_expands():
    p = PanelData(np.random.randn(4, 6), np.random.randn(4, 6))
    assert p.X.ndim == 3 and p.p == 1


def test_direct_pairs_alignment():
    y = np.arange(3 * 5).reshape(3, 5).astype(float)
    X = np.arange(3 * 5 * 2).reshape(3, 5, 2).astype(float)
    p = PanelData(y, X)
    feats, targ, uidx, tidx = p.direct_pairs(1)
    # every target period must be >= h and paired with x at t-1
    assert feats.shape == (3 * 4, 2)
    for k in range(len(targ)):
        i, t = uidx[k], tidx[k]
        assert np.allclose(feats[k], X[i, t - 1])
        assert targ[k] == y[i, t]


def test_demean_roundtrip():
    y = np.random.randn(4, 7) + np.array([[1], [2], [3], [4]])
    p = PanelData(y, np.random.randn(4, 7, 2))
    d = p.demean_y()
    assert np.allclose(d.y.mean(axis=1), 0.0, atol=1e-10)
    back = d.add_back_means(d.y)
    assert np.allclose(back, y)


def test_from_long():
    import pandas as pd
    rows = []
    for i in range(3):
        for t in range(4):
            rows.append({"id": i, "time": t, "y": i + t, "x1": i, "x2": t})
    df = pd.DataFrame(rows)
    p = PanelData.from_long(df, "id", "time", "y", ["x1", "x2"])
    assert (p.N, p.T, p.p) == (3, 4, 2)
    assert p.y[2, 3] == 5


def test_bad_horizon():
    p = PanelData(np.random.randn(3, 4), np.random.randn(3, 4, 1))
    with pytest.raises(ValueError):
        p.direct_pairs(10)
