import numpy as np
import torch

from deeppanel import MLP, TrainConfig
from deeppanel.training.trainer import train_mlp, predict
from deeppanel.training.normalize import Standardizer, MinMaxNormalizer
from deeppanel.inference.partial import partial_derivatives


def test_mlp_architecture_matches_paper_fig1():
    # P2 Fig.1: p=1, hidden [3,2], output 1  ->  11 weight connections
    net = MLP(p=1, depth=2, width=[3, 2], first_layer_bias=False)
    n_weights = sum(w.numel() for n, w in net.named_parameters() if "weight" in n)
    assert n_weights == 11


def test_mlp_forward_and_features():
    net = MLP(p=4, depth=3, width=8)
    x = torch.randn(10, 4)
    assert net(x).shape == (10,)
    assert net.features(x).shape == (10, 8)


def test_trainer_reduces_loss():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    a = rng.normal(size=3)
    y = np.sin(X @ a).astype(np.float32)
    Xs = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    net = MLP(p=3, depth=3, width=32)
    r = train_mlp(net, Xs, y, None, None,
                  TrainConfig(max_epochs=200, patience=10**9, lr=0.01, batch_size=64, seed=0))
    mse = np.mean((y - predict(net, Xs)) ** 2)
    assert mse < 0.1  # learns the single-index target well


def test_partial_derivatives_match_finite_diff():
    torch.manual_seed(0)
    net = MLP(p=4, depth=3, width=16, first_layer_bias=True).eval()
    X = np.random.default_rng(0).normal(size=(30, 4)).astype(np.float32)
    g = partial_derivatives(net, X)
    eps = 1e-3
    fd = np.zeros_like(g)
    with torch.no_grad():
        for j in range(4):
            Xp, Xm = X.copy(), X.copy()
            Xp[:, j] += eps
            Xm[:, j] -= eps
            fd[:, j] = (net(torch.tensor(Xp)).numpy() - net(torch.tensor(Xm)).numpy()) / (2 * eps)
    assert np.max(np.abs(g - fd)) < 1e-2


def test_normalizers_pooled_shape_and_inverse():
    X = np.random.default_rng(0).normal(size=(4, 6, 3))
    for Scaler in (Standardizer, MinMaxNormalizer):
        sc = Scaler().fit(X)
        feats = X.reshape(-1, 3)          # pooled (M, p) must scale correctly
        assert sc.transform(feats).shape == feats.shape
        # inverse of the transform on the first variable round-trips
        z0 = sc.transform(X)[..., 0].ravel()
        assert np.allclose(sc.inverse_target(z0, var=0), X[..., 0].ravel(), atol=1e-6)
