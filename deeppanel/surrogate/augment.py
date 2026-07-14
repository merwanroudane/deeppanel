"""Surrogate model and residual construction (Gao et al. 2026, Section 3.1).

The LLM/BERT pipeline that turns social-media text into surrogate outcomes and
embeddings is external to this library (it needs GPT/BERT/OpenAI).  What this
module implements is the *statistical* surrogate step that consumes those
quantities:

Surrogate model (Eq. 3.2)::

    y^S_{i,t} = G_i( x^S_{i,t}, lags of y^S ) + eps^S_{i,t}

a per-region network ``G_i`` fit on the (aggregated) high-frequency surrogate
outcome ``y^S`` and its predictors, whose residuals ``eps_hat^S`` are the clean
auxiliary signal fed into the surrogate-augmented target model (Eq. 3.4).  Using
the *residual* rather than ``y^S`` itself is deliberate (Section 3.1): it avoids
the "shortcut learning" that collapses the target's structural response, and it
has far smaller variance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.mlp import MLP
from ..training.normalize import Standardizer
from ..training.trainer import TrainConfig, predict as _predict, train_mlp

__all__ = ["surrogate_residuals"]


def _lag_matrix(series: np.ndarray, n_lags: int) -> np.ndarray:
    """Return lagged columns (t-1..t-n_lags) with zero-padding at the start."""
    T = series.size
    L = np.zeros((T, n_lags))
    for k in range(1, n_lags + 1):
        L[k:, k - 1] = series[:-k]
    return L


def surrogate_residuals(
    y_surr: np.ndarray,
    x_surr: Optional[np.ndarray] = None,
    n_lags: int = 1,
    depth: int = 2,
    width: int = 16,
    config: Optional[TrainConfig] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Fit the per-region surrogate model and return its residuals.

    Parameters
    ----------
    y_surr : ndarray, shape (N, T)
        Aggregated surrogate outcome (e.g. the daily/monthly averaged inflation
        sentiment score) for each region and period.
    x_surr : ndarray, shape (N, T, ds), optional
        Surrogate predictors (e.g. aggregated text embeddings).  If omitted only
        lagged ``y^S`` is used.
    n_lags : int, default 1
        Number of lags of ``y^S`` to include (the ``q_k`` lags of Eq. 3.2).
    depth, width : network size for ``G_i``.
    config : TrainConfig, optional
    seed : int, optional

    Returns
    -------
    eps : ndarray, shape (N, T)
        Surrogate residuals ``eps_hat^S_{i,t}``.
    """
    y_surr = np.asarray(y_surr, dtype=float)
    N, T = y_surr.shape
    cfg = config or TrainConfig(max_epochs=800, seed=seed)
    eps = np.zeros((N, T))
    for i in range(N):
        feats = [_lag_matrix(y_surr[i], n_lags)]
        if x_surr is not None:
            feats.append(np.asarray(x_surr, dtype=float)[i])
        F = np.concatenate(feats, axis=1)
        sc = Standardizer().fit(F[None])           # treat as (1, T, q)
        Fs = sc.transform(F[None]).reshape(F.shape)
        if T < 5:
            eps[i] = y_surr[i] - y_surr[i].mean()
            continue
        model = MLP(p=F.shape[1], depth=depth, width=width)
        train_mlp(model, Fs, y_surr[i], None, None, cfg)
        pred = _predict(model, Fs)
        eps[i] = y_surr[i] - pred
    return eps
