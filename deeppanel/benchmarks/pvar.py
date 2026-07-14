"""Panel VAR benchmark (linear, pooled, with unit fixed effects).

The linear benchmark in Chronopoulos et al. (2023, 2026):

    z_it = mu_i + A_1 z_{i,t-1} + ... + A_q z_{i,t-q} + eps_it,

with ``z_it = (y_it, x_it')'``, unit intercepts ``mu_i`` and *homogeneous*
dynamics ``A_1, ..., A_q`` shared across units (the paper imposes homogeneity
"for parsimony").  The lag length ``q`` is chosen by BIC.  ``h``-step forecasts
of ``y`` are produced by iteration.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.panel import PanelData

__all__ = ["PanelVAR"]


def _build_z(panel: PanelData) -> np.ndarray:
    """Stack ``z_it = (y_it, x_it')'`` into an array of shape (N, T, k)."""
    y = panel.y[:, :, None]
    return np.concatenate([y, panel.X], axis=2)


class PanelVAR:
    """Pooled panel VAR with unit fixed effects and BIC lag selection.

    Parameters
    ----------
    max_lags : int, default 4
        Maximum lag order searched by BIC (paper: 1-4 for inflation, up to 28
        for COVID-19).
    lags : int, optional
        Fix the lag order and skip BIC selection.
    """

    def __init__(self, max_lags: int = 4, lags: Optional[int] = None) -> None:
        self.max_lags = int(max_lags)
        self.lags = lags
        self._A = None          # (q, k, k) coefficient blocks
        self._mu = None         # (N, k) unit intercepts
        self._history = None    # (N, T, k)
        self._q = None

    # -- estimation for a fixed q ------------------------------------------
    def _estimate(self, Z: np.ndarray, q: int):
        N, T, k = Z.shape
        rows_y, rows_x, unit_of = [], [], []
        for i in range(N):
            for t in range(q, T):
                lags = np.concatenate([Z[i, t - l - 1] for l in range(q)])  # (q*k,)
                rows_y.append(Z[i, t])
                rows_x.append(lags)
                unit_of.append(i)
        Y = np.asarray(rows_y)             # (M, k)
        Xlag = np.asarray(rows_x)          # (M, q*k)
        unit_of = np.asarray(unit_of)
        # unit dummies for fixed effects
        D = np.zeros((len(unit_of), N))
        D[np.arange(len(unit_of)), unit_of] = 1.0
        design = np.column_stack([D, Xlag])   # (M, N + q*k)
        beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
        mu = beta[:N]                      # (N, k)
        A = beta[N:].reshape(q, k, k)      # coefficient blocks (per lag)
        resid = Y - design @ beta
        return mu, A, resid, len(unit_of)

    def _bic(self, resid: np.ndarray, n_obs: int, n_params: int) -> float:
        k = resid.shape[1]
        sigma = (resid.T @ resid) / n_obs
        sign, logdet = np.linalg.slogdet(sigma + 1e-10 * np.eye(k))
        return float(logdet + n_params * np.log(n_obs) / n_obs)

    def fit(self, panel: PanelData) -> "PanelVAR":
        Z = _build_z(panel)
        N, T, k = Z.shape
        if self.lags is not None:
            best_q = int(self.lags)
        else:
            best_q, best_bic = 1, np.inf
            for q in range(1, min(self.max_lags, T - 2) + 1):
                mu, A, resid, m = self._estimate(Z, q)
                n_params = (N + q * k) * k
                bic = self._bic(resid, m, n_params)
                if bic < best_bic:
                    best_bic, best_q = bic, q
        mu, A, _, _ = self._estimate(Z, best_q)
        self._mu, self._A, self._q = mu, A, best_q
        self._history = Z.copy()
        return self

    def forecast(self, steps: int) -> np.ndarray:
        """Iterated ``steps``-ahead forecast of ``y`` for every unit (``(N,)``)."""
        if self._A is None:
            raise RuntimeError("Call fit() first.")
        N, T, k = self._history.shape
        q = self._q
        out = np.empty(N)
        for i in range(N):
            hist = [self._history[i, T - l - 1] for l in range(q)]  # most-recent first
            z = hist[0]
            for _ in range(steps):
                z = self._mu[i].copy()
                for l in range(q):
                    z = z + self._A[l] @ hist[l]
                hist = [z] + hist[:-1]
            out[i] = z[0]  # y is the first element of z
        return out
