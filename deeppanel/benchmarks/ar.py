"""Autoregressive benchmark (unit-by-unit AR(p)).

Chronopoulos et al. (2026) use the AR(1) as their univariate benchmark and
produce ``h``-step forecasts by *iteration*.  This module fits an AR(``p``) for
each unit from its own history of the target and iterates it forward.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.panel import PanelData

__all__ = ["ARBenchmark"]


def _fit_ar(y: np.ndarray, p: int):
    """OLS AR(p) with intercept on a single series; returns (const, phi)."""
    T = y.size
    if T <= p + 1:
        return float(y.mean()), np.zeros(p)
    Y = y[p:]
    X = np.column_stack([np.ones(T - p)] + [y[p - k - 1:T - k - 1] for k in range(p)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(beta[0]), beta[1:]


class ARBenchmark:
    """Unit-by-unit AR(``p``) with iterated multi-step forecasts.

    Parameters
    ----------
    order : int, default 1
        Autoregressive order ``p`` (the paper's benchmark is AR(1)).
    """

    def __init__(self, order: int = 1) -> None:
        self.order = int(order)
        self._const = None
        self._phi = None
        self._history = None

    def fit(self, panel: PanelData) -> "ARBenchmark":
        p = self.order
        self._const = np.zeros(panel.N)
        self._phi = np.zeros((panel.N, p))
        for i in range(panel.N):
            c, phi = _fit_ar(panel.y[i], p)
            self._const[i] = c
            self._phi[i] = phi
        self._history = panel.y.copy()
        return self

    def forecast(self, steps: int) -> np.ndarray:
        """Iterated ``steps``-ahead forecast for every unit (shape ``(N,)``)."""
        if self._history is None:
            raise RuntimeError("Call fit() first.")
        p = self.order
        N = self._history.shape[0]
        out = np.empty(N)
        for i in range(N):
            hist = list(self._history[i, -p:]) if p > 0 else []
            val = self._history[i, -1]
            for _ in range(steps):
                lags = np.array(hist[-p:][::-1]) if p > 0 else np.zeros(0)
                if lags.size < p:
                    lags = np.concatenate([lags, np.zeros(p - lags.size)])
                val = self._const[i] + float(self._phi[i] @ lags)
                hist.append(val)
            out[i] = val
        return out
