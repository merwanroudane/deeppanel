"""Forecast-accuracy metrics used in the papers."""
from __future__ import annotations

import numpy as np

__all__ = ["mse", "rmse", "pmse", "mae"]


def _clean(y_true, y_pred):
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def mse(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.mean((a - b) ** 2))


def rmse(y_true, y_pred) -> float:
    """Root mean squared forecast error (Chronopoulos et al. 2026)."""
    return float(np.sqrt(mse(y_true, y_pred)))


def pmse(y_true, y_pred) -> float:
    """Prediction mean squared error (Gao et al. 2026, Table 5)."""
    return mse(y_true, y_pred)


def mae(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))
