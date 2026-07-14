"""Giacomini and Rossi (2010) fluctuation test for forecast comparison in
unstable environments (used in Chronopoulos et al. 2026 to date *when* policy
became effective).

The test computes a *rolling* out-of-sample Diebold-Mariano statistic

    F_t = ( 1 / sigma_hat ) * ( 1 / sqrt(m) ) * sum_{j in window(t)} d_j

over rolling windows of length ``m`` of the loss differential ``d_t``, where
``sigma_hat^2`` is the HAC long-run variance over the whole out-of-sample path.
The null of equal accuracy *at every point in time* is rejected if
``max_t |F_t|`` exceeds a critical value that depends on ``mu = m / P`` (``P`` =
number of out-of-sample points).  A time-varying path lets the analyst see when
one model dominates.

The two-sided critical values are from Giacomini and Rossi (2010, Table 1) and
are interpolated in ``mu``; pass ``crit_value`` to override with your own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FluctuationResult", "fluctuation_test", "gr_critical_value"]

# Giacomini & Rossi (2010, Table 1) two-sided critical values, indexed by mu.
_MU = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
_CV = {
    0.10: np.array([3.176, 2.938, 2.770, 2.626, 2.500, 2.399, 2.297, 2.205, 2.107]),
    0.05: np.array([3.393, 3.179, 3.012, 2.890, 2.779, 2.634, 2.560, 2.433, 2.248]),
    0.01: np.array([3.804, 3.594, 3.420, 3.293, 3.193, 3.101, 3.008, 2.911, 2.727]),
}


def gr_critical_value(mu: float, alpha: float = 0.05) -> float:
    """Interpolated two-sided GR(2010) critical value for a given ``mu``."""
    if alpha not in _CV:
        raise ValueError(f"alpha must be one of {sorted(_CV)}.")
    mu = min(max(mu, _MU[0]), _MU[-1])
    return float(np.interp(mu, _MU, _CV[alpha]))


@dataclass
class FluctuationResult:
    statistic: float               # max_t |F_t|
    crit_value: float
    reject: bool
    alpha: float
    mu: float
    window: int
    centers: np.ndarray            # time index of each rolling window centre
    path: np.ndarray               # F_t path
    argmax_time: int

    def __repr__(self) -> str:  # pragma: no cover
        return (f"FluctuationResult(max|F|={self.statistic:.3f}, "
                f"cv={self.crit_value:.3f}, reject={self.reject})")


def fluctuation_test(
    e1,
    e2,
    window: float = 0.3,
    horizon: int = 1,
    loss: str = "squared",
    alpha: float = 0.05,
    crit_value: float = None,
):
    """Rolling (fluctuation) forecast-comparison test.

    Parameters
    ----------
    e1, e2 : array_like
        Forecast-error paths of the two models (time-ordered).
    window : float or int, default 0.3
        Rolling window size ``m``.  A float in ``(0, 1)`` is read as a fraction
        of the out-of-sample length ``P``; an int is used directly.
    horizon : int, default 1
        Forecast horizon (HAC lag ``h - 1`` for the long-run variance).
    loss : {'squared','absolute'}, default 'squared'
    alpha : {0.10, 0.05, 0.01}, default 0.05
    crit_value : float, optional
        Override the tabulated critical value.

    Returns
    -------
    FluctuationResult
    """
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()
    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'.")
    P = d.size
    m = int(round(window * P)) if 0 < window < 1 else int(window)
    m = max(4, min(m, P))
    mu = m / P

    dc = d - d.mean()
    n = d.size
    lrv = float(dc @ dc) / n
    for k in range(1, max(1, horizon)):
        if k >= n:
            break
        lrv += 2.0 * float(dc[k:] @ dc[:-k]) / n
    sigma = np.sqrt(max(lrv, 1e-12))

    centers, path = [], []
    for start in range(0, P - m + 1):
        seg = d[start:start + m]
        F = seg.sum() / (sigma * np.sqrt(m))
        path.append(F)
        centers.append(start + m // 2)
    path = np.asarray(path)
    centers = np.asarray(centers)

    stat = float(np.max(np.abs(path)))
    cv = gr_critical_value(mu, alpha) if crit_value is None else float(crit_value)
    arg = int(centers[int(np.argmax(np.abs(path)))]) if path.size else 0
    return FluctuationResult(
        statistic=stat, crit_value=cv, reject=stat > cv, alpha=alpha, mu=mu,
        window=m, centers=centers, path=path, argmax_time=arg,
    )
