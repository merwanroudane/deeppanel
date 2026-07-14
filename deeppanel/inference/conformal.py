"""Split conformal prediction intervals, calibrated within latent groups
(Gao et al. 2026, Section 3.4).

Given a fitted predictor and a chronologically separated calibration set, the
absolute-residual nonconformity score is ``s_it = |y_it - y_hat_it|``.  Because
Deep Panel Training assigns each unit to a latent group ``g_hat(i)``, calibration
is done *within* each group: for group ``k`` with calibration scores
``A_k = {s_jt : g_hat(j) = k}`` of size ``m_k``, the conformal radius is

    q_k = s_{k, ( ceil( (m_k + 1)(1 - alpha) ) )}                        (order stat)

and the ``(1 - alpha)`` prediction set for a test point in group ``k`` is
``[ y_hat - q_k , y_hat + q_k ]``.  Grouping both exploits cross-sectional
information and, together with the variance reduction from surrogate
augmentation, yields tighter intervals than a target-only model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

__all__ = ["ConformalCalibration", "conformal_radii", "coverage"]


@dataclass
class ConformalCalibration:
    alpha: float
    radii: Dict[int, float]           # q_k per group
    global_radius: float              # fallback radius pooling all scores
    counts: Dict[int, int]

    def radius(self, group: int) -> float:
        return self.radii.get(int(group), self.global_radius)

    def interval(self, yhat: np.ndarray, groups: np.ndarray):
        """Return ``(lower, upper)`` arrays for point forecasts ``yhat``."""
        yhat = np.asarray(yhat, dtype=float)
        groups = np.asarray(groups, dtype=int)
        q = np.array([self.radius(g) for g in groups])
        return yhat - q, yhat + q


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    m = s.size
    if m == 0:
        return float("nan")
    rank = math.ceil((m + 1) * (1 - alpha))
    rank = min(max(rank, 1), m)          # clip to a valid order statistic
    return float(s[rank - 1])


def conformal_radii(scores: np.ndarray, groups: np.ndarray, alpha: float = 0.1) -> ConformalCalibration:
    """Compute within-group conformal radii from calibration scores.

    Parameters
    ----------
    scores : array_like
        Absolute-residual nonconformity scores on the calibration set.
    groups : array_like of int
        Group label ``g_hat`` for each calibration point.
    alpha : float, default 0.1
        Miscoverage level (``1 - alpha`` nominal coverage).
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=int)
    radii, counts = {}, {}
    for k in np.unique(groups):
        sel = groups == k
        radii[int(k)] = _conformal_quantile(scores[sel], alpha)
        counts[int(k)] = int(sel.sum())
    return ConformalCalibration(
        alpha=alpha, radii=radii,
        global_radius=_conformal_quantile(scores, alpha), counts=counts,
    )


def coverage(y_true, lower, upper) -> float:
    """Empirical coverage of prediction intervals."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))
