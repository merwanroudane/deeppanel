"""Diebold and Mariano (1995) test of equal predictive accuracy.

Used throughout Chronopoulos et al. (2026) to judge whether RMSE differences
across models are significant.  We include the Harvey, Leybourne and Newbold
(1997) small-sample correction, which is standard practice.

The loss-differential series is ``d_t = L(e1_t) - L(e2_t)`` for a loss ``L``
(squared by default).  A negative mean loss differential means model 1 is more
accurate.  The long-run variance uses a Newey-West HAC estimator with a
truncation lag tied to the forecast horizon ``h`` (``h - 1`` autocovariances,
the usual choice for ``h``-step forecasts).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["DMResult", "diebold_mariano"]


@dataclass
class DMResult:
    statistic: float
    p_value: float
    mean_loss_diff: float
    horizon: int
    n: int
    alternative: str

    def __repr__(self) -> str:  # pragma: no cover
        return (f"DMResult(DM={self.statistic:.3f}, p={self.p_value:.4f}, "
                f"mean_dloss={self.mean_loss_diff:.4g})")


def _loss(e: np.ndarray, kind: str, power: float) -> np.ndarray:
    if kind == "squared":
        return e ** 2
    if kind == "absolute":
        return np.abs(e)
    if kind == "power":
        return np.abs(e) ** power
    raise ValueError("loss must be 'squared', 'absolute' or 'power'.")


def diebold_mariano(
    e1,
    e2,
    horizon: int = 1,
    loss: str = "squared",
    power: float = 2.0,
    alternative: str = "two-sided",
    harvey: bool = True,
) -> DMResult:
    """Diebold-Mariano test comparing two forecast-error series.

    Parameters
    ----------
    e1, e2 : array_like
        Forecast errors ``y - yhat`` of model 1 and model 2 (same length).
    horizon : int, default 1
        Forecast horizon ``h``; sets the HAC truncation lag to ``h - 1``.
    loss : {'squared','absolute','power'}, default 'squared'
    alternative : {'two-sided','less','greater'}, default 'two-sided'
        ``'less'`` tests H1: model 1 more accurate (mean loss diff < 0).
    harvey : bool, default True
        Apply the Harvey-Leybourne-Newbold small-sample correction.

    Returns
    -------
    DMResult
    """
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()
    ok = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[ok], e2[ok]
    d = _loss(e1, loss, power) - _loss(e2, loss, power)
    n = d.size
    if n < 4:
        raise ValueError("Need at least 4 paired forecast errors for the DM test.")
    dbar = float(d.mean())

    # Newey-West long-run variance with h-1 lags
    dc = d - dbar
    gamma0 = float(dc @ dc) / n
    lrv = gamma0
    for k in range(1, max(1, horizon)):
        if k >= n:
            break
        cov = float(dc[k:] @ dc[:-k]) / n
        lrv += 2.0 * cov  # rectangular (Diebold-Mariano) window
    lrv = max(lrv, 1e-12)

    dm = dbar / np.sqrt(lrv / n)

    if harvey:
        corr = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        dm *= corr
        dist = stats.t(df=n - 1)
    else:
        dist = stats.norm

    if alternative == "two-sided":
        pval = float(2 * dist.sf(abs(dm)))
    elif alternative == "less":       # H1: model 1 more accurate (d < 0)
        pval = float(dist.cdf(dm))
    elif alternative == "greater":
        pval = float(dist.sf(dm))
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'.")

    return DMResult(statistic=float(dm), p_value=pval, mean_loss_diff=dbar,
                    horizon=int(horizon), n=int(n), alternative=alternative)
