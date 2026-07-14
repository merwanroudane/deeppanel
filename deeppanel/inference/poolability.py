"""Nonlinear poolability test (Chronopoulos et al. 2023, Remark 3).

Tests whether an idiosyncratic component is needed on top of the common one,
i.e. the null

    H0 :  h~_i(x_it) = h(x_it)   for all i        (pooling is enough)

Procedure (Remark 3).  Fit only the common component and form the residuals
``u_hat_it = y_it - g(x_it; theta_hat)``.  Regress, unit by unit, ``u_hat_it`` on
``f_i(x_it)`` to obtain unit-wise ``R^2_i``.  Then

    P = ( 1 / (sigma_hat * sqrt(N)) ) * sum_i ( T * R^2_i - m ),

with ``sigma_hat^2 = (1/N) sum_i (T R^2_i - m)^2`` and centering constant ``m``
(the dimension of ``f_i``).  Under ``H0`` (and cross-sectional independence),
``P`` is asymptotically standard normal, so large positive values reject pooling.

``f_i`` is, following the paper's "quasi-linear representation" (Eq. 6), taken by
default to be the network's last-hidden-layer features.  To reduce the influence
of estimated parameters (as the paper suggests), set ``split=True`` to estimate
the network and the unit regressions on disjoint time blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

__all__ = ["PoolabilityResult", "poolability_test"]


@dataclass
class PoolabilityResult:
    statistic: float
    p_value: float
    r2: np.ndarray            # per-unit R^2_i
    m: float                  # centering constant
    n_units: int

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"PoolabilityResult(P={self.statistic:.3f}, "
                f"p={self.p_value:.4f}, N={self.n_units})")


def _r2_on(residual: np.ndarray, feats: np.ndarray):
    """R^2 of regressing ``residual`` on ``feats``; also return columns used.

    Returns ``(r2, k_eff)`` where ``k_eff`` is the number of non-degenerate
    regressors actually used -- the correct centering ``E[T R^2] = k_eff`` under
    the null.
    """
    y = residual - residual.mean()
    F = feats - feats.mean(axis=0, keepdims=True)
    # drop (near-)constant / collinear columns to keep the design well conditioned
    keep = F.std(axis=0) > 1e-10
    F = F[:, keep]
    if F.shape[1] == 0:
        return 0.0, 0
    # effective rank for the centering constant
    k_eff = int(np.linalg.matrix_rank(F, tol=1e-8))
    beta, *_ = np.linalg.lstsq(F, y, rcond=None)
    fitted = F @ beta
    sst = float(y @ y)
    if sst <= 0:
        return 0.0, k_eff
    ssr = float((y - fitted) @ (y - fitted))
    return max(0.0, 1.0 - ssr / sst), k_eff


def poolability_test(
    estimator,
    panel,
    feature_map: str = "poly",
    poly_degree: int = 2,
    m: Optional[float] = None,
    split: bool = False,
) -> PoolabilityResult:
    """Run the nonlinear poolability test for a fitted :class:`DeepPooledPanel`.

    Parameters
    ----------
    estimator : DeepPooledPanel
        A fitted common-component estimator.
    panel : PanelData
        The panel the estimator was fit on (original units).
    feature_map : {'hidden','poly','linear'}, default 'hidden'
        Choice of ``f_i``: last-hidden-layer features, a polynomial of ``x``, or
        ``x`` itself.
    poly_degree : int, default 2
        Degree for ``feature_map='poly'``.
    m : float, optional
        Centering constant; defaults to the number of columns of ``f_i``.
    split : bool, default False
        If ``True`` use the first half of the periods for the residuals and the
        second half for the ``R^2`` regressions (reduces estimated-parameter bias).

    Returns
    -------
    PoolabilityResult
    """
    import torch

    f = estimator._check_fitted()
    model = f.common
    feat_scaler = f.feat_scaler
    h = estimator.horizon

    work = panel.demean_y() if estimator.demean else panel
    feats, targ, uidx, tidx = work.direct_pairs(h)
    Xs = feat_scaler.transform(feats)
    targ_s = f.targ_scaler.transform(targ[:, None]).ravel()

    # residuals on the transformed scale
    with torch.no_grad():
        gpred = model(torch.as_tensor(Xs, dtype=torch.float32)).numpy()
    resid = targ_s - gpred

    # build f_i(x)
    if feature_map == "hidden":
        with torch.no_grad():
            F = model.features(torch.as_tensor(Xs, dtype=torch.float32)).numpy()
    elif feature_map == "linear":
        F = Xs
    elif feature_map == "poly":
        cols = [Xs]
        for d in range(2, poly_degree + 1):
            cols.append(Xs ** d)
        F = np.concatenate(cols, axis=1)
    else:
        raise ValueError("feature_map must be 'hidden', 'poly' or 'linear'.")

    times = np.unique(tidx)
    if split:
        cut = times[len(times) // 2]
        reg_mask = tidx >= cut
    else:
        reg_mask = np.ones_like(tidx, dtype=bool)

    r2_list, weights, keffs = [], [], []
    for i in np.unique(uidx):
        sel = (uidx == i) & reg_mask
        if sel.sum() <= F.shape[1] + 1:
            continue
        Ti = int(sel.sum())
        r2, k_eff = _r2_on(resid[sel], F[sel])
        r2_list.append(Ti * r2)
        weights.append(Ti)
        keffs.append(k_eff)
    r2_arr = np.asarray(r2_list, dtype=float)
    N = len(r2_arr)
    if N < 2:
        raise ValueError("Too few units with enough observations for the test.")

    # centering: the effective number of regressors (E[T R^2] = k under the null)
    m_val = float(np.median(keffs) if m is None else m)
    centered = r2_arr - m_val
    sigma = np.sqrt(np.mean(centered ** 2))
    if sigma <= 0:
        sigma = 1e-12
    P = float(np.sum(centered) / (sigma * np.sqrt(N)))
    pval = float(2 * (1 - stats.norm.cdf(abs(P))))
    per_unit_r2 = (r2_arr / np.asarray(weights, dtype=float))
    return PoolabilityResult(statistic=P, p_value=pval, r2=per_unit_r2, m=m_val, n_units=N)
