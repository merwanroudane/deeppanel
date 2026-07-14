"""LLM-powered Deep Panel Modeling -- LDPM (Gao, Sun, Wang, Liu & Hsiao 2026).

LDPM combines three pieces developed in the paper:

1. **Surrogate augmentation via a residual link** (Eqs. 3.1-3.4).  Rather than
   adding the surrogate outcome ``y^S`` to the target model directly (which would
   induce shortcut learning), LDPM feeds the *surrogate residual* ``eps_hat^S``
   as an extra input, so the target model becomes

       y_it = Q_i( x_it, z_it, eps^S_it ) + e_it,     Q_i = F_i + Gamma.

2. **Deep Panel Training with homogeneity pursuit** (Section 3.2-3.3): a shared
   feature map + unit heads regularised toward ``K0`` latent group centres by the
   classifier-LASSO product penalty (see :class:`~deeppanel.models.classo.DeepPanelTraining`).

3. **Within-group split conformal intervals** (Section 3.4).

The LLM/BERT text pipeline is external; ``eps^S`` is supplied by the caller (or
built with :func:`deeppanel.surrogate.surrogate_residuals`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..inference.conformal import ConformalCalibration, conformal_radii
from ..training.normalize import Standardizer, identity
from .classo import DeepPanelTraining

__all__ = ["LDPM"]


def _scaler(kind):
    if kind in (None, "none"):
        return identity()
    if kind in ("standardize", "standard", "zscore"):
        return Standardizer()
    raise ValueError(kind)


def _assemble(X, Z, eps_S):
    """Stack the design ``[x_it, z_it, eps^S_it]`` into (N, T, q)."""
    parts = [np.asarray(X, dtype=float)]
    if parts[0].ndim == 2:
        parts[0] = parts[0][:, :, None]
    if Z is not None:
        Z = np.asarray(Z, dtype=float)
        parts.append(Z[:, :, None] if Z.ndim == 2 else Z)
    if eps_S is not None:
        e = np.asarray(eps_S, dtype=float)
        parts.append(e[:, :, None] if e.ndim == 2 else e)
    return np.concatenate(parts, axis=2)


class LDPM:
    """LLM-powered Deep Panel Modeling estimator.

    Parameters
    ----------
    n_groups : int, default 3
        Number of latent groups ``K0`` for homogeneity pursuit.
    d_h, depth, width : shared feature-map size.
    lam : float, default 0.05
        Homogeneity-pursuit penalty.
    alpha : float, default 0.1
        Conformal miscoverage (``1 - alpha`` nominal coverage).
    feature_scale : {'standardize','none'}, default 'standardize'
        Transform for the design (fit on the training block only).
    calib_frac : float, default 0.2
        Fraction of the final periods reserved for conformal calibration.
    warmup, epochs, lr, seed : passed to Deep Panel Training.
    """

    def __init__(self, n_groups: int = 3, d_h: int = 8, depth: int = 3, width: int = 20,
                 lam: float = 0.05, alpha: float = 0.1, feature_scale: str = "standardize",
                 calib_frac: float = 0.2, warmup: int = 200, epochs: int = 400,
                 lr: float = 0.01, seed: Optional[int] = None) -> None:
        self.n_groups = int(n_groups)
        self.d_h = int(d_h)
        self.depth = int(depth)
        self.width = int(width)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.feature_scale = feature_scale
        self.calib_frac = float(calib_frac)
        self.warmup = int(warmup)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = seed
        self._dpt: Optional[DeepPanelTraining] = None
        self._scaler = None
        self._calib: Optional[ConformalCalibration] = None
        self._q = None

    # ------------------------------------------------------------------ fit
    def fit(self, y, X, Z=None, surrogate_resid=None, calibrate: bool = True) -> "LDPM":
        """Fit the surrogate-augmented DPT model and calibrate conformal radii.

        Parameters
        ----------
        y : ndarray, shape (N, T)
            Target (e.g. normalised regional CPI).
        X : ndarray, shape (N, T, dx)
            Aggregated text-embedding predictors.
        Z : ndarray, shape (N, T, dz), optional
            Macro covariates.
        surrogate_resid : ndarray, shape (N, T, ds), optional
            Surrogate residuals ``eps_hat^S`` (from
            :func:`deeppanel.surrogate.surrogate_residuals`).  If ``None`` the
            model reduces to Deep Panel Training on ``[x, z]`` alone.
        calibrate : bool, default True
            Whether to hold out the final periods for conformal calibration.
        """
        y = np.asarray(y, dtype=float)
        N, T = y.shape
        design = _assemble(X, Z, surrogate_resid)             # (N, T, q)
        self._q = design.shape[2]

        if calibrate:
            n_cal = max(1, int(round(self.calib_frac * T)))
            t_fit = T - n_cal
        else:
            t_fit = T

        scaler = _scaler(self.feature_scale).fit(design[:, :t_fit, :])
        Ds = scaler.transform(design)
        self._scaler = scaler

        dpt = DeepPanelTraining(
            n_groups=self.n_groups, d_h=self.d_h, depth=self.depth, width=self.width,
            lam=self.lam, warmup=self.warmup, epochs=self.epochs, lr=self.lr, seed=self.seed,
        )
        dpt.fit(y[:, :t_fit], Ds[:, :t_fit, :])
        self._dpt = dpt

        if calibrate and t_fit < T:
            scores, groups = [], []
            g = dpt.groups
            for i in range(N):
                for t in range(t_fit, T):
                    yhat = dpt.predict(i, Ds[i, t][None, :])[0]
                    scores.append(abs(y[i, t] - yhat))
                    groups.append(g[i])
            self._calib = conformal_radii(np.asarray(scores), np.asarray(groups), self.alpha)
        return self

    # -------------------------------------------------------------- predict
    def _check(self):
        if self._dpt is None:
            raise RuntimeError("Call fit() first.")

    @property
    def groups(self) -> np.ndarray:
        self._check()
        return self._dpt.groups

    def forecast(self, X_last, Z_last=None, surrogate_resid_last=None) -> np.ndarray:
        """One-step point forecast ``y_{i,T+1}`` for every unit (shape ``(N,)``)."""
        self._check()
        design = _assemble(
            np.asarray(X_last)[:, None, :] if np.asarray(X_last).ndim == 2 else X_last,
            None if Z_last is None else (np.asarray(Z_last)[:, None] if np.asarray(Z_last).ndim == 1 else np.asarray(Z_last)[:, None, :]),
            None if surrogate_resid_last is None else (np.asarray(surrogate_resid_last)[:, None]),
        )
        # design assembled as (N, 1, q); squeeze the time axis
        rows = self._scaler.transform(design)[:, 0, :]
        return self._dpt.forecast_last(rows)

    def forecast_interval(self, X_last, Z_last=None, surrogate_resid_last=None):
        """Return ``(yhat, lower, upper)`` with within-group conformal radii."""
        self._check()
        if self._calib is None:
            raise RuntimeError("Fit with calibrate=True to obtain conformal intervals.")
        yhat = self.forecast(X_last, Z_last, surrogate_resid_last)
        lo, hi = self._calib.interval(yhat, self.groups)
        return yhat, lo, hi

    @property
    def conformal(self) -> ConformalCalibration:
        if self._calib is None:
            raise RuntimeError("No calibration available.")
        return self._calib
