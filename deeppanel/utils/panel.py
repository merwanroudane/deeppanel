"""Panel-data container used across :mod:`deeppanel`.

The estimators in Chronopoulos et al. (2023, 2026) and Gao et al. (2026) all
operate on a balanced panel indexed by a cross-sectional unit ``i = 1, ..., N``
and a time period ``t = 1, ..., T``.  Internally we store

    y : ndarray, shape (N, T)          the target/outcome
    X : ndarray, shape (N, T, p)       the p regressors

which is the natural layout for the "stack all units at time ``t``" view used in
the papers (``X_t`` is an ``N x p`` matrix, Eq. (16) of Working Paper 23-15).

Direct multi-step forecasting.  The deep models forecast *directly* (Chronopoulos
et al. 2026, footnote 5): the ``h``-step target ``y_{i,t}`` is related to the
lagged features ``x_{i,t-h}`` and the estimated map is then applied to the known
time-``t`` features to obtain ``y_{i,t+h}``.  :meth:`PanelData.direct_pairs`
builds exactly those ``(x_{i,t-h}, y_{i,t})`` pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = ["PanelData"]


@dataclass
class PanelData:
    """A balanced panel ``(y, X)``.

    Parameters
    ----------
    y : array_like, shape (N, T)
        Outcome for each unit and period.
    X : array_like, shape (N, T, p)
        ``p`` regressors for each unit and period.
    units : sequence, optional
        Labels for the ``N`` cross-sectional units (default ``0..N-1``).
    times : sequence, optional
        Labels for the ``T`` periods (default ``0..T-1``).
    feature_names : sequence of str, optional
        Names of the ``p`` regressors (default ``x0..x{p-1}``).
    """

    y: np.ndarray
    X: np.ndarray
    units: Optional[Sequence] = None
    times: Optional[Sequence] = None
    feature_names: Optional[Sequence[str]] = None
    _unit_means: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y, dtype=float)
        self.X = np.asarray(self.X, dtype=float)
        if self.y.ndim != 2:
            raise ValueError("y must be 2-D with shape (N, T).")
        if self.X.ndim == 2:  # single regressor passed as (N, T)
            self.X = self.X[:, :, None]
        if self.X.ndim != 3:
            raise ValueError("X must be 3-D with shape (N, T, p).")
        N, T = self.y.shape
        if self.X.shape[:2] != (N, T):
            raise ValueError(
                f"X leading shape {self.X.shape[:2]} must match y shape {(N, T)}."
            )
        self.units = list(range(N)) if self.units is None else list(self.units)
        self.times = list(range(T)) if self.times is None else list(self.times)
        if self.feature_names is None:
            self.feature_names = [f"x{j}" for j in range(self.p)]
        else:
            self.feature_names = list(self.feature_names)
        if len(self.feature_names) != self.p:
            raise ValueError("feature_names length must equal the number of regressors p.")

    # -- basic dimensions ---------------------------------------------------
    @property
    def N(self) -> int:
        return self.y.shape[0]

    @property
    def T(self) -> int:
        return self.y.shape[1]

    @property
    def p(self) -> int:
        return self.X.shape[2]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PanelData(N={self.N}, T={self.T}, p={self.p})"

    # -- construction helpers ----------------------------------------------
    @classmethod
    def from_long(
        cls,
        df,
        unit: str,
        time: str,
        y: str,
        x: Sequence[str],
    ) -> "PanelData":
        """Build a :class:`PanelData` from a long (tidy) ``pandas`` frame.

        The panel must be balanced: every unit must appear in every period.
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError("from_long expects a pandas.DataFrame.")
        x = list(x)
        units = list(pd.unique(df[unit]))
        times = list(pd.unique(df[time]))
        N, T, p = len(units), len(times), len(x)
        u_pos = {u: k for k, u in enumerate(units)}
        t_pos = {tt: k for k, tt in enumerate(times)}
        Y = np.full((N, T), np.nan)
        XX = np.full((N, T, p), np.nan)
        for _, row in df.iterrows():
            i, t = u_pos[row[unit]], t_pos[row[time]]
            Y[i, t] = row[y]
            XX[i, t, :] = [row[c] for c in x]
        if np.isnan(Y).any() or np.isnan(XX).any():
            raise ValueError("Panel is unbalanced or has missing values; balance it first.")
        return cls(Y, XX, units=units, times=times, feature_names=x)

    # -- transformations ----------------------------------------------------
    def demean_y(self) -> "PanelData":
        """Return a copy with ``y`` demeaned unit-by-unit.

        Implements the ``E(y_{it}) = 0`` normalisation of Chronopoulos et al.
        (2023, p. 6; 2026, p. 4): "achieved via simple unit-by-unit demeaning
        of the dependent variable."  The removed means are stored so forecasts
        can be returned on the original scale via :meth:`add_back_means`.
        """
        mu = self.y.mean(axis=1, keepdims=True)
        out = PanelData(
            self.y - mu, self.X, units=self.units, times=self.times,
            feature_names=self.feature_names,
        )
        out._unit_means = mu.ravel().copy()
        return out

    @property
    def unit_means(self) -> Optional[np.ndarray]:
        return self._unit_means

    def add_back_means(self, yhat: np.ndarray) -> np.ndarray:
        """Add the stored unit means back onto a forecast of shape (N, ...)."""
        if self._unit_means is None:
            return yhat
        mu = self._unit_means.reshape((-1,) + (1,) * (yhat.ndim - 1))
        return yhat + mu

    def direct_pairs(self, h: int):
        """Return direct ``h``-step training pairs.

        Builds, for every ``t`` with ``t - h >= 0``,

            features  x_{i, t-h}   (shape (M, p))
            targets   y_{i, t}     (shape (M,))

        stacked over all units ``i`` and valid periods, where ``M = N*(T-h)``.
        Also returns the unit index and the target-time index of each row so
        callers can regroup by unit (e.g. for the idiosyncratic step).

        Returns
        -------
        feats : ndarray, shape (M, p)
        targ  : ndarray, shape (M,)
        uidx  : ndarray, shape (M,)   unit index in ``0..N-1``
        tidx  : ndarray, shape (M,)   target-period index in ``h..T-1``
        """
        if h < 1:
            raise ValueError("Forecast horizon h must be >= 1.")
        if h >= self.T:
            raise ValueError(f"h={h} too large for T={self.T}.")
        feats, targ, uidx, tidx = [], [], [], []
        for i in range(self.N):
            for t in range(h, self.T):
                feats.append(self.X[i, t - h, :])
                targ.append(self.y[i, t])
                uidx.append(i)
                tidx.append(t)
        return (
            np.asarray(feats, dtype=float),
            np.asarray(targ, dtype=float),
            np.asarray(uidx, dtype=int),
            np.asarray(tidx, dtype=int),
        )

    def last_features(self) -> np.ndarray:
        """Time-``T`` regressor matrix ``X_T`` of shape (N, p).

        Used as the input for the direct ``h``-step-ahead forecast of
        ``y_{i,T+h}`` (Chronopoulos et al. 2026, footnote 5).
        """
        return self.X[:, -1, :].copy()

    def slice_time(self, start: int, stop: int) -> "PanelData":
        """Return the sub-panel over periods ``[start, stop)`` (all units)."""
        return PanelData(
            self.y[:, start:stop], self.X[:, start:stop, :],
            units=self.units, times=self.times[start:stop],
            feature_names=self.feature_names,
        )
